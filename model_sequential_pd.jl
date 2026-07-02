include("model_full_pd.jl")

using Dates
 
function default_weights(P)
    (; I, L, PTDF, gen_bus_idx) = P
    pos = [Dict(i => PTDF[l, gen_bus_idx[i]] for i in I) for l in L]
    neg = [Dict(i => -PTDF[l, gen_bus_idx[i]] for i in I) for l in L]
    return vcat(pos, neg)
end
 
# STAGE 1, iterated over ALL exploration weights under ONE shared wall-clock

function build_seq_stage1_sweep(P; weights = default_weights(P), total_time_limit = 3600.0,
                                tag = "seqpd_s1", log_dir = ".", threads = 0,
                                per_run_time_limit = total_time_limit)
    (; I, IJ, vc, T, W_t, fc) = P
    rows = DataFrame(w_idx=Int[], status=String[], solve_time=Float64[], mip_gap=Float64[],
                     n_qnz=Int[], q=Vector{Any}(), c=Vector{Any}(), ls=Vector{Any}())
    t_start = now()
    n_done = 0
    for (k, w) in enumerate(weights)
        elapsed = (now() - t_start).value / 1000.0         
        remaining = total_time_limit - elapsed
        if remaining <= 0.0
            @info "stage1 sweep: time budget exhausted after $(n_done)/$(length(weights)) weights"
            break
        end
        tl = min(per_run_time_limit, remaining)
 
        m = direct_model(Gurobi.Optimizer(GRB_ENV))
        set_optimizer_attribute(m, "NonConvex", 2)
        set_optimizer_attribute(m, "TimeLimit", tl)
        threads > 0 && set_optimizer_attribute(m, "Threads", threads)
        add_log!(m, "$(tag)_w$(k)", log_dir)
        V = add_market_cap_feas!(m, P)
        add_strong_duality!(m, P, V)
        @objective(m, Max, sum(w[i]*V.c[i] for i in I))     
        optimize!(m)
        n_done += 1
 
        gap = try relative_gap(m) catch; NaN end
        qnz = try MOI.get(m, Gurobi.ModelAttribute("NumQCNZs")) catch; -1 end
        if has_values(m)
            push!(rows, (k, string(termination_status(m)), solve_time(m), gap, qnz,
                        Dict((t,g)=>value(V.q[t,g]) for t in T, g in IJ),
                        Dict(i=>value(V.c[i]) for i in I),
                        Dict(t=>value(V.ls[t]) for t in T)))
        else
            push!(rows, (k, string(termination_status(m)), solve_time(m), gap, qnz, missing, missing, missing))
        end
        @printf("  w%-3d  %-12s  t=%6.1fs  gap=%6.3f  (budget left=%6.1fs)\n",
                k, string(termination_status(m)), solve_time(m), gap, total_time_limit-elapsed-solve_time(m))
    end
    return rows
end
 
function solve_redispatch_lp(P, q_vals, c_vals, ls_vals;
                             time_limit = 3600.0, tag = "redispatch_lp", log_dir = ".", threads = 0)
    (; N, L, PTDF, Fmax, IJ, I, vc, cap, a, T, W_t, gen_bus_idx, D_max, d_nodal, BUS, VOLL, hmk) = P
    rd = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(rd, "TimeLimit", time_limit)
    threads > 0 && set_optimizer_attribute(rd, "Threads", threads)
    add_log!(rd, tag, log_dir)
    q_bar = Dict((t,g) => (g in I ? c_vals[g] : cap[g]*a[(t,g)]) for t in T, g in IJ)
    d_n   = Dict((t,n) => d_nodal[(t,BUS[n])]*D_max[t] for t in T, n in N)
    serv  = Dict((t,n) => d_n[(t,n)] - d_nodal[(t,BUS[n])]*ls_vals[t] for t in T, n in N)
 
    @variable(rd, r_up[t in T, g in IJ] >= 0)
    @variable(rd, r_down[t in T, g in IJ] >= 0)
    @variable(rd, curt[t in T, n in N] >= 0)
    @variable(rd, p_flow[t in T, l in L])
    @expression(rd, p_inj[t in T, n in N],
        sum(q_vals[(t,g)] + r_up[t,g] - r_down[t,g] for g in IJ if gen_bus_idx[g]==n; init=0.0) - d_n[(t,n)] + curt[t,n])
    @constraint(rd, [t in T, g in IJ], r_up[t,g]   - max(0.0, q_bar[(t,g)] - q_vals[(t,g)]) <= 0)
    @constraint(rd, [t in T, g in IJ], r_down[t,g] - max(0.0, q_vals[(t,g)])                <= 0)
    @constraint(rd, [t in T, n in N],  curt[t,n]   - max(0.0, serv[(t,n)])                  <= 0)
    @constraint(rd, [t in T, l in L],  p_flow[t,l] - Fmax[l] <= 0)
    @constraint(rd, [t in T, l in L], -p_flow[t,l] - Fmax[l] <= 0)
    @constraint(rd, [t in T, l in L],  p_flow[t,l] - sum(PTDF[l,n]*p_inj[t,n] for n in N) == 0)
    @constraint(rd, [t in T], sum(r_up[t,g]-r_down[t,g] for g in IJ) + sum(curt[t,n] for n in N) == 0)
    @objective(rd, Min,
        sum(W_t[t]*( sum((vc[g]+hmk)*r_up[t,g] + (-vc[g]+hmk)*r_down[t,g] for g in IJ) +
                     VOLL*sum(curt[t,n] for n in N) ) for t in T))
    optimize!(rd)
    rd_cost_econ = has_values(rd) ?
        sum(W_t[t]*( sum(vc[g]*(value(r_up[t,g])-value(r_down[t,g])) for g in IJ) +
                     VOLL*sum(value(curt[t,n]) for n in N) ) for t in T) : NaN
    return rd, rd_cost_econ
end