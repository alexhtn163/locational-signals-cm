include("model_sequential_pd.jl")      

function build_milp_stage1(P; w = nothing, time_limit = 180.0,
                           tag = "milp_s1", log_dir = ".", threads = 0)
    (; N, BUS, I, J, IJ, vc, cap, fc, a, alpha, gen_to_zone, T, W_t, D_max, d_nodal,
       Z, I_in_z, J_in_z, d_c_z, d_c, mec_pairs, pair_keys, VOLL, PTDF, gen_bus_idx) = P
    Wmax = maximum(values(W_t)); Wsum = sum(values(W_t))
    mec_bound_z = Dict(z => sum((mec_pairs[p] for p in pair_keys if p[1]==z || p[2]==z); init=0.0) for z in Z)
    # default objective = your first PTDF exploration weight (a representative single run)
    if w === nothing
        w = Dict(i => PTDF[L_first(P), gen_bus_idx[i]] for i in I)
    end

    # ── K constants (your formulas; energy-side ones RESCALED by W_t) ────────
    K_q   = maximum(cap[g] for g in IJ; init = 0.0)
    K_ls  = maximum(D_max[t] for t in T; init = 0.0)
    K_c   = maximum(cap[i] for i in I; init = 0.0)
    K_cm  = maximum(cap[g] for g in IJ; init = 0.0)
    K_dem = maximum((d_c_z[z] + mec_bound_z[z] for z in Z); init = 0.0)
    K_sys = sum(d_c_z[z] + mec_bound_z[z] for z in Z; init = 0.0)
    K_slack_q_I      = maximum(cap[i] for i in I; init = 0.0)
    K_slack_q_J      = maximum(cap[j]*a[(t,j)] for t in T for j in J; init = 0.0)
    K_slack_cap      = maximum(cap[i] for i in I; init = 0.0)
    K_slack_cm_bud_I = maximum(cap[i] for i in I; init = 0.0)
    K_slack_cm_bud_J = maximum(alpha[j]*cap[j] for j in J; init = 0.0)
    lambda_c_ub = maximum(fc[i] for i in I)
    P_cap_cm    = 2.0*lambda_c_ub
    cap_up_ub   = VOLL*Wsum + lambda_c_ub                 # was VOLL*length(T); now W_t-scaled
    K_mu_g_I    = VOLL*Wmax;  K_mu_g_J = VOLL*Wmax         # mu_g_up ~ W_t*(le-vc)
    K_mu_cm_bud_I = P_cap_cm; K_mu_cm_bud_J = P_cap_cm
    K_mu_cap    = cap_up_ub
    K_nu_c = P_cap_cm; K_phi = P_cap_cm; K_rho = P_cap_cm
    K_stat_q_I = Wmax*(maximum(vc[i] for i in I; init=0.0) + VOLL)   # W_t-scaled
    K_stat_q_J = Wmax*(maximum(vc[j] for j in J; init=0.0) + VOLL)
    K_stat_c   = maximum(fc[g] for g in IJ; init=0.0) + cap_up_ub + 2*lambda_c_ub
    K_stat_cm = P_cap_cm; K_stat_dem_c = P_cap_cm
    K_stat_ls = VOLL*Wmax                                 # W_t*(VOLL-le)

    milp = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(milp, "TimeLimit",      time_limit)
    set_optimizer_attribute(milp, "Presolve",        2)
    set_optimizer_attribute(milp, "FeasibilityTol", 1e-3)
    set_optimizer_attribute(milp, "MIPGap",         0.01)
    set_optimizer_attribute(milp, "DualReductions",  0)
    threads > 0 && set_optimizer_attribute(milp, "Threads", threads)
    add_log!(milp, tag, log_dir)

    # ── Primal variables ────────────────────────────────────────────────────
    @variable(milp, ls[t in T] >= 0)
    @variable(milp, q[t in T, g in IJ] >= 0)
    @variable(milp, c[i in I] >= 0)
    @variable(milp, c_cm_I[i in I] >= 0)
    @variable(milp, c_cm_J[j in J] >= 0)
    @variable(milp, c_dem[z in Z] >= 0)
    @variable(milp, f_trans[p in pair_keys])
    @expression(milp, net_import[z in Z],
        sum(f_trans[p] for p in pair_keys if p[1]==z; init=0.0) -
        sum(f_trans[p] for p in pair_keys if p[2]==z; init=0.0))
    @expression(milp, q_max_I[t in T, i in I], c[i])
    @expression(milp, q_max_J[t in T, j in J], cap[j]*a[(t,j)])

    # ── Primal constraints ──────────────────────────────────────────────────
    @constraint(milp, [t in T], ls[t] <= D_max[t])
    @constraint(milp, [t in T], sum(q[t,g] for g in IJ) + ls[t] == D_max[t])
    @constraint(milp, [t in T, i in I], q[t,i] <= q_max_I[t,i])
    @constraint(milp, [t in T, j in J], q[t,j] <= q_max_J[t,j])
    @constraint(milp, [i in I], c[i] <= cap[i])
    @constraint(milp, [i in I], c_cm_I[i] <= alpha[i]*c[i])
    @constraint(milp, [j in J], c_cm_J[j] <= alpha[j]*cap[j])
    @constraint(milp, [p in pair_keys], f_trans[p] + mec_pairs[p] >= 0)
    @constraint(milp, [p in pair_keys], mec_pairs[p] - f_trans[p] >= 0)
    @constraint(milp, [z in Z], d_c_z[z] - net_import[z] >= 0)
    @constraint(milp, [z in Z], c_dem[z] - d_c_z[z] + net_import[z] >= 0)
    @constraint(milp, [z in Z],
        sum(c_cm_I[i] for i in I_in_z[z]; init=0.0) +
        sum(c_cm_J[j] for j in J_in_z[z]; init=0.0) - c_dem[z] == 0)
    @constraint(milp, sum(c_dem[z] for z in Z) >= d_c)

    # ── Dual variables  ──────────────────────
    @variable(milp, 0 <= lambda_e[t in T]          <= VOLL)
    @variable(milp, 0 <= lambda_c[z in Z]          <= P_cap_cm)
    @variable(milp, 0 <= lambda_c_sys              <= lambda_c_ub)
    @variable(milp, 0 <= mu_g_up_I[t in T, i in I] <= K_mu_g_I)
    @variable(milp, 0 <= mu_g_up_J[t in T, j in J] <= K_mu_g_J)
    @variable(milp, 0 <= mu_cm_bud_I[i in I]       <= K_mu_cm_bud_I)
    @variable(milp, 0 <= mu_cm_bud_J[j in J]       <= K_mu_cm_bud_J)
    @variable(milp, 0 <= mu_cap_up[i in I]         <= cap_up_ub)
    @variable(milp, 0 <= nu_c[z in Z]              <= K_nu_c)
    @variable(milp, 0 <= phi_lo[p in pair_keys]    <= K_phi)
    @variable(milp, 0 <= phi_hi[p in pair_keys]    <= K_phi)
    @variable(milp, 0 <= rho[z in Z]               <= K_rho)

    # ── Stationarity  ─────
    @constraint(milp, [t in T, i in I], -W_t[t]*(lambda_e[t]-vc[i]) + mu_g_up_I[t,i] >= 0)
    @constraint(milp, [t in T, j in J], -W_t[t]*(lambda_e[t]-vc[j]) + mu_g_up_J[t,j] >= 0)
    @constraint(milp, [i in I], fc[i] - sum(mu_g_up_I[t,i] for t in T) + mu_cap_up[i] - mu_cm_bud_I[i]*alpha[i] >= 0)
    @constraint(milp, [i in I], -lambda_c[gen_to_zone[i]] + mu_cm_bud_I[i] >= 0)
    @constraint(milp, [j in J], -lambda_c[gen_to_zone[j]] + mu_cm_bud_J[j] >= 0)
    @constraint(milp, [p in pair_keys],
        -(nu_c[p[1]]-rho[p[1]]) + (nu_c[p[2]]-rho[p[2]]) + phi_hi[p] - phi_lo[p] == 0)
    @constraint(milp, [z in Z], lambda_c[z] - nu_c[z] - lambda_c_sys >= 0)

    # ── Binaries ─────────────────────────────────────────────────────────────
    @variable(milp, r_q_I[t in T, i in I],   Bin); @variable(milp, r_q_J[t in T, j in J], Bin)
    @variable(milp, r_g_up_I[t in T, i in I], Bin); @variable(milp, r_g_up_J[t in T, j in J], Bin)
    @variable(milp, r_c[i in I], Bin); @variable(milp, r_cap_up[i in I], Bin)
    @variable(milp, r_cm_I[i in I], Bin); @variable(milp, r_cm_J[j in J], Bin)
    @variable(milp, r_cm_bud_I[i in I], Bin); @variable(milp, r_cm_bud_J[j in J], Bin)
    @variable(milp, r_dem[z in Z], Bin); @variable(milp, r_dem_cap[z in Z], Bin)
    @variable(milp, r_flo[p in pair_keys], Bin); @variable(milp, r_fhi[p in pair_keys], Bin)
    @variable(milp, r_rho[z in Z], Bin); @variable(milp, r_ls[t in T], Bin); @variable(milp, r_sys, Bin)

    # ── Complementarity ──
    @constraint(milp, [t in T, i in I], q[t,i] <= K_q*(1 - r_q_I[t,i]))
    @constraint(milp, [t in T, i in I], -W_t[t]*(lambda_e[t]-vc[i]) + mu_g_up_I[t,i] <= K_stat_q_I*r_q_I[t,i])
    @constraint(milp, [t in T, i in I], q_max_I[t,i] - q[t,i] <= K_slack_q_I*(1 - r_g_up_I[t,i]))
    @constraint(milp, [t in T, i in I], mu_g_up_I[t,i] <= K_mu_g_I*r_g_up_I[t,i])
    @constraint(milp, [i in I], c[i] <= K_c*(1 - r_c[i]))
    @constraint(milp, [i in I], fc[i] - sum(mu_g_up_I[t,i] for t in T) + mu_cap_up[i] - mu_cm_bud_I[i]*alpha[i] <= K_stat_c*r_c[i])
    @constraint(milp, [i in I], cap[i] - c[i] <= K_slack_cap*(1 - r_cap_up[i]))
    @constraint(milp, [i in I], mu_cap_up[i] <= K_mu_cap*r_cap_up[i])
    @constraint(milp, [i in I], c_cm_I[i] <= K_cm*(1 - r_cm_I[i]))
    @constraint(milp, [i in I], -lambda_c[gen_to_zone[i]] + mu_cm_bud_I[i] <= K_stat_cm*r_cm_I[i])
    @constraint(milp, [i in I], alpha[i]*c[i] - c_cm_I[i] <= K_slack_cm_bud_I*(1 - r_cm_bud_I[i]))
    @constraint(milp, [i in I], mu_cm_bud_I[i] <= K_mu_cm_bud_I*r_cm_bud_I[i])
    @constraint(milp, [t in T, j in J], q[t,j] <= K_q*(1 - r_q_J[t,j]))
    @constraint(milp, [t in T, j in J], -W_t[t]*(lambda_e[t]-vc[j]) + mu_g_up_J[t,j] <= K_stat_q_J*r_q_J[t,j])
    @constraint(milp, [t in T, j in J], q_max_J[t,j] - q[t,j] <= K_slack_q_J*(1 - r_g_up_J[t,j]))
    @constraint(milp, [t in T, j in J], mu_g_up_J[t,j] <= K_mu_g_J*r_g_up_J[t,j])
    @constraint(milp, [j in J], c_cm_J[j] <= K_cm*(1 - r_cm_J[j]))
    @constraint(milp, [j in J], -lambda_c[gen_to_zone[j]] + mu_cm_bud_J[j] <= K_stat_cm*r_cm_J[j])
    @constraint(milp, [j in J], alpha[j]*cap[j] - c_cm_J[j] <= K_slack_cm_bud_J*(1 - r_cm_bud_J[j]))
    @constraint(milp, [j in J], mu_cm_bud_J[j] <= K_mu_cm_bud_J*r_cm_bud_J[j])
    @constraint(milp, [z in Z], c_dem[z] <= K_dem*(1 - r_dem[z]))
    @constraint(milp, [z in Z], lambda_c[z] - nu_c[z] - lambda_c_sys <= K_stat_dem_c*r_dem[z])
    @constraint(milp, sum(c_dem[z] for z in Z) - d_c <= K_sys*(1 - r_sys))
    @constraint(milp, lambda_c_sys <= lambda_c_ub*r_sys)
    @constraint(milp, [z in Z], c_dem[z] - d_c_z[z] + net_import[z] <= K_dem*(1 - r_dem_cap[z]))
    @constraint(milp, [z in Z], nu_c[z] <= K_nu_c*r_dem_cap[z])
    @constraint(milp, [p in pair_keys], f_trans[p] + mec_pairs[p] <= (2*mec_pairs[p])*(1 - r_flo[p]))
    @constraint(milp, [p in pair_keys], phi_lo[p] <= K_phi*r_flo[p])
    @constraint(milp, [p in pair_keys], mec_pairs[p] - f_trans[p] <= (2*mec_pairs[p])*(1 - r_fhi[p]))
    @constraint(milp, [p in pair_keys], phi_hi[p] <= K_phi*r_fhi[p])
    @constraint(milp, [z in Z], d_c_z[z] - net_import[z] <= K_dem*(1 - r_rho[z]))
    @constraint(milp, [z in Z], rho[z] <= K_rho*r_rho[z])
    @constraint(milp, [t in T], ls[t] <= K_ls*(1 - r_ls[t]))
    @constraint(milp, [t in T], W_t[t]*(VOLL - lambda_e[t]) <= K_stat_ls*r_ls[t])   # W_t added

    @objective(milp, Max, sum(w[i]*c[i] for i in I))
    return milp, (; q, c, ls, lambda_e, lambda_c, lambda_c_sys, c_dem, nu_c, rho, mu_cap_up)
end

function build_milp_stage1_sweep(P; weights = default_weights(P), total_time_limit = 3600.0,
                                 tag = "milp_s1", log_dir = ".", threads = 0,
                                 per_run_time_limit = total_time_limit)
    rows = DataFrame(w_idx=Int[], status=String[], solve_time=Float64[], mip_gap=Float64[],
                     n_bin=Int[], q=Vector{Any}(), c=Vector{Any}(), ls=Vector{Any}())
    t_start = now()
    n_done = 0
    for (k, w) in enumerate(weights)
        elapsed = (now() - t_start).value / 1000.0
        remaining = total_time_limit - elapsed
        if remaining <= 0.0
            @info "MILP sweep: time budget exhausted after $(n_done)/$(length(weights)) weights"
            break
        end
        tl = min(per_run_time_limit, remaining)
        milp, V = build_milp_stage1(P; w = w, time_limit = tl, tag = "$(tag)_w$(k)", log_dir = log_dir, threads = threads)
        optimize!(milp)
        n_done += 1
 
        gap = try relative_gap(milp) catch; NaN end
        nbin = try MOI.get(milp, Gurobi.ModelAttribute("NumBinVars")) catch; -1 end
        if has_values(milp)
            push!(rows, (k, string(termination_status(milp)), solve_time(milp), gap, nbin,
                        Dict((t,g)=>value(V.q[t,g]) for t in P.T, g in P.IJ),
                        Dict(i=>value(V.c[i]) for i in P.I),
                        Dict(t=>value(V.ls[t]) for t in P.T)))
        else
            push!(rows, (k, string(termination_status(milp)), solve_time(milp), gap, nbin, missing, missing, missing))
        end
        @printf("  w%-3d  %-12s  t=%6.1fs  gap=%6.3f  (budget left=%6.1fs)\n",
                k, string(termination_status(milp)), solve_time(milp), gap, total_time_limit-elapsed-solve_time(milp))
    end
    return rows
end
 
# first transmission line index (for the default exploration weight)
L_first(P) = first(P.L)
