using JuMP, Gurobi, MathOptInterface, DataFrames, CSV
include("helper_functions.jl")
const MOI = MathOptInterface
const GRB_ENV = Gurobi.Env()        # one shared env for all benchmark solves

function add_log!(m::Model, tag::String, log_dir::String; to_console::Bool = true)
    set_optimizer_attribute(m, "LogFile", joinpath(log_dir, "$(tag).log"))
    set_optimizer_attribute(m, "LogToConsole", to_console ? 1 : 0)
    return m
end

function load_params(scenario::String; sigma::Float64 = 1.0)
    data = load_data(scenario); net = build_network(data)
    rep  = build_rep_days(data, net.bus_to_idx)
    gen  = build_generators(data, rep, net.bus_to_idx)
    (; BUS, LINE, N, L, PTDF, Fmax, bus_to_idx) = net
    (; I, J, vc, cap, tech, gens, fc, a, alpha)  = gen
    T = rep.T; W_t = rep.w; IJ = vcat(I, J)
    gen_bus_idx = Dict(g => bus_to_idx[gens[g]] for g in IJ)
    D_max   = Dict(t => rep.load[t] for t in T)
    d_nodal = Dict((t,bus) => rep.demand[(t,bus_to_idx[bus])]/rep.load[t] for t in T, bus in BUS)
    VOLL = 300.0; hmk = 150.0
    J_res = [j for j in J if tech[j] in ["solar","onwind","offwind-ac"]]
    zones = build_zones(data, net, gen; sigma = sigma)
    (; Z, gen_to_zone, gens_in_zone, buses_in_zone, MEC) = zones
    I_in_z = Dict(z => [i for i in I if gen_to_zone[i]==z] for z in Z)
    J_in_z = Dict(z => [j for j in J if gen_to_zone[j]==z] for z in Z)
    d_c_z = Dict(z => maximum(sum(rep.demand[(t,n)] for n in buses_in_zone[z]; init=0.0) -
            sum(cap[j]*a[(t,j)] for j in J_res if gen_to_zone[j]==z; init=0.0) for t in T) for z in Z)
    d_c   = maximum(D_max[t] - sum(cap[j]*a[(t,j)] for j in J_res; init=0.0) for t in T)
    @assert d_c <= sum(d_c_z[z] for z in Z) + 1e-6
    mec_pairs = Dict{Tuple{Int,Int},Float64}()
    for ((z1,z2),v) in MEC; z1<z2 || continue; mec_pairs[(z1,z2)] = v; end
    pair_keys = collect(keys(mec_pairs))
    return (; data, net, rep, gen, BUS, N, L, PTDF, Fmax, I, J, IJ, vc, cap, fc, a, alpha,
            tech, T, W_t, gen_bus_idx, D_max, d_nodal, VOLL, hmk, J_res,
            Z, gen_to_zone, buses_in_zone, I_in_z, J_in_z, d_c_z, d_c, mec_pairs, pair_keys)
end

function add_market_cap_feas!(m::Model, P)
    (; N, I, J, IJ, vc, cap, fc, a, alpha, T, W_t, gen_to_zone, D_max,
       VOLL, Z, I_in_z, J_in_z, d_c_z, d_c, mec_pairs, pair_keys) = P

    Wmax        = maximum(W_t[t] for t in T)
    total_hours = sum(W_t[t] for t in T)
    lambda_c_ub = maximum(fc[i] for i in I)
    cap_up_ub   = VOLL * total_hours + lambda_c_ub
    P_cap_cm    = 2.0 * lambda_c_ub
    K_mu_g_I    = VOLL * Wmax
    K_mu_g_J    = VOLL * Wmax
    K_q         = maximum(cap[g] for g in IJ; init = 0.0)
    K_c         = maximum(cap[i] for i in I; init = 0.0)
    K_cm        = maximum(cap[g] for g in IJ; init = 0.0)
    mec_bound_z = Dict(z => sum((mec_pairs[p] for p in pair_keys if p[1]==z || p[2]==z); init=0.0) for z in Z)
    K_dem       = maximum((d_c_z[z] + mec_bound_z[z] for z in Z); init = 0.0)
    K_ls        = maximum(D_max[t] for t in T; init = 0.0)
    
    @variable(m, 0 <= ls[t in T]      <= K_ls)
    @variable(m, 0 <= q[t in T, g in IJ] <= K_q)
    @variable(m, 0 <= c[i in I]       <= cap[i])         
    @variable(m, 0 <= c_cm_I[i in I]  <= K_cm)
    @variable(m, 0 <= c_cm_J[j in J]  <= K_cm)
    @variable(m, 0 <= c_dem[z in Z]   <= K_dem)
    @variable(m, -mec_pairs[p] <= f_trans[p in pair_keys] <= mec_pairs[p])
    @expression(m, net_import[z in Z],
        sum(f_trans[p] for p in pair_keys if p[1]==z; init=0.0) -
        sum(f_trans[p] for p in pair_keys if p[2]==z; init=0.0))
    @expression(m, q_max_I[t in T, i in I], c[i])
    @expression(m, q_max_J[t in T, j in J], cap[j]*a[(t,j)])
 
    @variable(m, 0 <= lambda_e[t in T]          <= VOLL)
    @variable(m, 0 <= lambda_c[z in Z]          <= P_cap_cm)
    @variable(m, 0 <= lambda_c_sys               <= lambda_c_ub)
    @variable(m, 0 <= mu_g_up_I[t in T, i in I] <= K_mu_g_I)
    @variable(m, 0 <= mu_g_up_J[t in T, j in J] <= K_mu_g_J)
    @variable(m, 0 <= mu_cm_bud_I[i in I]       <= P_cap_cm)
    @variable(m, 0 <= mu_cm_bud_J[j in J]       <= P_cap_cm)
    @variable(m, 0 <= mu_cap_up[i in I]         <= cap_up_ub)
    @variable(m, 0 <= nu_c[z in Z]              <= P_cap_cm)
    @variable(m, 0 <= phi_lo[p in pair_keys]    <= P_cap_cm)
    @variable(m, 0 <= phi_hi[p in pair_keys]    <= P_cap_cm)
    @variable(m, 0 <= rho[z in Z]               <= P_cap_cm)

    # primal feasibility 
    @expression(m, s_q_I[t in T, i in I], q_max_I[t,i] - q[t,i])
    @expression(m, s_q_J[t in T, j in J], q_max_J[t,j] - q[t,j])
    @expression(m, s_cap[i in I], cap[i] - c[i])
    @expression(m, s_cm_I[i in I], alpha[i]*c[i] - c_cm_I[i])
    @expression(m, s_cm_J[j in J], alpha[j]*cap[j] - c_cm_J[j])
    @expression(m, s_floor[z in Z], c_dem[z] - d_c_z[z] + net_import[z])
    @expression(m, s_req[z in Z], d_c_z[z] - net_import[z])
    @expression(m, s_flo[p in pair_keys], f_trans[p] + mec_pairs[p])
    @expression(m, s_fhi[p in pair_keys], mec_pairs[p] - f_trans[p])
    @expression(m, s_sys, sum(c_dem[z] for z in Z) - d_c)
    @constraint(m, [t in T], ls[t] <= D_max[t])
    @constraint(m, [t in T, i in I], s_q_I[t,i] >= 0)
    @constraint(m, [t in T, j in J], s_q_J[t,j] >= 0)
    @constraint(m, [i in I], s_cap[i] >= 0)
    @constraint(m, [i in I], s_cm_I[i] >= 0)
    @constraint(m, [j in J], s_cm_J[j] >= 0)
    @constraint(m, [z in Z], s_floor[z] >= 0)
    @constraint(m, [z in Z], s_req[z] >= 0)
    @constraint(m, [p in pair_keys], s_flo[p] >= 0)
    @constraint(m, [p in pair_keys], s_fhi[p] >= 0)
    @constraint(m, s_sys >= 0)

    # clearing 
    @constraint(m, [t in T], sum(q[t,g] for g in IJ) + ls[t] == D_max[t])
    @constraint(m, [z in Z],
        sum(c_cm_I[i] for i in I_in_z[z]; init=0.0) +
        sum(c_cm_J[j] for j in J_in_z[z]; init=0.0) - c_dem[z] == 0)

    # dual feasibility / stationarity 
    @expression(m, st_q_I[t in T, i in I], mu_g_up_I[t,i] - W_t[t]*(lambda_e[t]-vc[i]))
    @expression(m, st_q_J[t in T, j in J], mu_g_up_J[t,j] - W_t[t]*(lambda_e[t]-vc[j]))
    @expression(m, st_c[i in I], fc[i] - sum(mu_g_up_I[t,i] for t in T) + mu_cap_up[i] - mu_cm_bud_I[i]*alpha[i])
    @expression(m, st_cm_I[i in I], mu_cm_bud_I[i] - lambda_c[gen_to_zone[i]])
    @expression(m, st_cm_J[j in J], mu_cm_bud_J[j] - lambda_c[gen_to_zone[j]])
    @expression(m, st_dem[z in Z], lambda_c[z] - nu_c[z] - lambda_c_sys)
    @expression(m, st_ls[t in T], W_t[t]*(VOLL - lambda_e[t]))
    @constraint(m, [t in T, i in I], st_q_I[t,i] >= 0)
    @constraint(m, [t in T, j in J], st_q_J[t,j] >= 0)
    @constraint(m, [i in I], st_c[i] >= 0)
    @constraint(m, [i in I], st_cm_I[i] >= 0)
    @constraint(m, [j in J], st_cm_J[j] >= 0)
    @constraint(m, [z in Z], st_dem[z] >= 0)
    @constraint(m, [t in T], st_ls[t] >= 0)
    @constraint(m, [p in pair_keys],   # stat_f (equality; f is free)
        -(nu_c[p[1]]-rho[p[1]]) + (nu_c[p[2]]-rho[p[2]]) + phi_hi[p] - phi_lo[p] == 0)

    return (; ls, q, c, c_cm_I, c_cm_J, c_dem, f_trans, net_import, q_max_I, q_max_J,
            lambda_e, lambda_c, lambda_c_sys, mu_g_up_I, mu_g_up_J, mu_cm_bud_I, mu_cm_bud_J,
            mu_cap_up, nu_c, phi_lo, phi_hi, rho,
            s_q_I, s_q_J, s_cap, s_cm_I, s_cm_J, s_floor, s_req, s_flo, s_fhi, s_sys,
            st_q_I, st_q_J, st_c, st_cm_I, st_cm_J, st_dem, st_ls)
end

# ---------------------------------------------------------------------------
# STRONG DUALITY blocks (PD methods)  
# ---------------------------------------------------------------------------
function add_strong_duality!(m::Model, P, V)
    (; I, J, vc, cap, fc, a, alpha, T, W_t, gen_to_zone, Z, d_c_z, d_c, mec_pairs, pair_keys) = P
    @constraint(m, [i in I],
        sum(W_t[t]*(V.lambda_e[t]-vc[i])*V.q[t,i] for t in T) - fc[i]*V.c[i] +
        V.lambda_c[gen_to_zone[i]]*V.c_cm_I[i] == cap[i]*V.mu_cap_up[i])
    @constraint(m, [j in J],
        sum(W_t[t]*(V.lambda_e[t]-vc[j])*V.q[t,j] for t in T) + V.lambda_c[gen_to_zone[j]]*V.c_cm_J[j] ==
        sum(a[(t,j)]*cap[j]*V.mu_g_up_J[t,j] for t in T) + alpha[j]*cap[j]*V.mu_cm_bud_J[j])
    @constraint(m,
        sum(V.lambda_c[z]*V.c_dem[z] for z in Z) ==
        sum(d_c_z[z]*V.nu_c[z] for z in Z) - sum(d_c_z[z]*V.rho[z] for z in Z) -
        sum(mec_pairs[p]*(V.phi_lo[p]+V.phi_hi[p]) for p in pair_keys) + d_c*V.lambda_c_sys)
    @constraint(m, [t in T], W_t[t]*V.ls[t]*(P.VOLL - V.lambda_e[t]) == 0)   # energy demand
    return m
end

# ---------------------------------------------------------------------------
# FULL PD: feas + strong duality + redispatch IN-MODEL 
# ---------------------------------------------------------------------------
function build_full_pd(P; sense::Symbol = :Max, time_limit = 3600.0,
                       tag = "fullpd", log_dir = ".", threads = 0)
    (; N, L, PTDF, Fmax, I, IJ, vc, cap, a, T, W_t, gen_bus_idx, D_max, d_nodal, BUS, VOLL, hmk) = P
    m = direct_model(Gurobi.Optimizer(GRB_ENV))
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "TimeLimit", time_limit)
    threads > 0 && set_optimizer_attribute(m, "Threads", threads)
    add_log!(m, tag, log_dir)
    V = add_market_cap_feas!(m, P)
    add_strong_duality!(m, P, V)
    q = V.q

    @variable(m, r_up[t in T, g in IJ] >= 0)
    @variable(m, r_down[t in T, g in IJ] >= 0)
    @variable(m, curt[t in T, n in N] >= 0)
    @variable(m, p_flow[t in T, l in L])
    @expression(m, d[t in T, n in N], d_nodal[(t,BUS[n])]*D_max[t])
    @expression(m, served[t in T, n in N], d[t,n] - d_nodal[(t,BUS[n])]*V.ls[t])
    @expression(m, q_bar[t in T, g in IJ], g in I ? V.c[g] : cap[g]*a[(t,g)])
    @expression(m, p_inj[t in T, n in N],
        sum(q[t,g] + r_up[t,g] - r_down[t,g] for g in IJ if gen_bus_idx[g]==n; init=0.0) - d[t,n] + curt[t,n])

    @variable(m, beta_pos[t in T, g in IJ] >= 0)
    @variable(m, beta_neg[t in T, g in IJ] >= 0)
    @variable(m, beta_curt[t in T, n in N] >= 0)
    @variable(m, lambda_rd[t in T])
    @variable(m, rho_up[t in T, l in L] >= 0)
    @variable(m, rho_low[t in T, l in L] >= 0)
    @variable(m, phi[t in T, l in L])

    @constraint(m, [t in T, g in IJ], r_up[t,g] - (q_bar[t,g] - q[t,g]) <= 0)
    @constraint(m, [t in T, g in IJ], r_down[t,g] - q[t,g] <= 0)
    @constraint(m, [t in T, n in N], curt[t,n] - served[t,n] <= 0)
    @constraint(m, [t in T, l in L], p_flow[t,l] - Fmax[l] <= 0)
    @constraint(m, [t in T, l in L], -p_flow[t,l] - Fmax[l] <= 0)
    @constraint(m, [t in T, l in L], p_flow[t,l] - sum(PTDF[l,n]*p_inj[t,n] for n in N) == 0)
    @constraint(m, [t in T], sum(r_up[t,g]-r_down[t,g] for g in IJ) + sum(curt[t,n] for n in N) == 0)

    @constraint(m, [t in T, g in IJ],
        W_t[t]*(vc[g]+hmk) + beta_pos[t,g] + lambda_rd[t] - sum(PTDF[l,gen_bus_idx[g]]*phi[t,l] for l in L) >= 0)
    @constraint(m, [t in T, g in IJ],
        W_t[t]*(-vc[g]+hmk) + beta_neg[t,g] - lambda_rd[t] + sum(PTDF[l,gen_bus_idx[g]]*phi[t,l] for l in L) >= 0)
    @constraint(m, [t in T, n in N],
        W_t[t]*VOLL + beta_curt[t,n] + lambda_rd[t] - sum(PTDF[l,n]*phi[t,l] for l in L) >= 0)
    @constraint(m, [t in T, l in L], phi[t,l] + rho_up[t,l] - rho_low[t,l] == 0)

    @expression(m, f_mkt[t in T, l in L],
        sum(PTDF[l,n]*(sum(q[t,g] for g in IJ if gen_bus_idx[g]==n; init=0.0) - d[t,n]) for n in N))
    @constraint(m,            # redispatch strong duality (dense: beta*q, beta*c, PTDF*q*phi)
        sum(W_t[t]*( sum((vc[g]+hmk)*r_up[t,g] + (-vc[g]+hmk)*r_down[t,g] for g in IJ) +
                     VOLL*sum(curt[t,n] for n in N) ) for t in T)
        ==
        sum(W_t[t]*( - sum((q_bar[t,g]-q[t,g])*beta_pos[t,g] for g in IJ)
                     - sum(q[t,g]*beta_neg[t,g] for g in IJ)
                     - sum(served[t,n]*beta_curt[t,n] for n in N)
                     - sum(Fmax[l]*(rho_up[t,l]+rho_low[t,l]) for l in L)
                     - sum(f_mkt[t,l]*phi[t,l] for l in L) ) for t in T))

    @expression(m, rd_cost,
        sum(W_t[t]*( sum(vc[g]*(r_up[t,g]-r_down[t,g]) for g in IJ) +
                     VOLL*sum(curt[t,n] for n in N) ) for t in T))
    @objective(m, Min, rd_cost)
    set_objective_sense(m, sense == :Min ? MOI.MIN_SENSE : MOI.MAX_SENSE)
    return m, (; V..., r_up, r_down, curt, p_flow, rd_cost, q_bar, served,
              beta_pos, beta_neg, beta_curt, lambda_rd, rho_up, rho_low, phi, f_mkt)
end
