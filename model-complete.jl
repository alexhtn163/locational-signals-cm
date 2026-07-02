using JuMP
using Gurobi
using MathOptInterface
using Random
using DataFrames
using Statistics
using CSV
 
include("helper_functions.jl")
 
# ============================================================================
# Setup
# ============================================================================
scenario = "cs2_nl34_z3"
data = load_data(scenario)
net  = build_network(data)
rep  = build_rep_days(data, net.bus_to_idx)
gen  = build_generators(data, rep, net.bus_to_idx)
(; BUS, LINE, N, L, PTDF, Fmax, bus_to_idx) = net
(; I, J, vc, cap, tech, gens, fc, a, alpha) = gen
T = rep.T
W_t = rep.w
 
D_max   = Dict(t => rep.load[t] for t in T)
d_nodal = Dict((t, bus) => rep.demand[(t, net.bus_to_idx[bus])] / rep.load[t]
               for t in T, bus in BUS)
snapshots = Vector{String}(data.dem_df.snapshot)
 
IJ          = vcat(I, J)
gen_bus_idx = Dict(g => bus_to_idx[gens[g]] for g in IJ)
 
VOLL  = 100
PEN   = 69000
#PEN = 300 # poc
J_res = [j for j in J if tech[j] in ["solar", "onwind", "offwind-ac"]]
 
sigmas = [0.0, 0.25, 0.5]
#igmas = [0.5]
 
# ============================================================================
# Sigma loop
# ============================================================================
for sigma in sigmas
 
    zones = build_zones(data, net, gen; sigma = sigma)
    (; Z, gen_to_zone, gens_in_zone, buses_in_zone, MEC) = zones
 
    # ── Capacity requirements ─────────────────────────────────────────────
    d_c_z = Dict(z => maximum(
            sum(rep.demand[(t, n)] for n in buses_in_zone[z]; init = 0.0) -
            sum(cap[j] * a[(t, j)] for j in J_res if gen_to_zone[j] == z; init = 0.0)
            for t in T) for z in Z)
    
    d_c = maximum(D_max[t] - sum(cap[j] * a[(t, j)] for j in J_res; init = 0.0)
                      for t in T)
  
 
    println("Zonal capacity requirements (d_c_z):")
    for z in Z
        println("  Zone $z: $(round(d_c_z[z], digits=1)) MW")
    end
    println("System-wide capacity requirement (d_c): $(round(d_c, digits=1)) MW")
 
    # ── MEC pairs ─────────────────────────────────────────────────────────
    mec_pairs = Dict{Tuple{Int,Int}, Float64}()
    for ((z1, z2), val) in MEC
        z1 < z2 || continue
        mec_pairs[(z1, z2)] = val
    end
    pair_keys = collect(keys(mec_pairs))
 
    println("Cross-zonal transfer bounds (± sigma*TC), sigma=$sigma:")
    for p in pair_keys
        println("  Pair $(p[1])-$(p[2]): ±$(round(mec_pairs[p], digits=1)) MW")
    end
 
    mec_bound_z = Dict(z => sum(
        (mec_pairs[p] for p in pair_keys if p[1] == z || p[2] == z); init = 0.0)
        for z in Z)
 
    # ── Weight vectors ────────────────────────────────────────────────────
    weights_ptdf = []
    for l in L
        w_pos = Dict(i => PTDF[l, gen_bus_idx[i]] for i in I)
        w_neg = Dict(i => -PTDF[l, gen_bus_idx[i]] for i in I)
        push!(weights_ptdf, w_pos)
        push!(weights_ptdf, w_neg)
    end
 
    all_weights = vcat(weights_ptdf)
 
    # ── K constants ──────────────────────────────────────────────────────
     K_q   = maximum(cap[g] for g in IJ; init = 0.0)
    K_ls  = maximum(D_max[t] for t in T; init = 0.0)
    K_c   = maximum(cap[i] for i in I; init = 0.0)
    K_cm  = maximum(cap[g] for g in IJ; init = 0.0)
    K_dem = maximum((d_c_z[z] + mec_bound_z[z] for z in Z); init = 0.0)
    K_sys = sum(d_c_z[z] + mec_bound_z[z] for z in Z; init = 0.0)
 
    K_slack_q_I      = maximum(cap[i] for i in I; init = 0.0)
    K_slack_q_J      = maximum(cap[j] * a[(t, j)] for t in T for j in J; init = 0.0)
    K_slack_cap      = maximum(cap[i] for i in I; init = 0.0)
    K_slack_cm_bud_I = maximum(cap[i] for i in I; init = 0.0)
    K_slack_cm_bud_J = maximum(alpha[j] * cap[j] for j in J; init = 0.0)
 
    total_hours = sum(W_t[t] for t in T)
    Wmax        = maximum(W_t[t] for t in T)
    lambda_c_ub = maximum(fc[i] for i in I)
    P_cap_cm    = lambda_c_ub
    cap_up_ub   = total_hours * VOLL + P_cap_cm
 
    K_mu_g_I      = VOLL * Wmax
    K_mu_g_J      = VOLL * Wmax
    K_mu_cm_bud_I = P_cap_cm
    K_mu_cm_bud_J = P_cap_cm
    K_mu_cap      = cap_up_ub
    K_nu_c        = P_cap_cm
    K_phi         = P_cap_cm
    K_rho         = P_cap_cm
 
    K_stat_q_I   = Wmax * (maximum(vc[i] for i in I; init = 0.0) + VOLL)
    K_stat_q_J   = Wmax * (maximum(vc[j] for j in J; init = 0.0) + VOLL)
    K_stat_c     = maximum(fc[g] for g in IJ; init = 0.0) + cap_up_ub + lambda_c_ub
    K_stat_cm    = P_cap_cm
    K_stat_dem_c = P_cap_cm
    K_stat_ls    = VOLL * Wmax
 
    println("K_q=$K_q, K_ls=$K_ls, K_c=$K_c, K_cm=$K_cm, K_dem=$K_dem")
    println("K_stat_cm=$K_stat_cm, P_cap_cm=$P_cap_cm")
 
    # ── Result storage (initialised ONCE per sigma) ───────────────────────
    results = DataFrame(
        run_id       = Int[],    weight_type = String[], mpec_status = String[],
        solve_time   = Float64[], gap         = Float64[],
        total_cap    = Float64[], market_cost = Float64[],
        lambda_e_avg = Float64[], ls_total    = Float64[],
        rd_status    = String[],  rd_volume   = Float64[], rd_cost      = Float64[],
        curtailment  = Float64[], rd_up_vol   = Float64[], rd_down_vol  = Float64[],
        rd_cost_econ = Float64[], curt_cost   = Float64[] 
    )
    for z in Z
        results[!, "lambda_c_z$z"]   = Float64[]
        results[!, "c_dem_z$z"]      = Float64[]
        results[!, "nu_c_z$z"]       = Float64[]
        results[!, "req_z$z"]        = Float64[]
        results[!, "net_import_z$z"] = Float64[]
    end
    for i in I
        results[!, "c_$i"] = Float64[]
    end
 
    all_prices    = DataFrame(run_id=Int[], t=Int[], D_max=Float64[],
                              lambda_e=Float64[], ls=Float64[], total_gen=Float64[])
    all_dispatch  = DataFrame(run_id=Int[], t=Int[], generator=String[], bus=String[],
                              q=Float64[], q_max=Float64[], mu_g_up=Float64[], type=String[])
    all_invest    = DataFrame(run_id=Int[], generator=String[], bus=String[], tech=String[],
                              zone=Int[], cap_max=Float64[], c=Float64[], vc=Float64[],
                              fc=Float64[], mu_cap_up=Float64[], mu_cm_bud=Float64[],
                              energy_rent=Float64[], cm_rent=Float64[],
                              total_rent=Float64[], fixed_cost=Float64[], profit=Float64[])
    all_legacy    = DataFrame(run_id=Int[], generator=String[], bus=String[], zone=Int[],
                              cap=Float64[], vc=Float64[], mu_cm_bud=Float64[],
                              energy_rent=Float64[], cm_rent=Float64[], total_gen=Float64[])
    all_flows     = DataFrame(run_id=Int[], t=Int[], line=String[], market_flow=Float64[],
                              final_flow=Float64[], Fmax=Float64[], market_utilization=Float64[],
                              final_utilization=Float64[], market_overload=Float64[],
                              market_violated=Bool[], final_feasible=Bool[])
    all_rd_detail = DataFrame(run_id=Int[], t=Int[], generator=String[], bus=String[],
                              q_market=Float64[], r_up=Float64[], r_down=Float64[],
                              q_final=Float64[], q_max=Float64[], up_cost=Float64[],
                              down_refund=Float64[], lambda_e=Float64[], vc=Float64[])
    all_curt      = DataFrame(run_id=Int[], t=Int[], node=String[], demand=Float64[],
                              ls=Float64[], curtailment=Float64[])
    all_cm        = DataFrame(run_id=Int[], generator=String[], type=String[], bus=String[],
                              zone=Int[], cm_offer=Float64[], lambda_c_z=Float64[],
                              mu_cm_bud=Float64[])
    all_cm_dem    = DataFrame(run_id=Int[], zone=Int[], d_c_z=Float64[], net_import=Float64[],
                              req_z=Float64[], c_dem=Float64[], lambda_c_z=Float64[],
                              nu_c=Float64[], rho=Float64[])
    all_cm_diag   = DataFrame(run_id=Int[], sys_avail=Float64[],
                              sys_req=Float64[], sys_slack=Float64[], zone=Int[],
                              req_z=Float64[], net_import=Float64[], zone_avail=Float64[],
                              zone_offers=Float64[], zone_slack=Float64[],
                              lambda_c_z=Float64[], nu_c_z=Float64[])
    all_transfer  = DataFrame(run_id=Int[], zone_a=Int[], zone_b=Int[], bound=Float64[],
                              f_ab=Float64[], phi_lo=Float64[], phi_hi=Float64[],
                              at_bound=String[])
    all_values = DataFrame(run_id=Int[], sigma=Float64[], model=String[],
                              variable=String[], value=Float64[])
 
    println("\n" * "=" ^ 60)
    println("Starting $(length(all_weights)) runs (sigma=$sigma)...")
    println("=" ^ 60)
 
    # ========================================================================
    # Run loop — fresh model built every run
    # ========================================================================
    for (k, w) in enumerate(all_weights)
        local d_val, rd, r_up, r_down, p_flow, curt, p_inj
 
        n_ptdf = length(weights_ptdf)
        wtype  = k <= n_ptdf ? (isodd(k) ? "ptdf_pos" : "ptdf_neg") : "zone"
 
        print("Run $k/$(length(all_weights)) ($wtype)... ")
 
        # ── Build a fresh model for every run ───────────────────────────────
        milp = Model(Gurobi.Optimizer)
        set_optimizer_attribute(milp, "TimeLimit",      220)
        set_optimizer_attribute(milp, "Presolve",        2)
        set_optimizer_attribute(milp, "FeasibilityTol", 1e-3)
        set_optimizer_attribute(milp, "MIPGap",         0.01)
        set_optimizer_attribute(milp, "OutputFlag",      0)
        set_optimizer_attribute(milp, "DualReductions",  0)
 
        I_in_z = Dict(z => [i for i in I if gen_to_zone[i] == z] for z in Z)
        J_in_z = Dict(z => [j for j in J if gen_to_zone[j] == z] for z in Z)
 
        # ── Primal variables ─────────────────────────────────────────────────
        @expression(milp, d[t in T, n in N], d_nodal[(t, BUS[n])] * D_max[t])
        @variable(milp, ls[t in T] >= 0)
        @variable(milp, q[t in T, g in IJ] >= 0)
        @variable(milp, c[i in I] >= 0)
        @variable(milp, c_cm_I[i in I] >= 0)
        @variable(milp, c_cm_J[j in J] >= 0)
        @variable(milp, c_dem[z in Z] >= 0)
        @variable(milp, f_trans[p in pair_keys])   # free (signed)
 
        @expression(milp, net_import[z in Z],
            sum(f_trans[p] for p in pair_keys if p[1] == z; init = 0.0) -
            sum(f_trans[p] for p in pair_keys if p[2] == z; init = 0.0))
 
        @expression(milp, req_z[z in Z], d_c_z[z] - net_import[z])
        @expression(milp, q_max_I[t in T, i in I], c[i])
        @expression(milp, q_max_J[t in T, j in J], cap[j] * a[(t, j)])
 
        # ── Primal constraints (energy market) ───────────────────────────────
        @constraint(milp, [t in T], ls[t] <= D_max[t])
        @constraint(milp, [t in T], sum(q[t, g] for g in IJ) + ls[t] == D_max[t])
        @constraint(milp, [t in T, i in I], q[t, i] <= q_max_I[t, i])
        @constraint(milp, [t in T, j in J], q[t, j] <= q_max_J[t, j])
        @constraint(milp, [i in I], c[i] <= cap[i])
 
        # ── Primal constraints (capacity market) ─────────────────────────────
        @constraint(milp, cm_budget_I[i in I],  c_cm_I[i] <= alpha[i] * c[i])
        @constraint(milp, cm_budget_J[j in J],  c_cm_J[j] <= alpha[j] * cap[j])
        @constraint(milp, f_lo[p in pair_keys],  f_trans[p] + mec_pairs[p] >= 0)
        @constraint(milp, f_hi[p in pair_keys],  mec_pairs[p] - f_trans[p] >= 0)
        # Zero-sum obligation shift:
        #   req_z = d_c_z - net_import[z]
        #   c_dem[z] must be AT LEAST this shifted zonal requirement.
        @constraint(milp, req_nonneg[z in Z],    d_c_z[z] - net_import[z] >= 0)
        @constraint(milp, dem_c_floor[z in Z],   c_dem[z] - d_c_z[z] + net_import[z] >= 0)
        @constraint(milp, cm_clearing[z in Z],
            sum(c_cm_I[i] for i in I_in_z[z]; init = 0.0) +
            sum(c_cm_J[j] for j in J_in_z[z]; init = 0.0) - c_dem[z] == 0)
        @constraint(milp, cm_system,
            sum(c_dem[z] for z in Z) >= d_c)
 
        # ── Dual variables ───────────────────────────────────────────────────
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
 
        # ── Stationarity conditions ──────────────────────────────────────────
        @constraint(milp, [t in T, i in I],
            -W_t[t] * (lambda_e[t] - vc[i]) + mu_g_up_I[t, i] >= 0)
        @constraint(milp, [t in T, j in J],
            -W_t[t] * (lambda_e[t] - vc[j]) + mu_g_up_J[t, j] >= 0)
        @constraint(milp, stat_c[i in I],
            fc[i] - sum(mu_g_up_I[t, i] for t in T) +
            mu_cap_up[i] - mu_cm_bud_I[i] * alpha[i] >= 0)
        @constraint(milp, stat_cm_I[i in I],
            -lambda_c[gen_to_zone[i]] + mu_cm_bud_I[i] >= 0)
        @constraint(milp, stat_cm_J[j in J],
            -lambda_c[gen_to_zone[j]] + mu_cm_bud_J[j] >= 0)
        @constraint(milp, stat_f[p in pair_keys],
            -(nu_c[p[1]] - rho[p[1]]) + (nu_c[p[2]] - rho[p[2]]) +
            phi_hi[p] - phi_lo[p] == 0)
        @constraint(milp, stat_dem_c[z in Z],
            lambda_c[z] - nu_c[z] - lambda_c_sys >= 0)
 
        # ── Binary variables ────────────────────
        @variable(milp, r_q_I[t in T, i in I],   Bin)
        @variable(milp, r_q_J[t in T, j in J],   Bin)
        @variable(milp, r_g_up_I[t in T, i in I], Bin)
        @variable(milp, r_g_up_J[t in T, j in J], Bin)
        @variable(milp, r_c[i in I],              Bin)
        @variable(milp, r_cap_up[i in I],         Bin)
        @variable(milp, r_cm_I[i in I],           Bin)
        @variable(milp, r_cm_J[j in J],           Bin)
        @variable(milp, r_cm_bud_I[i in I],       Bin)
        @variable(milp, r_cm_bud_J[j in J],       Bin)
        @variable(milp, r_dem[z in Z],            Bin)
        @variable(milp, r_dem_cap[z in Z],        Bin)
        @variable(milp, r_flo[p in pair_keys],    Bin)
        @variable(milp, r_fhi[p in pair_keys],    Bin)
        @variable(milp, r_rho[z in Z],            Bin)
        @variable(milp, r_ls[t in T],             Bin)
        @variable(milp, r_sys,                    Bin)
 
        # 0 <= q[t,i] ⊥ stat_q_I >= 0
        @constraint(milp, [t in T, i in I],
            q[t, i] <= K_q * (1 - r_q_I[t, i]))
        @constraint(milp, [t in T, i in I],
            -W_t[t] * (lambda_e[t] - vc[i]) + mu_g_up_I[t, i] <= K_stat_q_I * r_q_I[t, i])
 
        # 0 <= (q_max_I - q) ⊥ mu_g_up_I >= 0
        @constraint(milp, [t in T, i in I],
            q_max_I[t, i] - q[t, i] <= K_slack_q_I * (1 - r_g_up_I[t, i]))
        @constraint(milp, [t in T, i in I],
            mu_g_up_I[t, i] <= K_mu_g_I * r_g_up_I[t, i])
 
        # 0 <= c ⊥ stat_c >= 0
        @constraint(milp, [i in I], c[i] <= K_c * (1 - r_c[i]))
        @constraint(milp, [i in I],
            fc[i] - sum(mu_g_up_I[t, i] for t in T) +
            mu_cap_up[i] - mu_cm_bud_I[i] * alpha[i] <=
            K_stat_c * r_c[i])
 
        # 0 <= (cap - c) ⊥ mu_cap_up >= 0
        @constraint(milp, [i in I], cap[i] - c[i] <= K_slack_cap * (1 - r_cap_up[i]))
        @constraint(milp, [i in I], mu_cap_up[i] <= K_mu_cap * r_cap_up[i])
 
        # 0 <= c_cm_I ⊥ stat_cm_I >= 0
        @constraint(milp, [i in I], c_cm_I[i] <= K_cm * (1 - r_cm_I[i]))
        @constraint(milp, [i in I],
            -lambda_c[gen_to_zone[i]] + mu_cm_bud_I[i] <= K_stat_cm * r_cm_I[i])
 
        # 0 <= (budget slack) ⊥ mu_cm_bud_I >= 0
        @constraint(milp, [i in I],
            alpha[i] * c[i] - c_cm_I[i] <= K_slack_cm_bud_I * (1 - r_cm_bud_I[i]))
        @constraint(milp, [i in I], mu_cm_bud_I[i] <= K_mu_cm_bud_I * r_cm_bud_I[i])
 
        # 0 <= q[t,j] ⊥ stat_q_J >= 0
        @constraint(milp, [t in T, j in J],
            q[t, j] <= K_q * (1 - r_q_J[t, j]))
        @constraint(milp, [t in T, j in J],
            -W_t[t] * (lambda_e[t] - vc[j]) + mu_g_up_J[t, j] <= K_stat_q_J * r_q_J[t, j])
 
        # 0 <= (q_max_J - q) ⊥ mu_g_up_J >= 0
        @constraint(milp, [t in T, j in J],
            q_max_J[t, j] - q[t, j] <= K_slack_q_J * (1 - r_g_up_J[t, j]))
        @constraint(milp, [t in T, j in J],
            mu_g_up_J[t, j] <= K_mu_g_J * r_g_up_J[t, j])
 
        # 0 <= c_cm_J ⊥ stat_cm_J >= 0
        @constraint(milp, [j in J], c_cm_J[j] <= K_cm * (1 - r_cm_J[j]))
        @constraint(milp, [j in J],
            -lambda_c[gen_to_zone[j]] + mu_cm_bud_J[j] <= K_stat_cm * r_cm_J[j])
 
        # 0 <= (budget slack J) ⊥ mu_cm_bud_J >= 0
        @constraint(milp, [j in J],
            alpha[j] * cap[j] - c_cm_J[j] <= K_slack_cm_bud_J * (1 - r_cm_bud_J[j]))
        @constraint(milp, [j in J], mu_cm_bud_J[j] <= K_mu_cm_bud_J * r_cm_bud_J[j])
 
        # 0 <= c_dem ⊥ stat_dem_c >= 0
        @constraint(milp, [z in Z], c_dem[z] <= K_dem * (1 - r_dem[z]))
        @constraint(milp, [z in Z],
            lambda_c[z] - nu_c[z] - lambda_c_sys <= K_stat_dem_c * r_dem[z])

        # 0 <= (system slack) ⊥ lambda_c_sys >= 0
        @constraint(milp,
            sum(c_dem[z] for z in Z) - d_c <= K_sys * (1 - r_sys))
        @constraint(milp, lambda_c_sys <= lambda_c_ub * r_sys)
 
        # 0 <= (c_dem - req_z) ⊥ nu_c >= 0
        @constraint(milp, [z in Z],
            c_dem[z] - d_c_z[z] + net_import[z] <= K_dem * (1 - r_dem_cap[z]))
        @constraint(milp, [z in Z], nu_c[z] <= K_nu_c * r_dem_cap[z])
 
        # 0 <= (f + mec) ⊥ phi_lo >= 0
        @constraint(milp, [p in pair_keys],
            f_trans[p] + mec_pairs[p] <= (2 * mec_pairs[p]) * (1 - r_flo[p]))
        @constraint(milp, [p in pair_keys], phi_lo[p] <= K_phi * r_flo[p])
 
        # 0 <= (mec - f) ⊥ phi_hi >= 0
        @constraint(milp, [p in pair_keys],
            mec_pairs[p] - f_trans[p] <= (2 * mec_pairs[p]) * (1 - r_fhi[p]))
        @constraint(milp, [p in pair_keys], phi_hi[p] <= K_phi * r_fhi[p])
 
        # 0 <= (d_c_z - net_import) ⊥ rho >= 0
        @constraint(milp, [z in Z],
            d_c_z[z] - net_import[z] <= K_dem * (1 - r_rho[z]))
        @constraint(milp, [z in Z], rho[z] <= K_rho * r_rho[z])

 
        # 0 <= ls ⊥ (VOLL - lambda_e) >= 0
        @constraint(milp, [t in T], ls[t] <= K_ls * (1 - r_ls[t]))
        @constraint(milp, [t in T], W_t[t] * (VOLL - lambda_e[t]) <= K_stat_ls * r_ls[t])
 
        # ── Objective and solve ──────────────────────────────────────────────
        @objective(milp, Max, sum(w[i] * c[i] for i in I))
        optimize!(milp)
 
        # ── Extract results ──────────────────────────────────────────────────
        if !has_values(milp)
            println("MPEC FAILED: ", termination_status(milp))
            push!(results, vcat(
                [k, wtype, string(termination_status(milp)), 0.0,
                 NaN, NaN, NaN, NaN, NaN,
                 "SKIPPED", NaN, NaN, NaN, NaN, NaN, NaN, NaN],
                [NaN for _ in 1:(5 * length(Z))],
                [NaN for _ in I]))
            continue
        end
 

        for v in all_variables(milp)
            is_binary(v) && continue
            push!(all_values, (run_id = k, sigma = sigma, model = "milp",
                 variable = name(v), value = value(v)))
        end
        c_vals     = Dict(i => value(c[i])          for i in I)
        q_vals     = Dict((t, g) => value(q[t, g])  for t in T, g in IJ)
        ls_vals    = Dict(t => value(ls[t])          for t in T)
        lc_sys_vals = Dict(t => value(lambda_c_sys) for t in T)
        le_vals    = Dict(t => value(lambda_e[t])    for t in T)
        lc_vals    = Dict(z => value(lambda_c[z])    for z in Z)
        cdem_vals  = Dict(z => value(c_dem[z])       for z in Z)
        nuc_vals   = Dict(z => value(nu_c[z])        for z in Z)
        rho_vals   = Dict(z => value(rho[z])         for z in Z)
        ni_vals    = Dict(z => value(net_import[z])  for z in Z)
        reqz_vals  = Dict(z => value(req_z[z])       for z in Z)
        ft_vals    = Dict(p => value(f_trans[p])     for p in pair_keys)
        plo_vals   = Dict(p => value(phi_lo[p])      for p in pair_keys)
        phi_vals   = Dict(p => value(phi_hi[p])      for p in pair_keys)
 
        total_cap = sum(values(c_vals))
        mc = sum(W_t[t] * vc[g] * q_vals[(t, g)] for t in T for g in IJ) +
             sum(fc[i] * c_vals[i] for i in I) +
             VOLL * sum(W_t[t] * ls_vals[t] for t in T)
        le_avg   = sum(W_t[t] * le_vals[t] for t in T) / sum(W_t[t] for t in T)
        ls_total = sum(values(ls_vals))
 
        # prices
        for t in T
            push!(all_prices, (
                run_id = k, t = t, D_max = D_max[t],
                lambda_e = le_vals[t], ls = ls_vals[t],
                total_gen = sum(q_vals[(t, g)] for g in IJ)))
        end
 
        # dispatch
        for t in T
            for i in I
                push!(all_dispatch, (
                    run_id = k, t = t, generator = i, bus = gens[i],
                    q = value(q[t, i]), q_max = value(q_max_I[t, i]),
                    mu_g_up = value(mu_g_up_I[t, i]), type = "investable"))
            end
            for j in J
                push!(all_dispatch, (
                    run_id = k, t = t, generator = j, bus = gens[j],
                    q = value(q[t, j]), q_max = value(q_max_J[t, j]),
                    mu_g_up = value(mu_g_up_J[t, j]), type = "legacy"))
            end
        end
 
        # investment
        for i in I
            c_val   = c_vals[i]
            e_rent  = sum((le_vals[t] - vc[i]) * q_vals[(t, i)] for t in T)
            cm_rent = lc_vals[gen_to_zone[i]] * value(c_cm_I[i])
            fc_tot  = fc[i] * c_val
            push!(all_invest, (
                run_id = k, generator = i, bus = gens[i], tech = tech[i],
                zone = gen_to_zone[i], cap_max = cap[i], c = c_val,
                vc = vc[i], fc = fc[i],
                mu_cap_up = value(mu_cap_up[i]),
                mu_cm_bud = value(mu_cm_bud_I[i]),
                energy_rent = e_rent, cm_rent = cm_rent,
                total_rent = e_rent + cm_rent,
                fixed_cost = fc_tot,
                profit = e_rent + cm_rent - fc_tot))
        end
 
        # legacy
        for j in J
            e_rent  = sum((le_vals[t] - vc[j]) * q_vals[(t, j)] for t in T)
            cm_rent = lc_vals[gen_to_zone[j]] * value(c_cm_J[j])
            push!(all_legacy, (
                run_id = k, generator = j, bus = gens[j],
                zone = gen_to_zone[j], cap = cap[j],
                vc = vc[j], mu_cm_bud = value(mu_cm_bud_J[j]),
                energy_rent = e_rent, cm_rent = cm_rent,
                total_gen = sum(q_vals[(t, j)] for t in T)))
        end
 
        # CM detail
        for i in I
            z = gen_to_zone[i]
            push!(all_cm, (
                run_id = k, generator = i, type = "investable",
                bus = gens[i], zone = z,
                cm_offer = value(c_cm_I[i]),
                lambda_c_z = lc_vals[z],
                mu_cm_bud = value(mu_cm_bud_I[i])))
        end
        for j in J
            z = gen_to_zone[j]
            push!(all_cm, (
                run_id = k, generator = j, type = "legacy",
                bus = gens[j], zone = z,
                cm_offer = value(c_cm_J[j]),
                lambda_c_z = lc_vals[z],
                mu_cm_bud = value(mu_cm_bud_J[j])))
        end
 

        sys_avail = sum(cdem_vals[z] for z in Z)
        sys_slack = sys_avail - d_c
        for z in Z
            zone_avail  = sum((alpha[i] * c_vals[i] for i in I_in_z[z]); init = 0.0) +
                          sum((alpha[j] * cap[j]     for j in J_in_z[z]); init = 0.0)
            zone_offers = sum((value(c_cm_I[i]) for i in I_in_z[z]); init = 0.0) +
                          sum((value(c_cm_J[j]) for j in J_in_z[z]); init = 0.0)
            push!(all_cm_diag, (
                run_id = k,
                sys_avail = sys_avail, sys_req = d_c, sys_slack = sys_slack,
                zone = z, req_z = reqz_vals[z], net_import = ni_vals[z],
                zone_avail = zone_avail, zone_offers = zone_offers,
                zone_slack = zone_avail - reqz_vals[z],
                lambda_c_z = lc_vals[z], nu_c_z = nuc_vals[z]))
        end
 
        # CM demand agent detail
        for z in Z
            push!(all_cm_dem, (
                run_id = k, zone = z, d_c_z = d_c_z[z],
                net_import = ni_vals[z], req_z = reqz_vals[z],
                c_dem = cdem_vals[z], lambda_c_z = lc_vals[z],
                nu_c = nuc_vals[z], rho = rho_vals[z]))
        end
 
        # transfer detail
        for p in pair_keys
            fab = ft_vals[p]; b = mec_pairs[p]
            tag = fab >  b - 1e-3 ? "upper" :
                  fab < -b + 1e-3 ? "lower" : "interior"
            push!(all_transfer, (
                run_id = k, zone_a = p[1], zone_b = p[2], bound = b,
                f_ab = fab, phi_lo = plo_vals[p], phi_hi = phi_vals[p],
                at_bound = tag))
        end
 
        # ── Redispatch model ─────────────────────────────────────────────────
        q_max_fx = Dict((t, g) => g in I ? c_vals[g] * a[(t, g)] : cap[g] * a[(t, g)]
                        for t in T for g in IJ)
        d_val  = Dict((t, n) => d_nodal[(t, BUS[n])] * D_max[t] for t in T for n in N)
        ls_nod = Dict((t, n) => d_nodal[(t, BUS[n])] * ls_vals[t] for t in T for n in N)

        market_inj = Dict((t, n) => sum((q_vals[(t, g)] for g in IJ if gen_bus_idx[g] == n); init = 0.0)- (d_val[(t, n)] - ls_nod[(t, n)]) for t in T, n in N)

        for t in T
            injection_balance = sum(market_inj[(t, n)] for n in N)

            if abs(injection_balance) > 1e-4
                @warn "Pre-redispatch nodal injections do not balance" timestep=t balance=injection_balance
            end
        end

        market_flow = Dict(
            (t, l) =>
                sum(PTDF[l, n] * market_inj[(t, n)] for n in N)
            for t in T, l in L
        )
 
        rd = Model(Gurobi.Optimizer)
        set_optimizer_attribute(rd, "OutputFlag",      0)
        set_optimizer_attribute(rd, "FeasibilityTol", 1e-4)
        set_optimizer_attribute(rd, "DualReductions",  0)
 
        @variable(rd, r_up[t in T, g in IJ]   >= 0)
        @variable(rd, r_down[t in T, g in IJ] >= 0)
        @variable(rd, p_flow[t in T, l in L])
        @variable(rd, curt[t in T, n in N]    >= 0)
 
        @constraint(rd, [t in T, g in IJ],
            r_up[t, g]   <= max(0, q_max_fx[(t, g)] - q_vals[(t, g)]))
        @constraint(rd, [t in T, g in IJ],
            r_down[t, g] <= max(0, q_vals[(t, g)]))
        @constraint(rd, [t in T, n in N],
            curt[t, n]   <= max(0, d_val[(t, n)] - ls_nod[(t, n)]))
 
        @expression(rd, p_inj[t in T, n in N],
            sum(q_vals[(t, g)] for g in IJ if gen_bus_idx[g] == n; init = 0.0) +
            sum(r_up[t, g] - r_down[t, g] for g in IJ if gen_bus_idx[g] == n; init = 0.0) -
            (d_val[(t, n)] - ls_nod[(t, n)] - curt[t, n]))
 
        @constraint(rd, [t in T, l in L],
            p_flow[t, l] == sum(PTDF[l, n] * p_inj[t, n] for n in N))
        @constraint(rd, [t in T, l in L], p_flow[t, l] <=  Fmax[l])
        @constraint(rd, [t in T, l in L], p_flow[t, l] >= -Fmax[l])
        @constraint(rd, [t in T],
            sum(r_up[t, g] for g in IJ) - sum(r_down[t, g] for g in IJ) +
            sum(curt[t, n] for n in N) == 0)
 
        x = 75
        @objective(rd, Min,
            sum(W_t[t] * (vc[g] + x) * r_up[t, g] +
                W_t[t] * (-vc[g] + x) * r_down[t, g]
                for t in T for g in IJ) +
            PEN * sum(W_t[t] * curt[t, n] for t in T for n in N))
 
        optimize!(rd)
 
        if termination_status(rd) == MOI.OPTIMAL
            rd_up_vol   = sum(W_t[t] * value(r_up[t, g])  for t in T for g in IJ)
            rd_down_vol = sum(W_t[t] * value(r_down[t, g]) for t in T for g in IJ)
            rd_curt     = sum(W_t[t] * value(curt[t, n])  for t in T for n in N)
            rd_vol      = rd_up_vol + rd_down_vol
 
            rd_cost_econ = sum(W_t[t] * vc[g] * value(r_up[t, g])   for t in T for g in IJ) -
                           sum(W_t[t] * vc[g] * value(r_down[t, g]) for t in T for g in IJ)
            curt_cost    = PEN * rd_curt
            rd_cost      = rd_cost_econ + curt_cost
            rd_stat      = "OPTIMAL"
            for v in all_variables(rd)
                push!(all_values, (run_id = k, sigma = sigma, model = "rd",
                    variable = name(v), value = value(v)))
                end
 
            flow_tolerance = 1e-4

            for t in T, l in L
                f_market = market_flow[(t, l)]
                f_final  = value(p_flow[t, l])
                limit    = Fmax[l]

                push!(all_flows, (
                    run_id             = k,
                    t                  = t,
                    line               = LINE[l],
                    market_flow        = f_market,
                    final_flow         = f_final,
                    Fmax               = limit,
                    market_utilization = abs(f_market) / limit,
                    final_utilization  = abs(f_final) / limit,
                    market_overload    = max(0.0, abs(f_market) - limit),
                    market_violated    = abs(f_market) > limit + flow_tolerance,
                    final_feasible     = abs(f_final) <= limit + flow_tolerance
                ))
            end

            for t in T, g in IJ
                rup = value(r_up[t, g]); rdn = value(r_down[t, g]); qm = q_vals[(t, g)]
                push!(all_rd_detail, (
                    run_id = k, t = t, generator = g, bus = gens[g],
                    q_market = qm, r_up = rup, r_down = rdn,
                    q_final = qm + rup - rdn, q_max = q_max_fx[(t, g)],
                    up_cost = vc[g] * rup, down_refund = vc[g] * rdn,
                    lambda_e = le_vals[t], vc = vc[g]))
            end

            for t in T, n in N
                cv = value(curt[t, n])
                if cv > 0.01 || ls_vals[t] > 0.01
                    push!(all_curt, (
                        run_id = k, t = t, node = BUS[n],
                        demand = d_val[(t, n)], ls = ls_nod[(t, n)],
                        curtailment = cv))
                end
            end
        else
            rd_vol = NaN; rd_cost = NaN; rd_curt = NaN
            rd_up_vol = NaN; rd_down_vol = NaN; rd_cost_econ = NaN; curt_cost = NaN
            rd_stat = string(termination_status(rd))
        end
 
        # summary row
        zone_vals = Float64[]
        for z in Z
            push!(zone_vals, lc_vals[z], cdem_vals[z], nuc_vals[z],
                  reqz_vals[z], ni_vals[z])
        end
        push!(results, vcat(
            [k, wtype, string(termination_status(milp)),
             solve_time(milp), relative_gap(milp),
             total_cap, mc, le_avg, ls_total,
             rd_stat, rd_vol, rd_cost, rd_curt,
             rd_up_vol, rd_down_vol, rd_cost_econ, curt_cost],
            zone_vals,
            [c_vals[i] for i in I]))
 
        lc_str = join(["z$z=$(round(lc_vals[z], digits=1))" for z in Z], " ")
        ft_str = join(["$(p[1])-$(p[2])=$(round(ft_vals[p], digits=1))"
                       for p in pair_keys], " ")
        println("cap=$(round(total_cap, digits=0)) ",
                "λc: $lc_str ",
                "f: $ft_str ",
                "rd_up=$(round(rd_up_vol, digits=1)) ",
                "rd_down=$(round(rd_down_vol, digits=1)) ",
                "rd_cost=$(round(rd_cost_econ, digits=1)) ",
                "curt=$(round(rd_curt, digits=1)) ",
                "curt_cost=$(round(curt_cost, digits=1))")
 
    end  # run loop
 
    # ========================================================================
    # Save results
    # ========================================================================
    run_name = "final_$(scenario)_sigma$(Int(sigma * 100))"
    run_dir  = joinpath("results", run_name)
    mkpath(run_dir)
 
    CSV.write(joinpath(run_dir, "summary_runs_$(scenario)_sigma$(Int(sigma * 100)).csv"),      results)
    CSV.write(joinpath(run_dir, "prices_$(scenario)_sigma$(Int(sigma * 100)).csv"),            all_prices)
    CSV.write(joinpath(run_dir, "dispatch_$(scenario)_sigma$(Int(sigma * 100)).csv"),          all_dispatch)
    CSV.write(joinpath(run_dir, "investment_$(scenario)_sigma$(Int(sigma * 100)).csv"),        all_invest)
    CSV.write(joinpath(run_dir, "legacy_$(scenario)_sigma$(Int(sigma * 100)).csv"),            all_legacy)
    CSV.write(joinpath(run_dir, "line_flows_$(scenario)_sigma$(Int(sigma * 100)).csv"),        all_flows)
    CSV.write(joinpath(run_dir, "redispatch_detail_$(scenario)_sigma$(Int(sigma * 100)).csv"), all_rd_detail)
    CSV.write(joinpath(run_dir, "curtailment_$(scenario)_sigma$(Int(sigma * 100)).csv"),       all_curt)
    CSV.write(joinpath(run_dir, "cm_detail_$(scenario)_sigma$(Int(sigma * 100)).csv"),         all_cm)
    CSV.write(joinpath(run_dir, "cm_diagnostics_$(scenario)_sigma$(Int(sigma * 100)).csv"),    all_cm_diag)
    CSV.write(joinpath(run_dir, "cm_demand_$(scenario)_sigma$(Int(sigma * 100)).csv"),         all_cm_dem)
    CSV.write(joinpath(run_dir, "transfers_$(scenario)_sigma$(Int(sigma * 100)).csv"),         all_transfer)
    CSV.write(joinpath(run_dir, "all_values_$(scenario)_sigma$(Int(sigma * 100)).csv"),         all_values)
 
    cols = ["total_cap", "market_cost", "lambda_e_avg", "ls_total",
            "rd_volume", "rd_cost", "curtailment",
            "rd_up_vol", "rd_down_vol", "rd_cost_econ", "curt_cost",
            ["lambda_c_z$z"   for z in Z]...,
            ["c_dem_z$z"      for z in Z]...,
            ["nu_c_z$z"       for z in Z]...,
            ["req_z$z"        for z in Z]...,
            ["net_import_z$z" for z in Z]...,
            ["c_$i"           for i in I]...]
 
    valid   = results[results.mpec_status .== "OPTIMAL", cols]
    overall = describe(valid, :min, :max, :mean, :median, :std)
    CSV.write(joinpath(run_dir, "summary_overall_$(scenario)_sigma$(Int(sigma * 100)).csv"), overall)
 
    println("\nResults saved to: $run_dir")
 
end  # sigma loop