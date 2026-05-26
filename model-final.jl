using JuMP
using Gurobi
using MathOptInterface
using Random
using DataFrames
using Statistics
using CSV
 
include("helper_functions-zonal.jl")
 
# ============================================================================
# Setup
# ============================================================================
scenario = "nl34_z1"
data = load_data(scenario)
net  = build_network(data)
rep  = build_rep_days(data, net.bus_to_idx)
gen  = build_generators(data, rep, net.bus_to_idx)
(; BUS, LINE, N, L, PTDF, Fmax, bus_to_idx) = net
(; I, J, vc, cap, tech, gens, fc, a, alpha) = gen
T = rep.T
W_t = rep.w
 
D_max = Dict(t => rep.load[t] for t in T)
d_nodal = Dict((t, bus) => rep.demand[(t, net.bus_to_idx[bus])] / rep.load[t] for t in T, bus in BUS)
snapshots = Vector{String}(data.dem_df.snapshot)
 
IJ = vcat(I, J)
gen_bus_idx = Dict(g => bus_to_idx[gens[g]] for g in IJ)
 
VOLL = 300
PEN  = 2000
J_res = [j for j in J if tech[j] in ["solar", "onwind", "offwind-ac"]]

#sigmas = [0.0, 0.25, 0.5, 0.75, 1.0]
sigmas = [0.0]
for sigma in sigmas
    zones = build_zones(data, net, gen; sigma = sigma)
    (; Z, gen_to_zone, gens_in_zone, buses_in_zone, MEC) = zones


if scenario ∉ ("poc_z3", "poc_z1")
    largest_unit_z = Dict(z => maximum((cap[g] for g in vcat(I, J) if gen_to_zone[g] == z && alpha[g] > 0.0);init=0.0) for z in Z)
    d_c_z = Dict(z => maximum(sum(rep.demand[(t, n)] for n in buses_in_zone[z]; init=0.0) - 
    sum(cap[j] * a[(t, j)] for j in J_res if gen_to_zone[j] == z; init=0.0) for t in T)
    + largest_unit_z[z] for z in Z)

    # System-wide peak residual load
    largest_unit = maximum(cap[g] for g in vcat(I, J) if alpha[g] > 0.0; init=0.0)
    d_c = maximum(D_max[t] - sum(cap[j] * a[(t, j)] for j in J_res; init=0.0) for t in T) + largest_unit
 
    println("Zonal capacity requirements (d_c_z):")
    for z in Z
        println("  Zone $z: $(round(d_c_z[z], digits=1)) MW")
    end
 
    println("System-wide capacity requirement (d_c): $(round(d_c, digits=1)) MW")

else
    d_c_z = Dict(z => maximum(
    sum(rep.demand[(t, n)] for n in buses_in_zone[z]; init=0.0) -
    sum(cap[j] * a[(t, j)] for j in J_res if gen_to_zone[j] == z; init=0.0) for t in T) for z in Z)

    # System-wide peak residual load
    d_c = maximum(D_max[t] - sum(cap[j] * a[(t, j)] for j in J_res; init=0.0) for t in T)
 
    println("Zonal capacity requirements (d_c_z):")
    for z in Z
        println("  Zone $z: $(round(d_c_z[z], digits=1)) MW")
    end
 
    println("System-wide capacity requirement (d_c): $(round(d_c, digits=1)) MW")

end

# --- Total MEC import limit per zone (sum of MEC from all neighbouring zones) ---
mec_import = Dict(z => sum((get(MEC, (z2, z), 0.0) for z2 in Z if z2 != z); init=0.0) for z in Z)
 
println("MEC import limits:")
for z in Z
    println("  Zone $z: $(round(mec_import[z], digits=1)) MW")
end

# ============================================================================
# Weight vectors (generated ONCE, reused across all CM designs)
# ============================================================================
# For each line, weight generators by their node's PTDF contribution
weights_ptdf = []
for l in L
    # Push capacity toward high-PTDF nodes (promote congestion on line l)
    w_pos = Dict(i => PTDF[l, gen_bus_idx[i]] for i in I)
    # Push capacity away (relieve congestion on line l)
    w_neg = Dict(i => -PTDF[l, gen_bus_idx[i]] for i in I)
    push!(weights_ptdf, w_pos)
    push!(weights_ptdf, w_neg)
end

weights_zone = []
for z in Z
    w = Dict(i => (gen_to_zone[i] == z ? 1.0 : -1.0) for i in I)
    push!(weights_zone, w)
end

all_weights = vcat(weights_ptdf, weights_zone)
# ============================================================================
# Big-M bounds
# ============================================================================
 
# primal bounds
K_q  = maximum(cap[g] for g in IJ; init=0.0)
K_ls = maximum(D_max[t] for t in T; init=0.0)
K_c  = maximum(cap[i] for i in I; init=0.0)
K_cm = maximum(cap[g] for g in IJ; init=0.0)
 
# slack bounds
K_slack_q_I  = maximum(cap[i] for i in I; init=0.0)
K_slack_q_J  = maximum(cap[j] * a[(t, j)] for t in T for j in J; init=0.0)
K_slack_cap  = maximum(cap[i] for i in I; init=0.0)
K_slack_cm_bud_I = maximum(cap[i] for i in I; init=0.0)
K_slack_cm_bud_J = maximum(alpha[j] * cap[j] for j in J; init=0.0)
 
# dual bounds (economically motivated)
lambda_c_ub = maximum(fc[tech[i]] for i in I)
cap_up_ub   = VOLL * length(T) + lambda_c_ub
 
K_mu_g_I  = VOLL
K_mu_g_J  = VOLL
K_mu_cm_bud_I = lambda_c_ub
K_mu_cm_bud_J = lambda_c_ub
K_mu_cap  = cap_up_ub
K_mu_mec  = lambda_c_ub    # dual on MEC constraint
 
# stationarity expression bounds
K_stat_q_I = maximum(vc[i] for i in I; init=0.0) + VOLL
K_stat_q_J = maximum(vc[j] for j in J; init=0.0) + VOLL
K_stat_c   = maximum(fc[tech[i]] for i in I; init=0.0) + cap_up_ub + lambda_c_ub * length(Z) + lambda_c_ub
K_stat_cm  = lambda_c_ub + K_mu_mec
K_stat_ls  = VOLL
 
println("K_q=$K_q, K_ls=$K_ls, K_c=$K_c, K_cm=$K_cm")
println("K_stat_cm=$K_stat_cm, K_mu_mec=$K_mu_mec")
 
# ============================================================================
# Stage 1: MILP with zonal CM
# ============================================================================
 
min = Model(Gurobi.Optimizer)
set_optimizer_attribute(min, "TimeLimit", 30)
set_optimizer_attribute(min, "Presolve", 2)
set_optimizer_attribute(min, "FeasibilityTol", 1e-3)
set_optimizer_attribute(min, "MIPGap", 0.01)
set_optimizer_attribute(min, "OutputFlag", 0)
 
# --- Primal variables ---
nodal_demand  = @expression(min, d[t in T, n in N], d_nodal[(t, BUS[n])] * D_max[t])
load_shedding = @variable(min, ls[t in T] >= 0)
power         = @variable(min, q[t in T, g in IJ] >= 0)
capacity      = @variable(min, c[i in I] >= 0)
 
# CM offers: generator g offers into zone z
cm_offer_I = @variable(min, c_cm_I[i in I, z in Z] >= 0)
cm_offer_J = @variable(min, c_cm_J[j in J, z in Z] >= 0)
 
# upper bound expressions
gen_max_I = @expression(min, q_max_I[t in T, i in I], c[i])
gen_max_J = @expression(min, q_max_J[t in T, j in J], cap[j] * a[(t, j)])
 
# --- Primal constraints (energy market) ---
@constraint(min, [t in T], ls[t] <= D_max[t])
@constraint(min, [t in T], sum(q[t, g] for g in IJ) + ls[t] == D_max[t])
@constraint(min, [t in T, i in I], q[t, i] <= q_max_I[t, i])
@constraint(min, [t in T, j in J], q[t, j] <= q_max_J[t, j])
@constraint(min, [i in I], c[i] <= cap[i])
 
# --- Primal constraints (zonal capacity market) ---
 
# Zonal clearing
@constraint(min, cm_clearing[z in Z],
    sum(c_cm_I[i, z] for i in I) +
    sum(c_cm_J[j, z] for j in J) == d_c_z[z])
 
# Per-zone budget: allows double-counting across zones
@constraint(min, cm_budget_I[i in I, z in Z],
    c_cm_I[i, z] <= alpha[i] * c[i])
@constraint(min, cm_budget_J[j in J, z in Z],
    c_cm_J[j, z] <= alpha[j] * cap[j])
 
# MEC: cross-zonal imports limited
@constraint(min, mec_limit[z in Z],
    sum(c_cm_I[i, z] for i in I if gen_to_zone[i] != z) +
    sum(c_cm_J[j, z] for j in J if gen_to_zone[j] != z) <= mec_import[z])
 
# System-wide capacity floor
@constraint(min, cm_system,
    sum(alpha[i] * c[i] for i in I) +
    sum(alpha[j] * cap[j] for j in J) >= d_c)
 
# --- Dual variables ---
@variable(min, 0 <= lambda_e[t in T] <= VOLL)
@variable(min, lambda_c[z in Z])                                  # free: dual on == clearing
@variable(min, 0 <= lambda_c_sys <= lambda_c_ub)                   # dual on system floor
@variable(min, 0 <= mu_g_up_I[t in T, i in I] <= K_mu_g_I)
@variable(min, 0 <= mu_g_up_J[t in T, j in J] <= K_mu_g_J)
@variable(min, 0 <= mu_cm_bud_I[i in I, z in Z] <= K_mu_cm_bud_I) # per (i,z)
@variable(min, 0 <= mu_cm_bud_J[j in J, z in Z] <= K_mu_cm_bud_J) # per (j,z)
@variable(min, 0 <= mu_cap_up[i in I] <= cap_up_ub)
@variable(min, 0 <= mu_mec[z in Z] <= K_mu_mec)
 
# --- Stationarity conditions ---
 
# ∂L/∂q[t,i]
@constraint(min, [t in T, i in I],
    -(lambda_e[t] - vc[i]) + mu_g_up_I[t, i] >= 0)
 
# ∂L/∂q[t,j]
@constraint(min, [t in T, j in J],
    -(lambda_e[t] - vc[j]) + mu_g_up_J[t, j] >= 0)
 
# ∂L/∂c[i]: sum over z for per-zone budget duals + system dual
@constraint(min, stat_c[i in I],
    fc[tech[i]]
    - sum(mu_g_up_I[t, i] * a[(t, i)] for t in T)
    + mu_cap_up[i]
    - sum(mu_cm_bud_I[i, z] for z in Z) * alpha[i]
    - lambda_c_sys * alpha[i] >= 0)
 
# ∂L/∂c_cm_I[i,z]: per-zone budget dual
@constraint(min, stat_cm_I[i in I, z in Z],
    -lambda_c[z] + mu_cm_bud_I[i, z] +
    (gen_to_zone[i] != z ? mu_mec[z] : 0.0) >= 0)
 
# ∂L/∂c_cm_J[j,z]
@constraint(min, stat_cm_J[j in J, z in Z],
    -lambda_c[z] + mu_cm_bud_J[j, z] +
    (gen_to_zone[j] != z ? mu_mec[z] : 0.0) >= 0)
 
# ============================================================================
# Complementarity conditions (Fortuny-Amat linearisation)
# ============================================================================
 
# --- Binary variables ---
@variable(min, r_q_I[t in T, i in I], Bin)
@variable(min, r_q_J[t in T, j in J], Bin)
@variable(min, r_g_up_I[t in T, i in I], Bin)
@variable(min, r_g_up_J[t in T, j in J], Bin)
@variable(min, r_c[i in I], Bin)
@variable(min, r_cap_up[i in I], Bin)
@variable(min, r_cm_I[i in I, z in Z], Bin)
@variable(min, r_cm_J[j in J, z in Z], Bin)
@variable(min, r_cm_bud_I[i in I, z in Z], Bin)     # per (i,z)
@variable(min, r_cm_bud_J[j in J, z in Z], Bin)     # per (j,z)
@variable(min, r_mec[z in Z], Bin)
@variable(min, r_sys, Bin)                            # system constraint
@variable(min, r_ls[t in T], Bin)
 
# --- Generator I: 0 ≤ q[t,i] ⊥ (-(λ_e - vc) + μ_g_up_I) ≥ 0 ---
@constraint(min, [t in T, i in I],
    q[t, i] <= K_q * (1 - r_q_I[t, i]))
@constraint(min, [t in T, i in I],
    -(lambda_e[t] - vc[i]) + mu_g_up_I[t, i] <= K_stat_q_I * r_q_I[t, i])
 
# --- Generator I: 0 ≤ (q_max_I - q) ⊥ μ_g_up_I ≥ 0 ---
@constraint(min, [t in T, i in I],
    q_max_I[t, i] - q[t, i] <= K_slack_q_I * (1 - r_g_up_I[t, i]))
@constraint(min, [t in T, i in I],
    mu_g_up_I[t, i] <= K_mu_g_I * r_g_up_I[t, i])
 
# --- Generator I: 0 ≤ c ⊥ stat_c ≥ 0 ---
@constraint(min, [i in I],
    c[i] <= K_c * (1 - r_c[i]))
@constraint(min, [i in I],
    fc[tech[i]] - sum(mu_g_up_I[t, i] * a[(t, i)] for t in T)
    + mu_cap_up[i]
    - sum(mu_cm_bud_I[i, z] for z in Z) * alpha[i]
    - lambda_c_sys * alpha[i] <= K_stat_c * r_c[i])
 
# --- Generator I: 0 ≤ (cap - c) ⊥ μ_cap_up ≥ 0 ---
@constraint(min, [i in I],
    cap[i] - c[i] <= K_slack_cap * (1 - r_cap_up[i]))
@constraint(min, [i in I],
    mu_cap_up[i] <= K_mu_cap * r_cap_up[i])
 
# --- Generator I: 0 ≤ c_cm_I[i,z] ⊥ stat_cm_I[i,z] ≥ 0 ---
@constraint(min, [i in I, z in Z],
    c_cm_I[i, z] <= K_cm * (1 - r_cm_I[i, z]))
@constraint(min, [i in I, z in Z],
    -lambda_c[z] + mu_cm_bud_I[i, z] +
    (gen_to_zone[i] != z ? mu_mec[z] : 0.0) <= K_stat_cm * r_cm_I[i, z])
 
# --- Generator I: 0 ≤ (per-zone budget slack) ⊥ μ_cm_bud_I[i,z] ≥ 0 ---
@constraint(min, [i in I, z in Z],
    alpha[i] * c[i] - c_cm_I[i, z] <= K_slack_cm_bud_I * (1 - r_cm_bud_I[i, z]))
@constraint(min, [i in I, z in Z],
    mu_cm_bud_I[i, z] <= K_mu_cm_bud_I * r_cm_bud_I[i, z])
 
# --- Generator J: 0 ≤ q[t,j] ⊥ (-(λ_e - vc) + μ_g_up_J) ≥ 0 ---
@constraint(min, [t in T, j in J],
    q[t, j] <= K_q * (1 - r_q_J[t, j]))
@constraint(min, [t in T, j in J],
    -(lambda_e[t] - vc[j]) + mu_g_up_J[t, j] <= K_stat_q_J * r_q_J[t, j])
 
# --- Generator J: 0 ≤ (q_max_J - q) ⊥ μ_g_up_J ≥ 0 ---
@constraint(min, [t in T, j in J],
    q_max_J[t, j] - q[t, j] <= K_slack_q_J * (1 - r_g_up_J[t, j]))
@constraint(min, [t in T, j in J],
    mu_g_up_J[t, j] <= K_mu_g_J * r_g_up_J[t, j])
 
# --- Generator J: 0 ≤ c_cm_J[j,z] ⊥ stat_cm_J[j,z] ≥ 0 ---
@constraint(min, [j in J, z in Z],
    c_cm_J[j, z] <= K_cm * (1 - r_cm_J[j, z]))
@constraint(min, [j in J, z in Z],
    -lambda_c[z] + mu_cm_bud_J[j, z] +
    (gen_to_zone[j] != z ? mu_mec[z] : 0.0) <= K_stat_cm * r_cm_J[j, z])
 
# --- Generator J: 0 ≤ (per-zone budget slack) ⊥ μ_cm_bud_J[j,z] ≥ 0 ---
@constraint(min, [j in J, z in Z],
    alpha[j] * cap[j] - c_cm_J[j, z] <= K_slack_cm_bud_J * (1 - r_cm_bud_J[j, z]))
@constraint(min, [j in J, z in Z],
    mu_cm_bud_J[j, z] <= K_mu_cm_bud_J * r_cm_bud_J[j, z])
 
# --- MEC: 0 ≤ (mec slack) ⊥ μ_mec[z] ≥ 0 ---
@constraint(min, [z in Z],
    mec_import[z] - (
        sum(c_cm_I[i, z] for i in I if gen_to_zone[i] != z; init=0.0) +
        sum(c_cm_J[j, z] for j in J if gen_to_zone[j] != z; init=0.0)
    ) <= mec_import[z] * (1 - r_mec[z]))
@constraint(min, [z in Z],
    mu_mec[z] <= K_mu_mec * r_mec[z])
 
# --- System floor: 0 ≤ (system slack) ⊥ λ_c_sys ≥ 0 ---
@constraint(min,
    sum(alpha[i] * c[i] for i in I) +
    sum(alpha[j] * cap[j] for j in J) - d_c <= K_c * length(I) * (1 - r_sys))
@constraint(min,
    lambda_c_sys <= lambda_c_ub * r_sys)
 
# --- TSO: 0 ≤ ls ⊥ (VOLL - λ_e) ≥ 0 ---
@constraint(min, [t in T],
    ls[t] <= K_ls * (1 - r_ls[t]))
@constraint(min, [t in T],
    VOLL - lambda_e[t] <= K_stat_ls * r_ls[t])
 
# ============================================================================
# Result storage
# ============================================================================
 
results = DataFrame(
    run_id = Int[], weight_type = String[], mpec_status = String[],
    solve_time = Float64[], gap = Float64[],
    total_cap = Float64[], market_cost = Float64[],
    lambda_e_avg = Float64[], ls_total = Float64[],
    rd_status = String[], rd_volume = Float64[], rd_cost = Float64[], curtailment = Float64[]
)
# zonal lambda_c columns
for z in Z
    results[!, "lambda_c_z$z"] = Float64[]
end
for i in I
    results[!, "c_$i"] = Float64[]
end
 
all_prices    = DataFrame(run_id=Int[], t=Int[], D_max=Float64[], lambda_e=Float64[], ls=Float64[], total_gen=Float64[])
all_dispatch  = DataFrame(run_id=Int[], t=Int[], generator=String[], bus=String[], q=Float64[], q_max=Float64[], mu_g_up=Float64[], type=String[])
all_invest    = DataFrame(run_id=Int[], generator=String[], bus=String[], tech=String[], zone=Int[], cap_max=Float64[], c=Float64[], vc=Float64[], fc=Float64[], mu_cap_up=Float64[], mu_cm_bud_total=Float64[], lambda_c_sys=Float64[], energy_rent=Float64[], cm_rent=Float64[], total_rent=Float64[], fixed_cost=Float64[], profit=Float64[])
all_legacy    = DataFrame(run_id=Int[], generator=String[], bus=String[], zone=Int[], cap=Float64[], vc=Float64[], mu_cm_bud_total=Float64[], energy_rent=Float64[], cm_rent=Float64[], total_gen=Float64[])
all_flows     = DataFrame(run_id=Int[], t=Int[], line=String[], flow=Float64[], Fmax=Float64[], utilization=Float64[], congested=Bool[])
all_rd_detail = DataFrame(run_id=Int[], t=Int[], generator=String[], bus=String[], q_market=Float64[], r_up=Float64[], r_down=Float64[], q_final=Float64[], q_max=Float64[], up_cost=Float64[], down_compensation=Float64[], lambda_e=Float64[], vc=Float64[])
all_curt      = DataFrame(run_id=Int[], t=Int[], node=String[], demand=Float64[], ls=Float64[], curtailment=Float64[])
all_cm        = DataFrame(run_id=Int[], generator=String[], type=String[], bus=String[], gen_zone=Int[], target_zone=Int[], cm_offer=Float64[], lambda_c_z=Float64[], mu_cm_bud=Float64[], mu_mec_z=Float64[])
 
println("\n" * "=" ^ 60)
println("Starting $(length(all_weights)) runs (sigma=$sigma)...")
println("=" ^ 60)
 
for (k, w) in enumerate(all_weights)
    local d_val, rd, r_up, r_down, p_flow, curt, p_inj
    wtype = k <= length(I) ? "unit_pos" :
            k <= 2*length(I) ? "unit_neg" : "random"
 
    print("Run $k/$(length(all_weights)) ($wtype)... ")
 
    @objective(min, Max, sum(w[i] * c[i] for i in I))
    set_optimizer_attribute(min, "DualReductions", 0)
    optimize!(min)
 
    if !has_values(min)
        println("MPEC FAILED: ", termination_status(min))
        push!(results, vcat(
            [k, wtype, string(termination_status(min)), 0.0, NaN, NaN, NaN, NaN, NaN,
             "SKIPPED", NaN, NaN, NaN],
            [NaN for _ in Z],
            [NaN for _ in I]))
        continue
    end
 
    c_vals  = Dict(i => value(c[i]) for i in I)
    q_vals  = Dict((t, g) => value(q[t, g]) for t in T, g in IJ)
    ls_vals = Dict(t => value(ls[t]) for t in T)
    le_vals = Dict(t => value(lambda_e[t]) for t in T)
    lc_vals = Dict(z => value(lambda_c[z]) for z in Z)
    lc_sys_val = value(lambda_c_sys)
 
    total_cap = sum(values(c_vals))
    mc = sum(W_t[t] * vc[g] * q_vals[(t, g)] for t in T for g in IJ) +
         sum(fc[tech[i]] * c_vals[i] for i in I) +
         VOLL * sum(W_t[t] * ls_vals[t] for t in T)
    le_avg = sum(W_t[t] * le_vals[t] for t in T) / sum(W_t[t] for t in T)
    ls_total = sum(values(ls_vals))
 
    # --- prices ---
    for t in T
        push!(all_prices, (
            run_id = k, t = t, D_max = D_max[t],
            lambda_e = le_vals[t], ls = ls_vals[t],
            total_gen = sum(q_vals[(t, g)] for g in IJ)
        ))
    end
 
    # --- dispatch ---
    for t in T
        for i in I
            push!(all_dispatch, (
                run_id = k, t = t, generator = i, bus = gens[i],
                q = value(q[t, i]), q_max = value(q_max_I[t, i]),
                mu_g_up = value(mu_g_up_I[t, i]), type = "investable"
            ))
        end
        for j in J
            push!(all_dispatch, (
                run_id = k, t = t, generator = j, bus = gens[j],
                q = value(q[t, j]), q_max = value(q_max_J[t, j]),
                mu_g_up = value(mu_g_up_J[t, j]), type = "legacy"
            ))
        end
    end
 
    # --- investment ---
    for i in I
        c_val = c_vals[i]
        e_rent = sum((le_vals[t] - vc[i]) * q_vals[(t, i)] for t in T)
        cm_rent = sum(lc_vals[z] * value(c_cm_I[i, z]) for z in Z)
        fc_total = fc[tech[i]] * c_val
        push!(all_invest, (
            run_id = k, generator = i, bus = gens[i], tech = tech[i],
            zone = gen_to_zone[i],
            cap_max = cap[i], c = c_val,
            vc = vc[i], fc = fc[tech[i]],
            mu_cap_up = value(mu_cap_up[i]),
            mu_cm_bud_total = sum(value(mu_cm_bud_I[i, z]) for z in Z),
            lambda_c_sys = lc_sys_val,
            energy_rent = e_rent, cm_rent = cm_rent,
            total_rent = e_rent + cm_rent,
            fixed_cost = fc_total,
            profit = e_rent + cm_rent - fc_total
        ))
    end
 
    # --- legacy ---
    for j in J
        e_rent = sum((le_vals[t] - vc[j]) * q_vals[(t, j)] for t in T)
        cm_rent = sum(lc_vals[z] * value(c_cm_J[j, z]) for z in Z)
        push!(all_legacy, (
            run_id = k, generator = j, bus = gens[j],
            zone = gen_to_zone[j],
            cap = cap[j],
            vc = vc[j], mu_cm_bud_total = sum(value(mu_cm_bud_J[j, z]) for z in Z),
            energy_rent = e_rent, cm_rent = cm_rent,
            total_gen = sum(q_vals[(t, j)] for t in T)
        ))
    end
 
    # --- capacity market detail (per generator × zone) ---
    for i in I, z in Z
        offer = value(c_cm_I[i, z])
        if offer > 1e-3 || gen_to_zone[i] == z
            push!(all_cm, (
                run_id = k, generator = i, type = "investable",
                bus = gens[i], gen_zone = gen_to_zone[i], target_zone = z,
                cm_offer = offer,
                lambda_c_z = lc_vals[z],
                mu_cm_bud = value(mu_cm_bud_I[i, z]),
                mu_mec_z = value(mu_mec[z])
            ))
        end
    end
    for j in J, z in Z
        offer = value(c_cm_J[j, z])
        if offer > 1e-3 || gen_to_zone[j] == z
            push!(all_cm, (
                run_id = k, generator = j, type = "legacy",
                bus = gens[j], gen_zone = gen_to_zone[j], target_zone = z,
                cm_offer = offer,
                lambda_c_z = lc_vals[z],
                mu_cm_bud = value(mu_cm_bud_J[j, z]),
                mu_mec_z = value(mu_mec[z])
            ))
        end
    end
 
    # ================================================================
    # Redispatch model
    # ================================================================
    q_max_fx = Dict((t, g) => g in I ? c_vals[g] * a[(t, g)] : cap[g] * a[(t, g)]
                for t in T for g in IJ)
    d_val    = Dict((t, n) => d_nodal[(t, BUS[n])] * D_max[t] for t in T for n in N)
    ls_nod   = Dict((t, n) => d_nodal[(t, BUS[n])] * ls_vals[t] for t in T for n in N)
 
    rd = Model(Gurobi.Optimizer)
    set_optimizer_attribute(rd, "OutputFlag", 0)
    set_optimizer_attribute(rd, "FeasibilityTol", 1e-4)
 
    @variable(rd, r_up[t in T, g in IJ] >= 0)
    @variable(rd, r_down[t in T, g in IJ] >= 0)
    @variable(rd, p_flow[t in T, l in L])
    @variable(rd, curt[t in T, n in N] >= 0)
 
    @constraint(rd, [t in T, g in IJ], r_up[t, g] <= max(0, q_max_fx[(t, g)] - q_vals[(t, g)]))
    @constraint(rd, [t in T, g in IJ], r_down[t, g] <= max(0, q_vals[(t, g)]))
    @constraint(rd, [t in T, n in N], curt[t, n] <= max(0, d_val[(t, n)] - ls_nod[(t, n)]))
 
    @expression(rd, p_inj[t in T, n in N],
        sum(q_vals[(t, g)] for g in IJ if gen_bus_idx[g] == n; init=0.0) +
        sum(r_up[t, g] - r_down[t, g] for g in IJ if gen_bus_idx[g] == n; init=0.0) -
        (d_val[(t, n)] - ls_nod[(t, n)] - curt[t, n]))
 
    @constraint(rd, [t in T, l in L],
        p_flow[t, l] == sum(PTDF[l, n] * p_inj[t, n] for n in N))
    @constraint(rd, [t in T, l in L], p_flow[t, l] <=  Fmax[l])
    @constraint(rd, [t in T, l in L], p_flow[t, l] >= -Fmax[l])
    @constraint(rd, [t in T],
        sum(r_up[t, g] for g in IJ) - sum(r_down[t, g] for g in IJ) + sum(curt[t, n] for n in N) == 0)
 
    x = 1000
    @objective(rd, Min,
        sum(W_t[t] * (vc[g] + x) * r_up[t, g] + W_t[t] * (-vc[g] + x) * r_down[t, g]
            for t in T for g in IJ) +
        PEN * sum(W_t[t] * curt[t, n] for t in T for n in N))
    set_optimizer_attribute(rd, "DualReductions", 0)
    optimize!(rd)
 
    if termination_status(rd) == MOI.OPTIMAL
        rd_vol  = sum(W_t[t] * (value(r_up[t, g]) + value(r_down[t, g])) for t in T for g in IJ)
        rd_cost = sum(W_t[t] * (vc[g] * value(r_up[t, g]) + (le_vals[t] - vc[g]) * value(r_down[t, g]))
                      for t in T for g in IJ) +
                  PEN * sum(W_t[t] * value(curt[t, n]) for t in T for n in N)
        rd_curt = sum(W_t[t] * value(curt[t, n]) for t in T for n in N)
        rd_stat = "OPTIMAL"
 
        for t in T, l in L
            f = value(p_flow[t, l])
            push!(all_flows, (
                run_id = k, t = t, line = LINE[l],
                flow = f, Fmax = Fmax[l],
                utilization = abs(f) / (Fmax[l] ),
                congested = abs(f) >= Fmax[l]  * 0.99
            ))
        end
        for t in T, g in IJ
            rup = value(r_up[t, g]); rdn = value(r_down[t, g]); qm = q_vals[(t, g)]
            push!(all_rd_detail, (
                run_id = k, t = t, generator = g, bus = gens[g],
                q_market = qm, r_up = rup, r_down = rdn,
                q_final = qm + rup - rdn, q_max = q_max_fx[(t, g)],
                up_cost = vc[g] * rup,
                down_compensation = (le_vals[t] - vc[g]) * rdn,
                lambda_e = le_vals[t], vc = vc[g]
            ))
        end
        for t in T, n in N
            cv = value(curt[t, n])
            if cv > 0.01 || ls_vals[t] > 0.01
                push!(all_curt, (
                    run_id = k, t = t, node = BUS[n],
                    demand = d_val[(t, n)], ls = ls_nod[(t, n)], curtailment = cv
                ))
            end
        end
    else
        rd_vol = NaN; rd_cost = NaN; rd_curt = NaN
        rd_stat = string(termination_status(rd))
    end
 
    # --- summarize ---
    push!(results, vcat(
        [k, wtype, string(termination_status(min)),
         solve_time(min), relative_gap(min),
         total_cap, mc, le_avg, ls_total,
         rd_stat, rd_vol, rd_cost, rd_curt],
        [lc_vals[z] for z in Z],
        [c_vals[i] for i in I]))
 
    lc_str = join(["z$z=$(round(lc_vals[z],digits=1))" for z in Z], " ")
    println("cap=$(round(total_cap, digits=0)) ",
            "λc: $lc_str ",
            "λc_sys=$(round(lc_sys_val, digits=1)) ",
            "ls=$(round(ls_total, digits=1)) ",
            "rd_vol=$(round(rd_vol, digits=1)) ",
            "rd_cost=$(round(rd_cost, digits=1)) ",
            "curt=$(round(rd_curt, digits=1))")
end
 
# ============================================================================
# Save results
# ============================================================================
 
run_name = "$(scenario)_sigma$(Int(sigma*100))"
run_dir  = joinpath("results", run_name)
mkpath(run_dir)
 
CSV.write(joinpath(run_dir, "summary_runs.csv"), results)
CSV.write(joinpath(run_dir, "prices.csv"), all_prices)
CSV.write(joinpath(run_dir, "dispatch.csv"), all_dispatch)
CSV.write(joinpath(run_dir, "investment.csv"), all_invest)
CSV.write(joinpath(run_dir, "legacy.csv"), all_legacy)
CSV.write(joinpath(run_dir, "line_flows.csv"), all_flows)
CSV.write(joinpath(run_dir, "redispatch_detail.csv"), all_rd_detail)
CSV.write(joinpath(run_dir, "curtailment.csv"), all_curt)
CSV.write(joinpath(run_dir, "cm_detail.csv"), all_cm)
 
cols = ["total_cap", "market_cost", "lambda_e_avg", "ls_total",
        "rd_volume", "rd_cost", "curtailment",
        ["lambda_c_z$z" for z in Z]...,
        ["c_$i" for i in I]...]
 
valid = results[results.mpec_status .== "OPTIMAL", cols]
overall = describe(valid, :min, :max, :mean, :median, :std)
 
CSV.write(joinpath(run_dir, "summary_overall.csv"), overall)
 
println("\nResults saved to: $run_dir")
end

