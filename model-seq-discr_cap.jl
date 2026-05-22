using JuMP
using Gurobi
using MathOptInterface
using Random
using DataFrames
using CSV

include("helper_functions.jl")

# --- Setup ---
T    = vcat(3100:3110)  
data = load_data("nl34")
net  = build_network(data)
gen  = build_generators(data, T, net.bus_to_idx)

(; BUS, LINE, N, L, PTDF, Fmax, bus_to_idx) = net
(; I, J, vc, cap, tech, gens, fc, a, alpha) = gen

D_max     = Dict(t => Float64(data.dem_tot_df.load[t]) * 1.2 for t in T)
snapshots = Vector{String}(data.dem_df.snapshot)


IJ = vcat(I, J)
gen_bus_idx = Dict(g => bus_to_idx[gens[g]] for g in IJ)

VOLL = 300
PEN  = 2000
d_c  = 0


d_nodal = Dict((t, bus) => data.dem_df[data.dem_df.snapshot .== snapshots[t], bus][1] / Float64(data.dem_tot_df.load[t]) for t in T, bus in BUS)
println(sum(d_nodal[(t, bus)] for bus in BUS for t in T) / length(T))
for t in T
    @assert sum(d_nodal[(t, bus)] for bus in BUS) ≈ 1
end

# ============================================================================
# Weight vectors (generated ONCE, reused across all CM designs)
# ============================================================================

Random.seed!(42)
K_random = 5

weights_unit   = [Dict(i => (i == k ? 1.0 : 0.0) for i in I) for k in I]
weights_neg    = [Dict(i => (i == k ? -1.0 : 0.0) for i in I) for k in I]
weights_random = [Dict(i => 2*rand()-1 for i in I) for _ in 1:K_random]
all_weights    = vcat(weights_unit, weights_neg, weights_random)

println("Total weight vectors: ", length(all_weights))
 
# primal bounds
K_q  = maximum(cap[g] for g in IJ; init=0.0)
K_ls = maximum(D_max[t] for t in T; init=0.0)
K_c  = maximum(cap[i] for i in I; init=0.0)
K_cm = maximum(cap[g] for g in IJ; init=0.0)
 
#slack bounds
K_slack_q_I  = maximum(cap[i] for i in I; init=0.0)
K_slack_q_J  = maximum(cap[j] * a[(t, j)] for t in T for j in J; init=0.0)
K_slack_cap  = maximum(cap[i] for i in I; init=0.0)
K_slack_cm_I = maximum(cap[i] for i in I; init=0.0)          # cm_I ∈ [0, c[i]] ≤ cap[i]
K_slack_cm_J = maximum(alpha[j] * cap[j] for j in J; init=0.0)
 
# dual bounds (economically motivated)
lambda_c_ub = maximum(fc[tech[i]] for i in I)
cap_up_ub   = VOLL * length(T) + lambda_c_ub

K_mu_g_I  = VOLL
K_mu_g_J  = VOLL
K_mu_cm_I = lambda_c_ub
K_mu_cm_J = lambda_c_ub
K_mu_cap  = cap_up_ub
 
# stationarity expression bounds
# stat_q: -(λ_e - vc) + μ_g_up ≥ 0  →  max value = max(vc) + VOLL
K_stat_q_I = maximum(vc[i] for i in I; init=0.0) + VOLL
K_stat_q_J = maximum(vc[j] for j in J; init=0.0) + VOLL
# stat_c: fc - Σ(μ_g_up * a) + μ_cap - μ_cm * α ≥ 0  →  max = fc_max + cap_up_ub
K_stat_c   = maximum(fc[tech[i]] for i in I; init=0.0) + cap_up_ub
# stat_cm: -λ_c + μ_cm_up ≥ 0  →  max = lambda_c_ub (since both bounded by it)
K_stat_cm   = lambda_c_ub
K_stat_cm_J = lambda_c_ub
# stat_ls: VOLL - λ_e ≥ 0  →  max = VOLL (when λ_e = 0)
K_stat_ls = VOLL
 

println("K_q = $K_q, K_ls = $K_ls, K_c = $K_c, K_cm = $K_cm")
println("K_slack_q_I = $K_slack_q_I, K_slack_q_J = $K_slack_q_J, K_slack_cap = $K_slack_cap")
println("K_slack_cm_I = $K_slack_cm_I, K_slack_cm_J = $K_slack_cm_J")
println("K_mu_g_I = $K_mu_g_I, K_mu_g_J = $K_mu_g_J, K_mu_cm_I = $K_mu_cm_I, K_mu_cm_J = $K_mu_cm_J, K_mu_cap = $K_mu_cap")
println("K_stat_q_I = $K_stat_q_I, K_stat_q_J = $K_stat_q_J, K_stat_c = $K_stat_c")
println("K_stat_cm = $K_stat_cm, K_stat_ls = $K_stat_ls")
 
# =============================================================================
# Stage 1: MILP with tight bounds
# =============================================================================
 
min = Model(Gurobi.Optimizer)
set_optimizer_attribute(min, "TimeLimit", 30)
set_optimizer_attribute(min, "Presolve", 2)
set_optimizer_attribute(min, "FeasibilityTol", 1e-3)
set_optimizer_attribute(min, "MIPGap", 0.01)
set_optimizer_attribute(min, "OutputFlag", 0)
#=
delta_C = 100.0
max_steps = Dict(i => Int(floor(cap[i] / delta_C)) for i in I)
z_inv = @variable(min, z_inv[i in I], Int, lower_bound=0, upper_bound=max_steps[i])
capacity = @expression(min, c[i in I], delta_C * z_inv[i])
=#
nodal_demand = @expression(min, d[t in T, n in N], d_nodal[(t, BUS[n])] * D_max[t])
load_shedding = @variable(min, ls[t in T] >= 0)
power = @variable(min, q[t in T, g in IJ] >= 0)
capacity = @variable(min, c[i in I] >= 0)
cm_offer_I = @variable(min, c_cm_I[i in I] >= 0)
cm_offer_J = @variable(min, c_cm_J[j in J] >= 0)
gen_max_I = @expression(min, q_max_I[t in T, i in I], c[i])
gen_max_J = @expression(min, q_max_J[t in T, j in J], cap[j] * a[(t, j)])
cm_max_I = @expression(min, cap_max_cm_I[i in I], c[i])
cm_max_J = @expression(min, cap_max_cm_J[j in J], alpha[j] * cap[j])
 
@constraint(min, [t in T], ls[t] <= D_max[t])
@constraint(min, [t in T], sum(q[t, g] for g in IJ) + ls[t] == D_max[t])
@constraint(min, sum(c_cm_I[i] for i in I) + sum(c_cm_J[j] for j in J) == d_c)
@constraint(min, [t in T, i in I], q[t, i] <= q_max_I[t, i])
@constraint(min, [t in T, j in J], q[t, j] <= q_max_J[t, j])
@constraint(min, [i in I], c_cm_I[i] <= cap_max_cm_I[i])
@constraint(min, [j in J], c_cm_J[j] <= cap_max_cm_J[j])
@constraint(min, [i in I], c[i] <= cap[i])
 
# dual variables
@variable(min, 0 <= lambda_e[t in T] <= VOLL)
@variable(min, 0 <= lambda_c <= lambda_c_ub)
@variable(min, VOLL >= mu_g_up_I[t in T, i in I] >= 0)
@variable(min, VOLL >= mu_g_up_J[t in T, j in J] >= 0)
@variable(min, lambda_c_ub >= mu_cm_up_I[i in I] >= 0)
@variable(min, lambda_c_ub >= mu_cm_up_J[j in J] >= 0)
@variable(min, cap_up_ub >= mu_cap_up[i in I] >= 0)
 

# stationarity conditions
@constraint(min, [t in T, i in I], -(lambda_e[t] - vc[i]) + mu_g_up_I[t, i] >= 0)
@constraint(min, [t in T, j in J], -(lambda_e[t] - vc[j]) + mu_g_up_J[t, j] >= 0)
@constraint(min, [i in I], fc[tech[i]] - sum(mu_g_up_I[t, i] * a[(t, i)] for t in T) + mu_cap_up[i] - mu_cm_up_I[i] * alpha[i] >= 0)
@constraint(min, [i in I], -lambda_c + mu_cm_up_I[i] >= 0)
@constraint(min, [j in J], -lambda_c + mu_cm_up_J[j] >= 0)
 
 
# --- Binary variables ---
@variable(min, r_q_I[t in T, i in I], Bin)
@variable(min, r_q_J[t in T, j in J], Bin)
@variable(min, r_g_up_I[t in T, i in I], Bin)
@variable(min, r_g_up_J[t in T, j in J], Bin)
@variable(min, r_c[i in I], Bin)
@variable(min, r_cap_up[i in I], Bin)
@variable(min, r_cm_I[i in I], Bin)
@variable(min, r_cm_J[j in J], Bin)
@variable(min, r_cm_up_I[i in I], Bin)
@variable(min, r_cm_up_J[j in J], Bin)
@variable(min, r_ls[t in T], Bin)
 
# --- Generator I: 0 ≤ q[t,i] ⊥ (-(λ_e - vc) + μ_g_up_I) ≥ 0 ---
@constraint(min, [t in T, i in I], q[t, i] <= K_q * (1 - r_q_I[t, i]))
@constraint(min, [t in T, i in I], -(lambda_e[t] - vc[i]) + mu_g_up_I[t, i] <= K_stat_q_I * r_q_I[t, i])
 
# --- Generator I: 0 ≤ (q_max_I - q) ⊥ μ_g_up_I ≥ 0 ---
@constraint(min, [t in T, i in I], q_max_I[t, i] - q[t, i] <= K_slack_q_I * (1 - r_g_up_I[t, i]))
@constraint(min, [t in T, i in I], mu_g_up_I[t, i] <= K_mu_g_I * r_g_up_I[t, i])
 
# --- Generator I: 0 ≤ c ⊥ (fc - Σμ·a + μ_cap - μ_cm·α) ≥ 0 ---
@constraint(min, [i in I], c[i] <= K_c * (1 - r_c[i]))
@constraint(min, [i in I], fc[tech[i]] - sum(mu_g_up_I[t, i] * a[(t, i)] for t in T) + mu_cap_up[i] - mu_cm_up_I[i] * alpha[i] <= K_stat_c * r_c[i])
 
# --- Generator I: 0 ≤ (cap - c) ⊥ μ_cap_up ≥ 0 ---
@constraint(min, [i in I], cap[i] - c[i] <= K_slack_cap * (1 - r_cap_up[i]))
@constraint(min, [i in I], mu_cap_up[i] <= K_mu_cap * r_cap_up[i])
 
# --- Generator I: 0 ≤ c_cm_I ⊥ (-λ_c + μ_cm_up_I) ≥ 0 ---
@constraint(min, [i in I], c_cm_I[i] <= K_cm * (1 - r_cm_I[i]))
@constraint(min, [i in I], -lambda_c + mu_cm_up_I[i] <= K_stat_cm * r_cm_I[i])
 
# --- Generator I: 0 ≤ (cm_max_I - c_cm_I) ⊥ μ_cm_up_I ≥ 0 ---
@constraint(min, [i in I], cap_max_cm_I[i] - c_cm_I[i] <= K_slack_cm_I * (1 - r_cm_up_I[i]))
@constraint(min, [i in I], mu_cm_up_I[i] <= K_mu_cm_I * r_cm_up_I[i])
 
# --- Generator J: 0 ≤ q[t,j] ⊥ (-(λ_e - vc) + μ_g_up_J) ≥ 0 ---
@constraint(min, [t in T, j in J], q[t, j] <= K_q * (1 - r_q_J[t, j]))
@constraint(min, [t in T, j in J], -(lambda_e[t] - vc[j]) + mu_g_up_J[t, j] <= K_stat_q_J * r_q_J[t, j])
 
# --- Generator J: 0 ≤ (q_max_J - q) ⊥ μ_g_up_J ≥ 0 ---
@constraint(min, [t in T, j in J], q_max_J[t, j] - q[t, j] <= K_slack_q_J * (1 - r_g_up_J[t, j]))
@constraint(min, [t in T, j in J], mu_g_up_J[t, j] <= K_mu_g_J * r_g_up_J[t, j])
 
# --- Generator J: 0 ≤ c_cm_J ⊥ (-λ_c + μ_cm_up_J) ≥ 0 ---
@constraint(min, [j in J], c_cm_J[j] <= K_cm * (1 - r_cm_J[j]))
@constraint(min, [j in J], -lambda_c + mu_cm_up_J[j] <= K_stat_cm_J * r_cm_J[j])
 
# --- Generator J: 0 ≤ (cm_max_J - c_cm_J) ⊥ μ_cm_up_J ≥ 0 ---
@constraint(min, [j in J], cap_max_cm_J[j] - c_cm_J[j] <= K_slack_cm_J * (1 - r_cm_up_J[j]))
@constraint(min, [j in J], mu_cm_up_J[j] <= K_mu_cm_J * r_cm_up_J[j])
 
# --- TSO: 0 ≤ ls ⊥ (VOLL - λ_e) ≥ 0 ---
@constraint(min, [t in T], ls[t] <= K_ls * (1 - r_ls[t]))
@constraint(min, [t in T], VOLL - lambda_e[t] <= K_stat_ls * r_ls[t])

results = DataFrame(
    run_id = Int[], weight_type = String[], mpec_status = String[],
    solve_time = Float64[], gap = Float64[],
    total_cap = Float64[], market_cost = Float64[],
    lambda_e_avg = Float64[], lambda_c_val = Float64[], ls_total = Float64[],
    rd_status = String[], rd_volume = Float64[], rd_cost = Float64[], curtailment = Float64[]
)
for i in I
    results[!, "c_$i"] = Float64[]
end
 
all_prices    = DataFrame(run_id=Int[], t=Int[], D_max=Float64[], lambda_e=Float64[], ls=Float64[], total_gen=Float64[])
all_dispatch  = DataFrame(run_id=Int[], t=Int[], generator=String[], bus=String[], q=Float64[], q_max=Float64[], mu_g_up=Float64[], type=String[])
all_invest    = DataFrame(run_id=Int[], generator=String[], bus=String[], tech=String[], cap_max=Float64[], c=Float64[], c_cm=Float64[], vc=Float64[], fc=Float64[], mu_cap_up=Float64[], mu_cm_up=Float64[], lambda_c=Float64[], energy_rent=Float64[], cm_rent=Float64[], total_rent=Float64[], fixed_cost=Float64[], profit=Float64[])
all_legacy    = DataFrame(run_id=Int[], generator=String[], bus=String[], cap=Float64[], c_cm=Float64[], vc=Float64[], mu_cm_up=Float64[], energy_rent=Float64[], cm_rent=Float64[], total_gen=Float64[])
all_flows     = DataFrame(run_id=Int[], t=Int[], line=String[], flow=Float64[], Fmax=Float64[], utilization=Float64[], congested=Bool[])
all_rd_detail = DataFrame(run_id=Int[], t=Int[], generator=String[], bus=String[], q_market=Float64[], r_up=Float64[], r_down=Float64[], q_final=Float64[], q_max=Float64[], up_cost=Float64[], down_compensation=Float64[], lambda_e=Float64[], vc=Float64[])
all_curt      = DataFrame(run_id=Int[], t=Int[], node=String[], demand=Float64[], ls=Float64[], curtailment=Float64[])
all_cm        = DataFrame(run_id=Int[], generator=String[], type=String[], bus=String[], capacity=Float64[], cm_offer=Float64[], cm_max=Float64[], alpha=Float64[], mu_cm_up=Float64[])
 
println("\n" * "=" ^ 60)
println("Starting $(length(all_weights)) runs...")
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
            [k, wtype, string(termination_status(min)), 0.0, NaN, NaN, NaN, NaN, NaN, NaN,
             "SKIPPED", NaN, NaN, NaN],
            [NaN for _ in I]))
        continue
    end
 
    c_vals  = Dict(i => value(c[i]) for i in I)
    q_vals  = Dict((t, g) => value(q[t, g]) for t in T, g in IJ)
    ls_vals = Dict(t => value(ls[t]) for t in T)
    le_vals = Dict(t => value(lambda_e[t]) for t in T)
    lc_val  = value(lambda_c)
 
    total_cap = sum(values(c_vals))
    mc = sum(vc[g] * q_vals[(t, g)] for t in T for g in IJ) +
         sum(fc[tech[i]] * c_vals[i] for i in I) +
         VOLL * sum(values(ls_vals))
    le_avg   = sum(values(le_vals)) / length(T)
    ls_total = sum(values(ls_vals))
 
    # prices
    for t in T
        push!(all_prices, (
            run_id = k, t = t, D_max = D_max[t],
            lambda_e = le_vals[t], ls = ls_vals[t],
            total_gen = sum(q_vals[(t, g)] for g in IJ)
        ))
    end
 
    # dispatch
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
 
    # investment
    for i in I
        c_val = c_vals[i]
        ccm_val = value(c_cm_I[i])
        e_rent = sum((le_vals[t] - vc[i]) * q_vals[(t, i)] for t in T)
        cm_rent = lc_val * ccm_val
        fc_total = fc[tech[i]] * c_val
        push!(all_invest, (
            run_id = k, generator = i, bus = gens[i], tech = tech[i],
            cap_max = cap[i], c = c_val, c_cm = ccm_val,
            vc = vc[i], fc = fc[tech[i]],
            mu_cap_up = value(mu_cap_up[i]), mu_cm_up = value(mu_cm_up_I[i]),
            lambda_c = lc_val,
            energy_rent = e_rent, cm_rent = cm_rent,
            total_rent = e_rent + cm_rent,
            fixed_cost = fc_total,
            profit = e_rent + cm_rent - fc_total
        ))
    end
 
    # legacy gens
    for j in J
        ccm_val = value(c_cm_J[j])
        e_rent = sum((le_vals[t] - vc[j]) * q_vals[(t, j)] for t in T)
        cm_rent = lc_val * ccm_val
        push!(all_legacy, (
            run_id = k, generator = j, bus = gens[j],
            cap = cap[j], c_cm = ccm_val,
            vc = vc[j], mu_cm_up = value(mu_cm_up_J[j]),
            energy_rent = e_rent, cm_rent = cm_rent,
            total_gen = sum(q_vals[(t, j)] for t in T)
        ))
    end
 
    # capacity market
    for i in I
        push!(all_cm, (
            run_id = k, generator = i, type = "investable", bus = gens[i],
            capacity = c_vals[i], cm_offer = value(c_cm_I[i]),
            cm_max = value(cap_max_cm_I[i]),
            alpha = alpha[i], mu_cm_up = value(mu_cm_up_I[i])
        ))
    end
    for j in J
        push!(all_cm, (
            run_id = k, generator = j, type = "legacy", bus = gens[j],
            capacity = cap[j], cm_offer = value(c_cm_J[j]),
            cm_max = alpha[j] * cap[j],
            alpha = alpha[j], mu_cm_up = value(mu_cm_up_J[j])
        ))
    end
 
    # redispatch model
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
    @constraint(rd, [t in T, l in L], p_flow[t, l] <= Fmax[l] * 0.5)
    @constraint(rd, [t in T, l in L], p_flow[t, l] >= -Fmax[l] * 0.5)

    @constraint(rd, [t in T],
    sum(r_up[t, g] for g in IJ) - sum(r_down[t, g] for g in IJ) + sum(curt[t, n] for n in N) == 0)

    # mark up redispatch x, to ensure it is only done due to grid constraints, not economic incentives
    x = 1000
    #@objective(rd, Min, sum((vc[g] + x) * r_up[t, g] + (- vc[g] + x) * r_down[t, g] for t in T for g in IJ) + PEN * sum(curt[t, n] for t in T for n in N))
    @objective(rd, Min, sum(vc[g] * r_up[t, g] + (le_vals[t] - vc[g]) * r_down[t, g] for t in T for g in IJ) + PEN * sum(curt[t, n] for t in T for n in N))
    set_optimizer_attribute(rd, "DualReductions", 0)
    optimize!(rd)
 
    if termination_status(rd) == MOI.OPTIMAL
        rd_vol  = sum(value(r_up[t, g]) + value(r_down[t, g]) for t in T for g in IJ)
        rd_cost = sum(vc[g] * value(r_up[t, g]) + (le_vals[t] - vc[g]) * value(r_down[t, g]) for t in T for g in IJ) + PEN * sum(value(curt[t, n]) for t in T for n in N)
        rd_curt = sum(value(curt[t, n]) for t in T for n in N)
        rd_stat = "OPTIMAL"
 
        # line flows
        for t in T
            for l in L
                f = value(p_flow[t, l])
                push!(all_flows, (
                    run_id = k, t = t, line = LINE[l],
                    flow = f, Fmax = Fmax[l] * 0.5,
                    utilization = abs(f) / (Fmax[l] * 0.5),
                    congested = abs(f) >= Fmax[l] * 0.5 * 0.99
                ))
            end
        end
 
        # redispatch details
        for t in T
            for g in IJ
                rup = value(r_up[t, g])
                rdn = value(r_down[t, g])
                qm = q_vals[(t, g)]
                push!(all_rd_detail, (
                    run_id = k, t = t, generator = g, bus = gens[g],
                    q_market = qm,
                    r_up = rup, r_down = rdn,
                    q_final = qm + rup - rdn,
                    q_max = q_max_fx[(t, g)],
                    up_cost = vc[g] * rup,
                    down_compensation = (le_vals[t] - vc[g]) * rdn,
                    lambda_e = le_vals[t], vc = vc[g]
                ))
            end
        end
 
        # curtailment
        for t in T
            for n in N
                cv = value(curt[t, n])
                if cv > 0.01 || ls_vals[t] > 0.01
                    push!(all_curt, (
                        run_id = k, t = t, node = BUS[n],
                        demand = d_val[(t, n)],
                        ls = ls_nod[(t, n)],
                        curtailment = cv
                    ))
                end
            end
        end
    else
        rd_vol  = NaN
        rd_cost = NaN
        rd_curt = NaN
        rd_stat = string(termination_status(rd))
    end
 
    # summyrize results
    push!(results, vcat(
        [k, wtype, string(termination_status(min)),
         solve_time(min), relative_gap(min),
         total_cap, mc, le_avg, lc_val, ls_total,
         rd_stat, rd_vol, rd_cost, rd_curt],
        [c_vals[i] for i in I]))
 
    println("cap=$(round(total_cap, digits=0)) ",
            "λc=$(round(lc_val, digits=1)) ",
            "ls=$(round(ls_total, digits=1)) ",
            "rd_vol=$(round(rd_vol, digits=1)) ",
            "rd_cost=$(round(rd_cost, digits=1)) ",
            "curt=$(round(rd_curt, digits=1))")
end
 
# save results
 
run_name = "seq_nl34_2205-WrongRedispatch"
run_dir  = joinpath("results", run_name)
mkpath(run_dir)
 
CSV.write(joinpath(run_dir, "summary.csv"), results)
CSV.write(joinpath(run_dir, "prices.csv"), all_prices)
CSV.write(joinpath(run_dir, "dispatch.csv"), all_dispatch)
CSV.write(joinpath(run_dir, "investment.csv"), all_invest)
CSV.write(joinpath(run_dir, "legacy.csv"), all_legacy)
CSV.write(joinpath(run_dir, "line_flows.csv"), all_flows)
CSV.write(joinpath(run_dir, "redispatch_detail.csv"), all_rd_detail)
CSV.write(joinpath(run_dir, "curtailment.csv"), all_curt)
CSV.write(joinpath(run_dir, "capacity_market.csv"), all_cm)
 
# summary
 
valid = filter(r -> r.rd_status == "OPTIMAL", results)
 
println("\n" * "=" ^ 60)
println("SUMMARY ($(nrow(valid)) / $(nrow(results)) successful)")
println("=" ^ 60)
if nrow(valid) > 0
    println("  Total cap:    $(round(minimum(valid.total_cap), digits=0)) - $(round(maximum(valid.total_cap), digits=0)) MW")
    println("  Market cost:  $(round(minimum(valid.market_cost), digits=0)) - $(round(maximum(valid.market_cost), digits=0))")
    println("  RD volume:    $(round(minimum(valid.rd_volume), digits=1)) - $(round(maximum(valid.rd_volume), digits=1)) MW")
    println("  RD cost:      $(round(minimum(valid.rd_cost), digits=1)) - $(round(maximum(valid.rd_cost), digits=1))")
    println("  Curtailment:  $(round(minimum(valid.curtailment), digits=1)) - $(round(maximum(valid.curtailment), digits=1)) MW")
    println("  Avg λe:       $(round(minimum(valid.lambda_e_avg), digits=2)) - $(round(maximum(valid.lambda_e_avg), digits=2))")
    println("  λc:           $(round(minimum(valid.lambda_c_val), digits=2)) - $(round(maximum(valid.lambda_c_val), digits=2))")
end
