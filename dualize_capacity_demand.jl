using JuMP, Gurobi, Dualization, Printf

# --- Toy data (3 zones) ------------------------------------------------------
Z         = [1, 2, 3]
pair_keys = [(1, 2), (1, 3), (2, 3)]
mec_pairs = Dict((1, 2) => 200.0, (1, 3) => 150.0, (2, 3) => 100.0)
d_c_z     = Dict(1 => 1000.0, 2 => 800.0, 3 => 1200.0)
lambda_c_fix = Dict(1 => 50_000.0, 2 => 30_000.0, 3 => 60_000.0)

# System requirement. Sum of zonal floors (= sum d_c_z, since sum net_import = 0) is 3000.
# Set d_c BELOW 3000 -> system constraint slack -> lambda_c_sys = 0 (matches your real model).
# Set d_c ABOVE 3000 -> system constraint binds  -> lambda_c_sys > 0 (to actually TEST the term).
# Test the binding case so the lambda_c_sys terms are exercised:
d_c = 2200.0

# =============================================================================
# 1. PRIMAL  (with cm_system / lambda_c_sys)
# =============================================================================
function build_primal_capacity(; with_optimizer = true)
    m = with_optimizer ? Model(Gurobi.Optimizer) : Model()
    with_optimizer && set_silent(m)
    @variable(m, c_dem[Z] >= 0)
    @variable(m, f_trans[pair_keys])                       # free (signed)
    @expression(m, net_import[z in Z],
        sum(f_trans[p] for p in pair_keys if p[1] == z; init = 0.0) -
        sum(f_trans[p] for p in pair_keys if p[2] == z; init = 0.0))
    @constraint(m, dem_c_floor[z in Z], c_dem[z] - d_c_z[z] + net_import[z] >= 0)  # nu_c
    @constraint(m, req_nonneg[z in Z],  d_c_z[z] - net_import[z]            >= 0)  # rho
    @constraint(m, f_lo[p in pair_keys], f_trans[p] + mec_pairs[p]          >= 0)  # phi_lo
    @constraint(m, f_hi[p in pair_keys], mec_pairs[p] - f_trans[p]          >= 0)  # phi_hi
    @constraint(m, cm_system,           sum(c_dem[z] for z in Z) - d_c      >= 0)  # lambda_c_sys
    @objective(m, Min, sum(lambda_c_fix[z] * c_dem[z] for z in Z))
    return m, (; c_dem, f_trans, net_import, dem_c_floor, req_nonneg, f_lo, f_hi, cm_system)
end

# =============================================================================
# 2. MANUAL DUAL  (with lambda_c_sys; matches your stat_dem_c & stat_f)
# =============================================================================
function build_manual_dual_capacity()
    d = Model(Gurobi.Optimizer); set_silent(d)
    @variable(d, nu_c[Z]           >= 0)
    @variable(d, rho[Z]            >= 0)
    @variable(d, phi_lo[pair_keys] >= 0)
    @variable(d, phi_hi[pair_keys] >= 0)
    @variable(d, lambda_c_sys      >= 0)
    # dual feasibility
    @constraint(d, df_cdem[z in Z], lambda_c_fix[z] - nu_c[z] - lambda_c_sys >= 0)  # wrt c_dem >= 0
    @constraint(d, df_f[p in pair_keys],                                            # wrt f_trans (free)
        -(nu_c[p[1]] - rho[p[1]]) + (nu_c[p[2]] - rho[p[2]]) + phi_hi[p] - phi_lo[p] == 0)
    # dual objective = sum of (parameter RHS * dual)
    @objective(d, Max,
        sum(d_c_z[z] * nu_c[z] for z in Z) -
        sum(d_c_z[z] * rho[z]  for z in Z) -
        sum(mec_pairs[p] * (phi_lo[p] + phi_hi[p]) for p in pair_keys) +
        d_c * lambda_c_sys)
    return d, (; nu_c, rho, phi_lo, phi_hi, lambda_c_sys)
end

# =============================================================================
# Run the three-way check
# =============================================================================
prim, pc = build_primal_capacity()
optimize!(prim); p_obj = objective_value(prim)

prim2, _ = build_primal_capacity(with_optimizer = false)
autodual = dualize(prim2)
set_optimizer(autodual, Gurobi.Optimizer); set_silent(autodual)
optimize!(autodual); a_obj = objective_value(autodual)

man, mc = build_manual_dual_capacity()
optimize!(man); m_obj = objective_value(man)

println("="^66)
@printf("primal      obj = %16.2f\n", p_obj)
@printf("auto-dual   obj = %16.2f   (Dualization.jl)\n", a_obj)
@printf("manual dual obj = %16.2f\n", m_obj)
println("-"^66)
@printf("lambda_c_sys (manual) = %.2f   | binds? %s\n",
        value(mc.lambda_c_sys), value(mc.lambda_c_sys) > 1e-6 ? "YES" : "no (slack)")
println("-"^66)
@printf("%-4s %10s %12s %12s %10s\n", "zone", "lambda_c", "nu_c", "+lc_sys", "c_dem")
for z in Z
    @printf("%-4d %10.1f %12.1f %12.1f %10.1f\n",
            z, lambda_c_fix[z], value(mc.nu_c[z]), value(mc.lambda_c_sys), value(pc.c_dem[z]))
end
println("-"^66)
println("net_import: ", Dict(z => round(value(pc.net_import[z]), digits = 1) for z in Z))
println("="^66)

@assert isapprox(p_obj, a_obj; rtol = 1e-6) "primal != auto-dual -> primal model issue"
@assert isapprox(p_obj, m_obj; rtol = 1e-6) "manual dual objective mismatch"
# decomposition: lambda_c[z] == nu_c[z] + lambda_c_sys wherever c_dem[z] > 0
for z in Z
    if value(pc.c_dem[z]) > 1e-6
        @assert isapprox(lambda_c_fix[z], value(mc.nu_c[z]) + value(mc.lambda_c_sys); rtol = 1e-5) "decomposition off in zone $z"
    end
end
println("OK: primal == auto-dual == manual dual, and lambda_c[z] == nu_c[z] + lambda_c_sys.")