using JuMP, Gurobi, Dualization, Printf

# --- Toy data  --
T        = [1, 2, 3]
W        = Dict(1 => 4000.0, 2 => 2760.0, 3 => 2000.0)   
lambda_e = Dict(1 => 80.0, 2 => 45.0, 3 => 30.0)         
lambda_c = 55_000.0                                       


vc_i, fc_i, cap_i, alpha_i = 40.0, 50_000.0, 600.0, 1.0  
vc_j, cap_j, alpha_j = 35.0, 400.0, 1.0
a_j = Dict(1 => 1.0, 2 => 1.0, 3 => 1.0)

# =============================================================================
# INVESTABLE generator
# =============================================================================
function primal_investable(; with_optimizer = true)
    m = with_optimizer ? Model(Gurobi.Optimizer) : Model(); with_optimizer && set_silent(m)
    @variable(m, q[T] >= 0); @variable(m, c >= 0); @variable(m, c_cm >= 0)
    @constraint(m, gen_cap[t in T], q[t] - c          <= 0)    
    @constraint(m, invest,          c   - cap_i        <= 0)   
    @constraint(m, cm_bud,          c_cm - alpha_i*c   <= 0)    
    @objective(m, Max, sum(W[t]*(lambda_e[t]-vc_i)*q[t] for t in T) - fc_i*c + lambda_c*c_cm)
    return m, (; gen_cap, invest, cm_bud)
end

function dual_investable()
    d = Model(Gurobi.Optimizer); set_silent(d)
    @variable(d, mu_g_up[T] >= 0); @variable(d, mu_cap >= 0); @variable(d, mu_cm_bud >= 0)
    @constraint(d, df_q[t in T], mu_g_up[t] >= W[t]* (lambda_e[t]-vc_i))                 
    @constraint(d, df_c, fc_i - sum(mu_g_up[t] for t in T) + mu_cap - alpha_i*mu_cm_bud >= 0)
    @constraint(d, df_ccm, mu_cm_bud >= lambda_c)
    @objective(d, Min, cap_i*mu_cap)      
    return d, (; mu_cap, mu_g_up, mu_cm_bud)
end

# =============================================================================
# LEGACY generator
# =============================================================================
function primal_legacy(; with_optimizer = true)
    m = with_optimizer ? Model(Gurobi.Optimizer) : Model(); with_optimizer && set_silent(m)
    @variable(m, q[T] >= 0); @variable(m, c_cm >= 0)
    @constraint(m, gen_cap[t in T], q[t] - a_j[t]*cap_j  <= 0)   
    @constraint(m, cm_bud,          c_cm - alpha_j*cap_j <= 0)   
    @objective(m, Max, sum(W[t]*(lambda_e[t]-vc_j)*q[t] for t in T) + lambda_c*c_cm)
    return m, (; gen_cap, cm_bud)
end

function dual_legacy()
    d = Model(Gurobi.Optimizer); set_silent(d)
    @variable(d, mu_g_up[T] >= 0); @variable(d, mu_cm_bud >= 0)
    @constraint(d, df_q[t in T], mu_g_up[t] >= W[t]*(lambda_e[t]-vc_j))                 
    @constraint(d, df_ccm, mu_cm_bud >= lambda_c)
    @objective(d, Min, sum(a_j[t]*cap_j*mu_g_up[t] for t in T) + alpha_j*cap_j*mu_cm_bud)  
    return d, (; mu_g_up, mu_cm_bud)
end

# =============================================================================
# Runner
# =============================================================================
function check(name, build_primal, build_dual)
    prim, pc = build_primal(); optimize!(prim); p = objective_value(prim)
    prim2, _ = build_primal(with_optimizer = false)
    ad = dualize(prim2); set_optimizer(ad, Gurobi.Optimizer); set_silent(ad); optimize!(ad)
    a = objective_value(ad)
    man, mc = build_dual(); optimize!(man); m = objective_value(man)
    println("="^58); println(name)
    @printf("  primal      obj = %16.2f\n", p)
    @printf("  auto-dual   obj = %16.2f   (Dualization.jl)\n", a)
    @printf("  manual dual obj = %16.2f\n", m)
    @assert isapprox(p, a; rtol=1e-6) "$name: primal != auto-dual"
    @assert isapprox(p, m; rtol=1e-6) "$name: manual dual objective mismatch"
    println("  OK  (all three agree)")
    return prim, pc, man, mc
end

pi_, pci, di_, dci = check("INVESTABLE generator", primal_investable, dual_investable)
@printf("  -> dual objective = cap_i * mu_cap = %.1f * %.2f   (only the invest term)\n",
        cap_i, value(dci.mu_cap))

pj_, pcj, dj_, dcj = check("LEGACY generator", primal_legacy, dual_legacy)
avail_term = sum(a_j[t]*cap_j*value(dcj.mu_g_up[t]) for t in T)
cm_term    = alpha_j*cap_j*value(dcj.mu_cm_bud)
@printf("  -> dual objective = avail term (%.1f) + CM term (%.1f)   (both survive)\n",
        avail_term, cm_term)

println("="^58)
println("Contrast: investable dual objective has 1 term (mu_cap); legacy has 2 ",
        "(mu_g_up + mu_cm_bud). That is the parameter-vs-variable effect on cap.")
