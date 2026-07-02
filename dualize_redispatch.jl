using JuMP, Gurobi, Dualization, Printf

N    = [1, 2, 3]
L    = [1, 2, 3]                                  # line 1: 1-2, line 2: 2-3, line 3: 1-3
PTDF = Dict((1,1)=>0.0, (1,2)=> 0.667, (1,3)=>0.333,     # line 1-2
            (2,1)=>0.0, (2,2)=>-0.333, (2,3)=>0.333,     # line 2-3
            (3,1)=>0.0, (3,2)=> 0.333, (3,3)=>0.667)     # line 1-3
Fmax = Dict(1 => 50.0, 2 => 50.0, 3 => 30.0)    

G       = ["g1", "g2", "g3"]
gen_bus = Dict("g1"=>1, "g2"=>2, "g3"=>3)
vc      = Dict("g1"=>10.0, "g2"=>30.0, "g3"=>50.0)
q       = Dict("g1"=>120.0, "g2"=>30.0, "g3"=>0.0)     
q_max   = Dict("g1"=>150.0, "g2"=>100.0, "g3"=>100.0) 
d       = Dict(1 => 50.0, 2 => 50.0, 3 => 50.0)        

h   = 150.0    
PEN = 300.0    

gen_at(n) = sum(q[g] for g in G if gen_bus[g] == n; init = 0.0)
# market flow on each line (constant): becomes PTDF*q*phi (bilinear) once coupled
f_mkt = Dict(l => sum(PTDF[(l,n)] * (gen_at(n) - d[n]) for n in N) for l in L)

# =============================================================================
# 1. PRIMAL  
# =============================================================================
function build_primal_rd(; with_optimizer = true)
    m = with_optimizer ? Model(Gurobi.Optimizer) : Model()
    with_optimizer && set_silent(m)
    @variable(m, r_up[G]   >= 0)
    @variable(m, r_down[G] >= 0)
    @variable(m, curt[N]   >= 0)
    @variable(m, p_flow[L])
    @expression(m, p_inj[n in N],
        sum(q[g] + r_up[g] - r_down[g] for g in G if gen_bus[g] == n; init = 0.0) - d[n] + curt[n])
    @constraint(m, c_rup[g in G],  r_up[g]   - (q_max[g] - q[g]) <= 0)                      # beta_pos
    @constraint(m, c_rdn[g in G],  r_down[g] - q[g]              <= 0)                      # beta_neg
    @constraint(m, c_curt[n in N], curt[n]   - d[n]             <= 0)                      # beta_curt
    @constraint(m, c_bal, sum(r_up[g] - r_down[g] for g in G) + sum(curt[n] for n in N) == 0)  # lambda_rd
    @constraint(m, c_fup[l in L],  p_flow[l] - Fmax[l] <= 0)                                # rho_up
    @constraint(m, c_flo[l in L], -p_flow[l] - Fmax[l] <= 0)                                # rho_low
    @constraint(m, c_flow[l in L], p_flow[l] - sum(PTDF[(l,n)] * p_inj[n] for n in N) == 0) # phi
    @objective(m, Min,
        sum((vc[g] + h) * r_up[g] + (-vc[g] + h) * r_down[g] for g in G) +
        PEN * sum(curt[n] for n in N))
    return m, (; r_up, r_down, curt, p_flow)
end

# =============================================================================
# 2. MANUAL DUAL  
# =============================================================================
function build_manual_dual_rd()
    dm = Model(Gurobi.Optimizer); set_silent(dm)
    @variable(dm, beta_pos[G]  >= 0)
    @variable(dm, beta_neg[G]  >= 0)
    @variable(dm, beta_curt[N] >= 0)
    @variable(dm, lambda_rd)                 # free
    @variable(dm, rho_up[L]    >= 0)
    @variable(dm, rho_low[L]   >= 0)
    @variable(dm, phi[L])                    # free
    # dual feasibility (stationarity)
    @constraint(dm, df_rup[g in G],
        (vc[g] + h)  + beta_pos[g]  + lambda_rd - sum(PTDF[(l,gen_bus[g])] * phi[l] for l in L) >= 0)
    @constraint(dm, df_rdn[g in G],
        (-vc[g] + h) + beta_neg[g]  - lambda_rd + sum(PTDF[(l,gen_bus[g])] * phi[l] for l in L) >= 0)
    @constraint(dm, df_curt[n in N],
        PEN          + beta_curt[n] + lambda_rd - sum(PTDF[(l,n)] * phi[l] for l in L)          >= 0)
    @constraint(dm, df_pflow[l in L], phi[l] + rho_up[l] - rho_low[l] == 0)
    # dual objective = inf_x L  (constant part of the Lagrangian)
    @objective(dm, Max,
        - sum((q_max[g] - q[g]) * beta_pos[g] for g in G)
        - sum(q[g] * beta_neg[g]              for g in G)
        - sum(d[n] * beta_curt[n]             for n in N)
        - sum(Fmax[l] * rho_up[l]             for l in L)
        - sum(Fmax[l] * rho_low[l]            for l in L)
        - sum(f_mkt[l] * phi[l]               for l in L))
    return dm, (; beta_pos, beta_neg, beta_curt, lambda_rd, rho_up, rho_low, phi)
end

# =============================================================================
# Three-way check
# =============================================================================
prim, pc = build_primal_rd(); optimize!(prim); p = objective_value(prim)

prim2, _ = build_primal_rd(with_optimizer = false)
ad = dualize(prim2); set_optimizer(ad, Gurobi.Optimizer); set_silent(ad); optimize!(ad)
a = objective_value(ad)

man, mc = build_manual_dual_rd(); optimize!(man); m = objective_value(man)

println("="^62)
@printf("primal      obj = %14.4f\n", p)
@printf("auto-dual   obj = %14.4f   (Dualization.jl)\n", a)
@printf("manual dual obj = %14.4f\n", m)
println("-"^62)
println("r_up   : ", Dict(g => round(value(pc.r_up[g]),   digits=2) for g in G))
println("r_down : ", Dict(g => round(value(pc.r_down[g]), digits=2) for g in G))
println("flows  : ", Dict(l => round(value(pc.p_flow[l]), digits=2) for l in L), "  (Fmax ", Fmax, ")")
println("phi    : ", Dict(l => round(value(mc.phi[l]),    digits=3) for l in L))
println("  internal check phi == rho_low - rho_up: ",
        Dict(l => round(value(mc.rho_low[l]) - value(mc.rho_up[l]), digits=3) for l in L))
println("="^62)

@assert isapprox(p, a; rtol = 1e-6) "primal != auto-dual -> primal model issue"
@assert isapprox(p, m; rtol = 1e-6) "manual dual objective mismatch -> sign/structure bug"
println("OK: primal == auto-dual == manual dual.")


