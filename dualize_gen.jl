using JuMP
using CSV
using DataFrames
using Gurobi
using Dualization

gen_df = CSV.read("data/nl20_controllable_generators_poc.csv", DataFrame)
fixed_om_df = CSV.read("data/fixed_om_costs.csv", DataFrame)
cf_df = CSV.read("data/nl20_renewable_cf_per_generator.csv", DataFrame)

T = vcat(651)
hours = length(T)
vc   = Dict(row.name => row.marginal_cost for row in eachrow(gen_df))
cap = Dict(row.name => row.p_nom for row in eachrow(gen_df))
tech = Dict(row.name => row.carrier for row in eachrow(gen_df))
I = Vector{String}(gen_df.name)

fc = Dict(row.tech => (row.fixed_om_eur_MW_hour * hours) for row in eachrow(fixed_om_df))

a = Dict()
for t in T
    for i in I
        if i in names(cf_df)
            a[(t, i)] = Float64(cf_df[t, i])
        else
            a[(t, i)] = 1.0
        end
    end
end

renewable_carriers = Set(["onwind", "offwind-ac", "solar"])
alpha = Dict(i => in(tech[i], renewable_carriers) ? 0.0 : 1.0 for i in keys(tech))

m_gen = Model(Gurobi.Optimizer)

power = @variable(m_gen, q[t in T, i in I] >= 0)
cm_offer = @variable(m_gen, c_cm[i in I] >= 0)
capacity = @variable(m_gen, c[i in I] >= 0)
gen_max = @expression(m_gen, q_max[t in T, i in I], c[i] * a[(t, i)])
cm_max = @expression(m_gen, cap_max_cm[i in I], alpha[i] * c[i])

gen_up = @constraint(m_gen, q_up[t in T, i in I], q[t, i] - q_max[t, i] <= 0)
cap_cm_up = @constraint(m_gen, c_cm_up[i in I], c_cm[i] - cap_max_cm[i] <= 0)
cap_up = @constraint(m_gen, cap_up[i in I], c[i] <= cap[i])

lambda_e = Dict(t => 5.0 for t in T)
lambda_c = 25.0

strong_dual_gen = @objective(m_gen, Max,
    sum(sum((lambda_e[t] - vc[i]) * q[t, i] for t in T) - fc[tech[i]] * c[i] + lambda_c * c_cm[i] for i in I)
)

optimize!(m_gen)

dual_model_gen = dualize(m_gen; dual_names = DualNames("dual_var_", "dual_constr_"))
set_optimizer(dual_model_gen, Gurobi.Optimizer)
optimize!(dual_model_gen)

cap_param = Dict(i => 200 for i in I)

d_gen = Model(Gurobi.Optimizer)

@variable(d_gen, mu_g_up[t in T, i in I] >= 0)
@variable(d_gen, mu_cm_up[i in I] >= 0)
@variable(d_gen, mu_dis_up[i in I] >= 0)

@constraint(d_gen, dual_feas_q[t in T, i in I], -(lambda_e[t] - vc[i]) + mu_g_up[t, i] >= 0)
@constraint(d_gen, dual_feas_c[i in I], fc[tech[i]] - sum(mu_g_up[t, i] * a[(t, i)] for t in T) + mu_dis_up[i] - mu_cm_up[i] * alpha[i] >= 0)
@constraint(d_gen, dual_feas_ccm[i in I], -lambda_c + mu_cm_up[i] >= 0)

@objective(d_gen, Min, sum(cap_param[i] * mu_dis_up[i] for i in I))

optimize!(d_gen)

println("Dual objective: ", objective_value(d_gen))
println("Dual model objective: ", objective_value(dual_model_gen))
println("Primal objective: ", objective_value(m_gen))