using JuMP
using CSV
using DataFrames
using Gurobi
using Dualization

gen_df = CSV.read("data/nl20_controllable_generators_poc.csv", DataFrame) #same capacities at each node
#gen_df = CSV.read("data/nl20_controllable_generators.csv", DataFrame)
renewable_df = CSV.read("data/nl20_renewable_generators.csv", DataFrame)
dem_tot_df = CSV.read("data/nl20_total_load_timeseries_poc.csv", DataFrame)
dem_df = CSV.read("data/nl20_load_timeseries_per_bus_poc.csv", DataFrame)


T = vcat(651)
hours = length(T)
tech = Dict(row.name => row.carrier for row in eachrow(gen_df))
gens = Dict(row.name => row.bus for row in eachrow(gen_df))
I = Vector{String}(gen_df.name)

q_param = Dict((t, i) => 200 for t in T, i in I)
#q_param = Dict((t, i) => 400 for t in T, i in I)

D_param = Dict(t => 1200.0 for t in T)   

#VOLL = 4_000.0      # capped market scenario
VOLL = 20_000.0   # perfect market scenario

m_so = Model(Gurobi.Optimizer)

@variable(m_so, ls[t in T] >= 0)

@constraint(m_so, ls_cap[t in T], ls[t] <= D_param[t])
@constraint(m_so, energy_balance[t in T], sum(q_param[(t, i)] for i in I) + ls[t] == D_param[t])

@objective(m_so, Min, sum(VOLL * ls[t] for t in T))

optimize!(m_so)

dual_model_so = dualize(m_so, dual_names = DualNames("dual_var_", "dual_constr_"))
set_optimizer(dual_model_so, Gurobi.Optimizer)
optimize!(dual_model_so)
print(dual_model_so)

d_so = Model(Gurobi.Optimizer)


@variable(d_so, lambda_e[t in T])            
@variable(d_so, nu_ls_up[t in T] <= 0)         

@constraint(d_so, [t in T], lambda_e[t] + nu_ls_up[t] <= VOLL)

@objective(d_so, Max, sum(sum(q_param[(t,i)] for i in I) * lambda_e[t] for t in T) + sum(nu_ls_up[t] * D_param[t] for t in T)) 
optimize!(d_so)
print(d_so)

