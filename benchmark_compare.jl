using Dates, LinearAlgebra, Printf, JuMP, Statistics
include("model_milp.jl")            

scenario   = "poc_z3"
sigma      = 1.0
time_limit = 3600.0
threads    = 8          

run_dir = "benchmark/bench_$(scenario)_sigma$(Int(sigma*100))_$(Dates.format(now(),"yyyymmdd_HHMMSS"))"
log_dir = joinpath(run_dir, "logs"); mkpath(log_dir)
P = load_params(scenario; sigma = sigma)
 
# ---- machine / solver metadata (one file) ----
function gurobi_version()
    tmp = direct_model(Gurobi.Optimizer(GRB_ENV))
    try MOI.get(tmp, MOI.SolverVersion()) catch; "see top of any .log file" end
end
open(joinpath(run_dir, "machine_info.txt"), "w") do io
    println(io, "timestamp         : ", now())
    println(io, "scenario / sigma  : ", scenario, " / ", sigma)
    println(io, "time_limit (s)    : ", time_limit)
    println(io, "Gurobi Threads set: ", threads)
    println(io, "hostname          : ", gethostname())
    println(io, "julia version     : ", VERSION)
    println(io, "gurobi version    : ", gurobi_version())
    println(io, "logical CPU cores : ", Sys.CPU_THREADS)
    println(io, "CPU model         : ", Sys.cpu_info()[1].model)
    println(io, "total RAM (GB)    : ", round(Sys.total_memory()/2^30, digits=1))
    println(io, "free RAM  (GB)    : ", round(Sys.free_memory()/2^30,  digits=1))
    println(io, "BLAS threads      : ", LinearAlgebra.BLAS.get_num_threads())
    println(io, "JULIA_NUM_THREADS : ", get(ENV, "JULIA_NUM_THREADS", "unset"))
    println(io, "(Gurobi's ACTUAL thread/core use is in each model's .log header)")
end
 
results = DataFrame(method=String[], status=String[], solve_time=Float64[],
                    rd_cost=Float64[], mip_gap=Float64[],
                    n_vars=Int[], n_bin=Int[], n_qconstr=Int[], n_qnz=Int[], n_obs=Int[])
safe_gap(m) = try relative_gap(m) catch; NaN end
gattr(m, s) = try MOI.get(m, Gurobi.ModelAttribute(s)) catch; -1 end
function grab!(name, m, rd_cost, t = solve_time(m))
    push!(results, (name, string(termination_status(m)), t, rd_cost, safe_gap(m),
                    gattr(m,"NumVars"), gattr(m,"NumBinVars"), gattr(m,"NumQConstrs"), gattr(m,"NumQCNZs"), 1))
    @printf("%-18s %-14s t=%7.1fs gap=%6.3f qnz=%d bin=%d\n",
            name, string(termination_status(m)), t, safe_gap(m), gattr(m,"NumQCNZs"), gattr(m,"NumBinVars"))
end

sitings = DataFrame(method=String[], run=Int[], gen=String[], c=Float64[])
add_siting!(method, run, c_dict) =
    for (i, ci) in c_dict; push!(sitings, (method, run, string(i), ci)); end
rd_volumes = DataFrame(method=String[], run=Int[], r_up=Float64[], r_down=Float64[], curt=Float64[])
gen_volumes = DataFrame(method=String[], run=Int[], gen=String[], vc=Float64[], vc_group=Int[],
                        r_up=Float64[], r_down=Float64[])

function vc_groups(P)
    uniq = sort(unique(round.(values(P.vc); digits=6)))
    return Dict(g => findfirst(==(round(P.vc[g]; digits=6)), uniq) for g in P.IJ)
end

function record_gen_volumes!(gen_volumes, label, run, P, H)
    groups = vc_groups(P)
    for g in P.IJ
        rup = sum(value(H.r_up[t,g]) for t in P.T)
        rdn = sum(value(H.r_down[t,g]) for t in P.T)
        push!(gen_volumes, (label, run, g, P.vc[g], groups[g], rup, rdn))
    end
end
sense_checks = DataFrame(method=String[], run=Int[], check=String[], value=Float64[], note=String[])

function run_sense_checks!(sense_checks, label, run, P, H)
    (; T, IJ, I, J, N, Z, vc, hmk, VOLL) = P
    threshold = VOLL - hmk
    push!(sense_checks, (label, run, "voll_minus_markup", threshold,
        "generators with vc < this make r_up strictly cheaper than curt in the markup-priced objective"))

    comp_q_I  = maximum(value(H.q[t,i])*value(H.st_q_I[t,i])      for t in T, i in I; init=0.0)
    comp_q_J  = maximum(value(H.q[t,j])*value(H.st_q_J[t,j])      for t in T, j in J; init=0.0)
    comp_c    = maximum(value(H.c[i])*value(H.st_c[i])            for i in I; init=0.0)
    comp_ccmI = maximum(value(H.c_cm_I[i])*value(H.st_cm_I[i])    for i in I; init=0.0)
    comp_ccmJ = maximum(value(H.c_cm_J[j])*value(H.st_cm_J[j])    for j in J; init=0.0)
    comp_cdem = maximum(value(H.c_dem[z])*value(H.st_dem[z])      for z in Z; init=0.0)
    push!(sense_checks, (label, run, "max_comp_slack_q_I", comp_q_I, "q*stat_q (investable); should be ~0"))
    push!(sense_checks, (label, run, "max_comp_slack_q_J", comp_q_J, "q*stat_q (legacy); should be ~0"))
    push!(sense_checks, (label, run, "max_comp_slack_c", comp_c, "c*stat_c (investment); should be ~0"))
    push!(sense_checks, (label, run, "max_comp_slack_ccm_I", comp_ccmI, "c_cm*stat_cm (investable); should be ~0"))
    push!(sense_checks, (label, run, "max_comp_slack_ccm_J", comp_ccmJ, "c_cm*stat_cm (legacy); should be ~0"))
    push!(sense_checks, (label, run, "max_comp_slack_cdem", comp_cdem, "c_dem*stat_dem (capacity); should be ~0"))


    stat_rup(t,g) = P.W_t[t]*(vc[g]+hmk) + value(H.beta_pos[t,g]) + value(H.lambda_rd[t]) -
                    sum(P.PTDF[l,P.gen_bus_idx[g]]*value(H.phi[t,l]) for l in P.L)
    stat_rdn(t,g) = P.W_t[t]*(-vc[g]+hmk) + value(H.beta_neg[t,g]) - value(H.lambda_rd[t]) +
                    sum(P.PTDF[l,P.gen_bus_idx[g]]*value(H.phi[t,l]) for l in P.L)
    stat_curt(t,n) = P.W_t[t]*VOLL + value(H.beta_curt[t,n]) + value(H.lambda_rd[t]) -
                     sum(P.PTDF[l,n]*value(H.phi[t,l]) for l in P.L)
    comp_rup  = maximum(value(H.r_up[t,g])*stat_rup(t,g)     for t in T, g in IJ)
    comp_rdn  = maximum(value(H.r_down[t,g])*stat_rdn(t,g)   for t in T, g in IJ)
    comp_curt = maximum(value(H.curt[t,n])*stat_curt(t,n)    for t in T, n in N)
    push!(sense_checks, (label, run, "max_comp_slack_rup", comp_rup,
        "r_up * stationarity_rup; should be ~0 if complementarity holds despite no explicit constraint"))
    push!(sense_checks, (label, run, "max_comp_slack_rdn", comp_rdn, "r_down * stationarity_rdn; should be ~0"))
    push!(sense_checks, (label, run, "max_comp_slack_curt", comp_curt, "curt * stationarity_curt; should be ~0"))

    below = [g for g in IJ if vc[g] < threshold]
    push!(sense_checks, (label, run, "n_gens_below_threshold", length(below), join(below, ";")))
    rup_below = sum(value(H.r_up[t,g]) for t in T, g in below; init=0.0)
    rup_total = sum(value(H.r_up[t,g]) for t in T, g in IJ)
    push!(sense_checks, (label, run, "share_rup_from_below_threshold_gens",
        rup_total > 1e-9 ? rup_below/rup_total : NaN, "1.0 = ALL r_up comes from gens cheaper than VOLL-h"))
    sd_lhs = sum(P.W_t[t]*( sum((vc[g]+hmk)*value(H.r_up[t,g]) + (-vc[g]+hmk)*value(H.r_down[t,g]) for g in IJ) +
                            VOLL*sum(value(H.curt[t,n]) for n in N) ) for t in T)
    sd_rhs = sum(P.W_t[t]*( - sum((value(H.q_bar[t,g])-value(H.q[t,g]))*value(H.beta_pos[t,g]) for g in IJ)
                            - sum(value(H.q[t,g])*value(H.beta_neg[t,g]) for g in IJ)
                            - sum(value(H.served[t,n])*value(H.beta_curt[t,n]) for n in N)
                            - sum(P.Fmax[l]*(value(H.rho_up[t,l])+value(H.rho_low[t,l])) for l in P.L)
                            - sum(value(H.f_mkt[t,l])*value(H.phi[t,l]) for l in P.L) ) for t in T)
    push!(sense_checks, (label, run, "markup_priced_redispatch_obj", sd_lhs,
        "(vc+h)*r_up + (-vc+h)*r_down + VOLL*curt, W_t-weighted; compare directly across runs"))
    push!(sense_checks, (label, run, "redispatch_SD_residual", sd_lhs - sd_rhs,
        "should be ~0; confirms the KKT point actually satisfies redispatch strong duality"))
    viol_rup = maximum(value(H.r_up[t,g]) - (value(H.q_bar[t,g])-value(H.q[t,g])) for t in T, g in IJ)
    viol_rdn = maximum(value(H.r_down[t,g]) - value(H.q[t,g]) for t in T, g in IJ)
    push!(sense_checks, (label, run, "max_rup_bound_violation", viol_rup, "should be <= ~0"))
    push!(sense_checks, (label, run, "max_rdn_bound_violation", viol_rdn, "should be <= ~0"))
end
 
# --- Full PD: upper band (max); lower band (min) ---
for (sense, nm) in ((:Max, "FullPD_max"), (:Min, "FullPD_min"))
    println("="^72); println(nm)
    m, H = build_full_pd(P; sense = sense, time_limit = time_limit, threads = threads,
                         tag = lowercase(nm), log_dir = log_dir)
    optimize!(m)
    grab!(nm, m, has_values(m) ? value(H.rd_cost) : NaN)
    if has_values(m)
        println(nm, " c: ", Dict(i => round(value(H.c[i]), digits=2) for i in P.I))
        add_siting!(nm, 1, Dict(i => value(H.c[i]) for i in P.I))
        rup_tot = sum(value(H.r_up[t,g]) for t in P.T, g in P.IJ)
        rdn_tot = sum(value(H.r_down[t,g]) for t in P.T, g in P.IJ)
        curt_tot = sum(value(H.curt[t,n]) for t in P.T, n in P.N)
        push!(rd_volumes, (nm, 1, rup_tot, rdn_tot, curt_tot))
        run_sense_checks!(sense_checks, nm, 1, P, H)
        record_gen_volumes!(gen_volumes, nm, 1, P, H)
    end
end
 
 
# redispatch-cost distribution
function report_rd_distribution!(results, label, costs::Vector{Float64}, t_elapsed)
    n = length(costs)
    if n == 0
        push!(results, (label*"_rd_cost", "0 obs — no solved weights", t_elapsed, NaN, NaN,-1,-1,-1,-1, 0))
        return
    end
    s = sort(costs)
    push!(results, (label*"_rd_cost_min", "—", t_elapsed, minimum(s), NaN,-1,-1,-1,-1, n))
    push!(results, (label*"_rd_cost_max", "—", t_elapsed, maximum(s), NaN,-1,-1,-1,-1, n))
end
 
# --- Sequential PD: stage 1 SWEPT over all weights, stage 2 per successful weight ---
println("="^72); println("Sequential PD  (sweeping all exploration weights, budget = $(time_limit)s)")
seq_s1 = build_seq_stage1_sweep(P; total_time_limit = time_limit, threads = threads,
                                tag = "seqpd_s1", log_dir = log_dir)
seq_rd_costs = Float64[]
t_seq_s2 = 0.0
for row in eachrow(seq_s1)
    ismissing(row.q) && continue
    add_siting!("SeqPD", row.w_idx, row.c)
    rd, rdc = solve_redispatch_lp(P, row.q, row.c, row.ls;
        time_limit = time_limit, threads = threads, tag = "seqpd_redispatch_w$(row.w_idx)", log_dir = log_dir)
    if has_values(rd)
        push!(rd_volumes, ("SeqPD", row.w_idx,
            sum(value(rd[:r_up][t,g]) for t in P.T, g in P.IJ),
            sum(value(rd[:r_down][t,g]) for t in P.T, g in P.IJ),
            sum(value(rd[:curt][t,n]) for t in P.T, n in P.N)))
    end
    push!(seq_rd_costs, rdc); global t_seq_s2 += solve_time(rd)
end
n_seq = nrow(seq_s1); n_seq_solved = count(!ismissing, seq_s1.q)
push!(results, ("SeqPD_stage1_sweep", "$(n_seq_solved)/$(n_seq) solved", sum(seq_s1.solve_time), NaN,
                maximum(seq_s1.mip_gap; init=NaN), -1,-1,-1, maximum(seq_s1.n_qnz; init=-1), n_seq_solved))
report_rd_distribution!(results, "SeqPD", seq_rd_costs, t_seq_s2)
@printf("SeqPD: %d/%d weights solved in %.1fs; rd_cost over %d solved weights: min=%.1f max=%.1f\n",
        n_seq_solved, n_seq, sum(seq_s1.solve_time), length(seq_rd_costs),
        isempty(seq_rd_costs) ? NaN : minimum(seq_rd_costs),
        isempty(seq_rd_costs) ? NaN : maximum(seq_rd_costs))
 
# --- MILP-sequential: stage 1 SWEPT, stage 2 per successful weight ---
println("="^72); println("MILP-sequential  (sweeping all exploration weights, budget = $(time_limit)s)")
milp_s1 = build_milp_stage1_sweep(P; total_time_limit = time_limit, threads = threads,
                                  tag = "milp_s1", log_dir = log_dir)
milp_rd_costs = Float64[]
t_milp_s2 = 0.0
for row in eachrow(milp_s1)
    ismissing(row.q) && continue
    add_siting!("MILP", row.w_idx, row.c)
    rd, rdc = solve_redispatch_lp(P, row.q, row.c, row.ls;
        time_limit = time_limit, threads = threads, tag = "milp_redispatch_w$(row.w_idx)", log_dir = log_dir)
    if has_values(rd)
        push!(rd_volumes, ("MILP", row.w_idx,
            sum(value(rd[:r_up][t,g]) for t in P.T, g in P.IJ),
            sum(value(rd[:r_down][t,g]) for t in P.T, g in P.IJ),
            sum(value(rd[:curt][t,n]) for t in P.T, n in P.N)))
    end
    push!(milp_rd_costs, rdc); global t_milp_s2 += solve_time(rd)
end
n_milp = nrow(milp_s1); n_milp_solved = count(!ismissing, milp_s1.q)
push!(results, ("MILP_stage1_sweep", "$(n_milp_solved)/$(n_milp) solved", sum(milp_s1.solve_time), NaN,
                maximum(milp_s1.mip_gap; init=NaN), -1, maximum(milp_s1.n_bin; init=-1), -1, -1, n_milp_solved))
report_rd_distribution!(results, "MILP", milp_rd_costs, t_milp_s2)
@printf("MILP: %d/%d weights solved in %.1fs; rd_cost over %d solved weights: min=%.1f max=%.1f\n",
        n_milp_solved, n_milp, sum(milp_s1.solve_time), length(milp_rd_costs),
        isempty(milp_rd_costs) ? NaN : minimum(milp_rd_costs),
        isempty(milp_rd_costs) ? NaN : maximum(milp_rd_costs))
 
CSV.write(joinpath(run_dir, "seqpd_sweep.csv"), select(seq_s1, Not([:q,:c,:ls])))
CSV.write(joinpath(run_dir, "milp_sweep.csv"), select(milp_s1, Not([:q,:c,:ls])))
CSV.write(joinpath(run_dir, "sitings.csv"), sitings)
CSV.write(joinpath(run_dir, "rd_volumes.csv"), rd_volumes)
CSV.write(joinpath(run_dir, "sense_checks.csv"), sense_checks)
CSV.write(joinpath(run_dir, "gen_volumes.csv"), gen_volumes)
 
println("="^72)
show(results, allrows = true, allcols = true); println()
CSV.write(joinpath(run_dir, "benchmark_results.csv"), results)
println("\nSaved to $(run_dir)/  (results CSV, machine_info.txt, logs/*.log)")
println("Key contrast = n_qnz: FullPD carries the dense redispatch block (PTDF*q*phi);")
println("SeqPD/MILP stage 1 do not. n_bin shows the Big-M binary count of the MILP.")