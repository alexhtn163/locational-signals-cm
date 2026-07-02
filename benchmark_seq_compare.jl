using Dates, LinearAlgebra, Printf, JuMP, Statistics, DataFrames, CSV
include("model_milp.jl")  


scenario          = "nl34_z3"
sigma             = 1.0
per_run_time_limit = 220.0        
threads           = 8
both_directions   = true          
n_light           = 2            
n_heavy           = 2            
n_mid             = 1             
sig_frac          = 0.05       

run_dir = "benchseq_$(scenario)_sigma$(Int(sigma*100))_$(Dates.format(now(),"yyyymmdd_HHMMSS"))"
log_dir = joinpath(run_dir, "logs"); mkpath(log_dir)
P = load_params(scenario; sigma = sigma)

function line_density_table(P; sig_frac = 0.05)
    lines = collect(P.L)
    gmax  = maximum(abs(P.PTDF[l, n]) for l in P.L, n in P.N)
    thr   = sig_frac * gmax
    df = DataFrame(line = Any[], L1 = Float64[], nsig = Int[], maxabs = Float64[])
    for l in lines
        l1  = sum(abs(P.PTDF[l, n]) for n in P.N)
        ns  = count(abs(P.PTDF[l, n]) > thr for n in P.N)
        mx  = maximum(abs(P.PTDF[l, n]) for n in P.N)
        push!(df, (l, l1, ns, mx))
    end
    sort!(df, :L1)                       # ascending: most radial first
    df.rank_light = 1:nrow(df)
    return df
end

function select_probe_lines(dens; n_light, n_heavy, n_mid)
    nrow(dens) == 0 && error("empty density table")
    light = dens.line[1:min(n_light, nrow(dens))]
    heavy = dens.line[max(1, nrow(dens)-n_heavy+1):end]
    midi  = nrow(dens) ÷ 2
    lo    = max(1, midi - (n_mid ÷ 2)); hi = min(nrow(dens), lo + n_mid - 1)
    mid   = dens.line[lo:hi]
    radial = dens.line[argmin(dens.nsig)]      # the structurally lightest direction
    sel = unique(vcat([radial], collect(light), collect(mid), collect(heavy)))
    return sel
end

make_weight(P, l; positive = true) =
    Dict(i => (positive ? 1.0 : -1.0) * P.PTDF[l, P.gen_bus_idx[i]] for i in P.I)

dens = line_density_table(P; sig_frac = sig_frac)
CSV.write(joinpath(run_dir, "line_density_full.csv"), dens)

sel_lines = select_probe_lines(dens; n_light = n_light, n_heavy = n_heavy, n_mid = n_mid)

probe_weights = Dict[]
probe_meta = DataFrame(w_idx = Int[], line = Any[], dir = String[],
                       L1 = Float64[], nsig = Int[], maxabs = Float64[])
dirs = both_directions ? (("+", true), ("-", false)) : (("+", true),)
let k = 0
    for l in sel_lines
        drow = dens[findfirst(==(l), dens.line), :]
        for (dsym, pos) in dirs
            k += 1
            push!(probe_weights, make_weight(P, l; positive = pos))
            push!(probe_meta, (k, l, dsym, drow.L1, drow.nsig, drow.maxabs))
        end
    end
end
CSV.write(joinpath(run_dir, "probe_lines.csv"), probe_meta)

println("="^78)
println("Probe directions (density-ranked PTDF rows):")
show(probe_meta, allrows = true, allcols = true); println()
println("Most radial probe = line $(dens.line[argmin(dens.nsig)]) ",
        "(nsig=$(minimum(dens.nsig))) -> decisive best-case for the a-fortiori claim.")
println("="^78)

# ----------------------------- machine metadata ------------------------------
open(joinpath(run_dir, "machine_info.txt"), "w") do io
    println(io, "timestamp          : ", now())
    println(io, "purpose            : sequential PD vs sequential MILP across PTDF-row densities")
    println(io, "scenario / sigma   : ", scenario, " / ", sigma)
    println(io, "per_run_time_limit : ", per_run_time_limit, " s  (IDENTICAL for both methods)")
    println(io, "both_directions    : ", both_directions)
    println(io, "n_light/heavy/mid  : ", n_light, "/", n_heavy, "/", n_mid)
    println(io, "Gurobi Threads set : ", threads, "   (NonConvex=2 for SeqPD, as in main bench)")
    println(io, "hostname           : ", gethostname())
    println(io, "julia version      : ", VERSION)
    println(io, "logical CPU cores  : ", Sys.CPU_THREADS)
    println(io, "CPU model          : ", Sys.cpu_info()[1].model)
    println(io, "total RAM (GB)     : ", round(Sys.total_memory()/2^30, digits = 1))
    println(io, "(Gurobi's ACTUAL thread/core use is in each model's .log header)")
end

# =============================================================================
# RUN SEQUENTIAL-PD AND SEQUENTIAL-MILP
# ============================================================================
budget = length(probe_weights) * per_run_time_limit + 600.0   # never truncates

println("\n", "#"^78)
println("# Sequential-PD (non-convex strong duality) over ", length(probe_weights), " probe directions")
println("#"^78)
seqpd = build_seq_stage1_sweep(P; weights = probe_weights,
            total_time_limit = budget, per_run_time_limit = per_run_time_limit,
            threads = threads, tag = "seqcmp_pd", log_dir = log_dir)

println("\n", "#"^78)
println("# Sequential-MILP (disjunctive) over ", length(probe_weights), " probe directions")
println("#"^78)
milp = build_milp_stage1_sweep(P; weights = probe_weights,
            total_time_limit = budget, per_run_time_limit = per_run_time_limit,
            threads = threads, tag = "seqcmp_milp", log_dir = log_dir)

# ----------------------------- normalise + join ------------------------------
hasinc(row) = !ismissing(row.q)                      
seqpd_by = Dict(r.w_idx => r for r in eachrow(seqpd))
milp_by  = Dict(r.w_idx => r for r in eachrow(milp))

cmp = DataFrame(w_idx = Int[], line = Any[], dir = String[], L1 = Float64[], nsig = Int[],
                seqpd_status = String[], seqpd_time = Float64[],
                seqpd_incumbent = Bool[], seqpd_gap = Float64[],
                milp_status = String[], milp_time = Float64[],
                milp_incumbent = Bool[], milp_gap = Float64[])
for m in eachrow(probe_meta)
    sp = get(seqpd_by, m.w_idx, nothing)
    mp = get(milp_by,  m.w_idx, nothing)
    push!(cmp, (m.w_idx, m.line, m.dir, m.L1, m.nsig,
        sp === nothing ? "NA" : sp.status, sp === nothing ? NaN : sp.solve_time,
        sp === nothing ? false : hasinc(sp), sp === nothing ? NaN : sp.mip_gap,
        mp === nothing ? "NA" : mp.status, mp === nothing ? NaN : mp.solve_time,
        mp === nothing ? false : hasinc(mp), mp === nothing ? NaN : mp.mip_gap))
end
sort!(cmp, :L1)
CSV.write(joinpath(run_dir, "seq_compare.csv"), cmp)

# =============================================================================
# ANALYSIS 
# =============================================================================
println("\n", "="^78)
println("HEAD-TO-HEAD  (sorted by PTDF-row density L1, ascending = most radial first)")
show(cmp, allrows = true, allcols = true); println()

# ---- Q1: Sequential-PD ----
n_probe   = nrow(cmp)
n_pd_inc  = count(cmp.seqpd_incumbent)
radial_l  = dens.line[argmin(dens.nsig)]
radial_rows = filter(r -> r.line == radial_l, cmp)
println("\n[Q1] Sequential-PD feasible incumbents: $(n_pd_inc)/$(n_probe) probe directions.")
if n_pd_inc == 0
    println("     -> NO direction reached a feasible incumbent in $(per_run_time_limit)s,")
    println("        INCLUDING the most radial row (line $radial_l, nsig=$(minimum(dens.nsig))).")
    println("        => 'all 2|L| exploration directions are intractable' holds a fortiori:")
    println("           the structurally lightest direction already fails.")
else
    println("     -> WARNING: some directions DID get an incumbent. The intractability is")
    println("        NOT uniform across directions. Inspect which (likely the radial ones):")
    show(filter(r -> r.seqpd_incumbent, cmp), allrows = true, allcols = true); println()
    println("        Discussion must then differentiate dense vs. radial directions.")
end
if nrow(radial_rows) > 0
    println("     Most-radial probe result(s):")
    show(radial_rows, allrows = true, allcols = true); println()
end

# ---- Q2: MILP gap vs density ----
milp_ok = filter(r -> r.milp_incumbent && isfinite(r.milp_gap), cmp)
println("\n[Q2] MILP gap vs PTDF-row density:")
if nrow(milp_ok) >= 3 && length(unique(milp_ok.L1)) >= 2
    ρ = cor(milp_ok.L1, milp_ok.milp_gap)
    @printf("     Pearson cor(L1, milp_gap) = %+.3f over %d MILP runs with a finite gap.\n",
            ρ, nrow(milp_ok))
    println("     (Indicative only at this sample size — read alongside the table, not as a test.)")
    hi = milp_ok[argmax(milp_ok.milp_gap), :]
    lo = milp_ok[argmin(milp_ok.milp_gap), :]
    @printf("     Largest gap:  line %s (L1=%.2f) gap=%.3f\n", string(hi.line), hi.L1, hi.milp_gap)
    @printf("     Smallest gap: line %s (L1=%.2f) gap=%.3f\n", string(lo.line), lo.L1, lo.milp_gap)
    println("     If the largest gaps sit on the highest-L1 rows, the MILP gap distribution")
    println("     is EXPLAINED by direction density rather than being random timeouts.")
    println("     CAUTION: correlation != mechanism. Before claiming density DRIVES the gap,")
    println("     check whether a long-gap run CLOSES with more time (=> timeout artefact) or")
    println("     STAYS open (=> structurally flat objective). One extra long run settles it.")
else
    println("     Too few finite-gap MILP runs to correlate; inspect the table directly.")
end

println("\nSaved to $(run_dir)/")
println("  probe_lines.csv      - which PTDF rows were probed + their density")
println("  line_density_full.csv- density ranking of ALL lines (provenance)")
println("  seq_compare.csv      - head-to-head per direction (the headline table)")
println("  logs/*.log           - per-run Gurobi logs (node counts, root gap, dual-inf)")
println("="^78)
