using CSV
using DataFrames

const SCENARIOS = Dict(

    "nl10" => (
        invest_generators   = "nl10_investable_generators.csv",
        legacy_generators   = "nl10_legacy_generators.csv",
        renewable_gens      = "nl10_renewable_generators.csv",
        total_load          = "nl10_total_load_timeseries.csv",
        load_per_bus        = "nl10_load_timeseries_per_bus.csv",
        ptdf                = "nl10_ptdf_matrix.csv",
        bus_order           = "nl10_bus_order.csv",
        line_order          = "nl10_line_order.csv",
        fixed_om            = "fixed_om_costs.csv",
        renewable_cf        = "nl10_renewable_cf_per_generator.csv",
    ),

    "nl20" => (
        invest_generators   = "nl20_investable_generators.csv",
        legacy_generators   = "nl20_legacy_generators.csv",
        renewable_gens      = "nl20_renewable_generators.csv",
        total_load          = "nl20_total_load_timeseries.csv",
        load_per_bus        = "nl20_load_timeseries_per_bus.csv",
        ptdf                = "ptdf_matrix.csv",
        bus_order           = "bus_order.csv",
        line_order          = "line_order.csv",
        fixed_om            = "fixed_om_costs.csv",
        renewable_cf        = "nl20_renewable_cf_per_generator.csv",
    ),

    "nl34_z3" => (
        invest_generators   = "nl34_investable_generators.csv",
        legacy_generators   = "nl34_legacy_generators.csv",
        renewable_gens      = "nl34_renewable_generators.csv",
        total_load          = "nl34_total_load_timeseries.csv",
        load_per_bus        = "nl34_load_timeseries_per_bus.csv",
        ptdf                = "nl34_ptdf_matrix.csv",
        bus_order           = "nl34_bus_order_z3.csv",
        line_order          = "nl34_line_order.csv",
        fixed_om            = "fixed_om_costs.csv",
        renewable_cf        = "nl34_renewable_cf_per_generator.csv",
        rep_profiles        = "repdays_8-unequal_weights/resulting_profiles.csv",        
        rep_decisions       = "repdays_8-unequal_weights/decision_variables_short.csv",  
        mec_profiles        = "nl34_interzonal_mec_z3.csv",
    ),


    "nl34_z1" => (
        invest_generators   = "nl34_investable_generators.csv",
        legacy_generators   = "nl34_legacy_generators.csv",
        renewable_gens      = "nl34_renewable_generators.csv",
        total_load          = "nl34_total_load_timeseries.csv",
        load_per_bus        = "nl34_load_timeseries_per_bus.csv",
        ptdf                = "nl34_ptdf_matrix.csv",
        bus_order           = "nl34_bus_order_z1.csv",
        line_order          = "nl34_line_order.csv",
        fixed_om            = "fixed_om_costs.csv",
        renewable_cf        = "nl34_renewable_cf_per_generator.csv",
        rep_profiles        = "repdays_8-unequal_weights/resulting_profiles.csv",        
        rep_decisions       = "repdays_8-unequal_weights/decision_variables_short.csv",  
    ),

    "poc_z1" => (
        invest_generators   = "poc_controllable_generators.csv",
        legacy_generators   = "poc_legacy.csv",
        renewable_gens      = "poc_renewable_generators.csv",
        total_load          = "poc_total_load_timeseries.csv",
        load_per_bus        = "poc_load_timeseries_per_bus.csv",
        ptdf                = "poc_ptdf_matrix.csv",
        bus_order           = "poc_bus_order_z1.csv",
        line_order          = "poc_line_order.csv",
        fixed_om            = "fixed_om_costs.csv",
        renewable_cf        = "nl20_renewable_cf_per_generator.csv",
        rep_profiles        = "poc_time/resulting_profiles.csv",        
        rep_decisions       = "poc_time/decision_variables_short.csv",  
    ),

    "poc_z3" => (
        invest_generators   = "poc_controllable_generators.csv",
        legacy_generators   = "poc_legacy.csv",
        renewable_gens      = "poc_renewable_generators.csv",
        total_load          = "poc_total_load_timeseries.csv",
        load_per_bus        = "poc_load_timeseries_per_bus.csv",
        ptdf                = "poc_ptdf_matrix.csv",
        bus_order           = "poc_bus_order_z3.csv",
        line_order          = "poc_line_order.csv",
        fixed_om            = "fixed_om_costs.csv",
        renewable_cf        = "nl20_renewable_cf_per_generator.csv",
        rep_profiles        = "poc_time/resulting_profiles.csv",        
        rep_decisions       = "poc_time/decision_variables_short.csv", 
        mec_profiles        = "poc_interzonal_mec_z3.csv",
    ),
)

function load_data(scenario="nl20"; data_dir="data", overrides...)
    haskey(SCENARIOS, scenario) || error("Unknown scenario \"$scenario\". Known: $(join(keys(SCENARIOS), ", "))")
    cfg = merge(SCENARIOS[scenario], NamedTuple(overrides))
    p(key) = joinpath(data_dir, cfg[key])

    base = (
        invest_gen_df = CSV.read(p(:invest_generators), DataFrame),
        legacy_gen_df = CSV.read(p(:legacy_generators), DataFrame),
        renewable_df  = CSV.read(p(:renewable_gens), DataFrame),
        dem_tot_df    = CSV.read(p(:total_load),     DataFrame),
        dem_df        = CSV.read(p(:load_per_bus),   DataFrame),
        ptdf_df       = CSV.read(p(:ptdf),           DataFrame),
        bus_order_df  = CSV.read(p(:bus_order),      DataFrame),
        line_order_df = CSV.read(p(:line_order),     DataFrame),
        fixed_om_df   = CSV.read(p(:fixed_om),       DataFrame),
        cf_df         = CSV.read(p(:renewable_cf),   DataFrame),
    )

    extras = NamedTuple()
 
    if haskey(cfg, :rep_profiles)
        extras = merge(extras, (
            rep_profiles_df  = CSV.read(p(:rep_profiles),  DataFrame),
            rep_decisions_df = CSV.read(p(:rep_decisions), DataFrame),
        ))
    end
 
    if haskey(cfg, :mec_profiles)
        extras = merge(extras, (
            mec_df = CSV.read(p(:mec_profiles), DataFrame),
        ))
    end
 
    return merge(base, extras)
end

function build_load_shares(data)
    dem_df  = data.dem_df
    dem_tot = data.dem_tot_df
    bus_cols = string.(names(dem_df)[2:end])
    raw = Dict(col => dem_df[1, col] / dem_tot.load[1] for col in bus_cols)
    total = sum(values(raw))
    return Dict(k => v / total for (k, v) in raw)
end

function build_rep_days(data, bus_to_idx)
    rp  = data.rep_profiles_df
    dec = data.rep_decisions_df

    T = 1:nrow(rp)
    period_weight = Dict(row.periods => row.weights for row in eachrow(dec))
    w = Dict(t => period_weight[rp.period[t]] for t in T)
    load = Dict(t => Float64(rp.Load[t]) for t in T)

    cf_cols = [c for c in names(rp) if startswith(string(c), "CF_")]
    function cf_to_gen(col)
        raw = replace(string(col), "CF_" => "")
        raw = replace(raw, "offwind_ac" => "offwind-ac")
        return replace(raw, "_" => " ")
    end

    a = Dict{Tuple{Int,String}, Float64}()
    for t in T, col in cf_cols
        a[(t, cf_to_gen(col))] = Float64(rp[t, col])
    end

    share = build_load_shares(data)
    demand = Dict{Tuple{Int,Int}, Float64}()
    for t in T, (bus_name, s) in share
        haskey(bus_to_idx, bus_name) && (demand[(t, bus_to_idx[bus_name])] = load[t] * s)
    end

    return (T=T, w=w, load=load, a=a, demand=demand)
end

function build_generators(data, rep, bus_to_idx;
    renewable_carriers = Set(["onwind", "offwind-ac", "solar"])
)
    invest_gen_df = data.invest_gen_df
    legacy_gen_df = data.legacy_gen_df
    fixed_om_df   = data.fixed_om_df

    I = Vector{String}(invest_gen_df.name)
    J = Vector{String}(legacy_gen_df.name)

    function gen_dicts(df)
        (
            vc   = Dict(row.name => row.marginal_cost for row in eachrow(df)),
            cap  = Dict(row.name => row.p_nom         for row in eachrow(df)),
            tech = Dict(row.name => row.carrier        for row in eachrow(df)),
            gens = Dict(row.name => row.bus            for row in eachrow(df)),
        )
    end

    gi = gen_dicts(invest_gen_df)
    gj = gen_dicts(legacy_gen_df)

    vc   = merge(gi.vc,   gj.vc)
    cap  = merge(gi.cap,  gj.cap)
    tech = merge(gi.tech, gj.tech)
    gens = merge(gi.gens, gj.gens)

    IJ          = vcat(I, J)
    gen_bus_idx = Dict(g => bus_to_idx[gens[g]] for g in IJ)

    total_hours = sum(rep.w[t] for t in rep.T)
    fc = Dict(row.tech => row.fixed_om_eur_MW_hour * total_hours for row in eachrow(fixed_om_df))
    for t in unique(values(gi.tech))
        haskey(fc, t) || (fc[t] = 0.0)
    end

    a = Dict{Tuple{Int,String},Float64}()
    for t in rep.T, g in IJ
        a[(t, g)] = get(rep.a, (t, g), 1.0)
    end

    alpha = Dict(g => tech[g] in renewable_carriers ? 0.0 : 1.0 for g in IJ)

    return (I=I, J=J, vc=vc, cap=cap, tech=tech, gens=gens, gen_bus_idx=gen_bus_idx, fc=fc, a=a, alpha=alpha)
end

function build_network(data)
    BUS  = string.(data.bus_order_df.name)
    LINE = string.(data.line_order_df.name)

    bus_to_idx  = Dict(BUS[i]  => i for i in eachindex(BUS))
    line_to_idx = Dict(LINE[i] => i for i in eachindex(LINE))

    ptdf_bus_names  = string.(names(data.ptdf_df)[2:end])
    ptdf_line_names = string.(data.ptdf_df[:, 1])
    @assert ptdf_line_names == LINE "Line order in PTDF does not match bus_order file"
    @assert ptdf_bus_names  == BUS  "Bus order in PTDF does not match bus_order file"

    PTDF = Matrix{Float64}(data.ptdf_df[:, 2:end])
    Fmax = Vector{Float64}(data.line_order_df.s_nom)

    return (
        BUS         = BUS,
        LINE        = LINE,
        N           = 1:length(BUS),
        L           = 1:length(LINE),
        bus_to_idx  = bus_to_idx,
        line_to_idx = line_to_idx,
        PTDF        = PTDF,
        Fmax        = Fmax,
    )
end

function build_zones(data, net, gen_data; sigma::Float64 = 1.0)
    bus_order_df = data.bus_order_df
 
    bus_to_zone = Dict{Int,Int}()
    for row in eachrow(bus_order_df)
        bus_name = string(row.name)
        haskey(net.bus_to_idx, bus_name) || continue
        bus_to_zone[net.bus_to_idx[bus_name]] = Int(row.zone)
    end
 
    Z = sort(unique(values(bus_to_zone)))
 
    buses_in_zone = Dict{Int,Vector{Int}}(z => Int[] for z in Z)
    for (n, z) in bus_to_zone
        push!(buses_in_zone[z], n)
    end
    for z in Z
        sort!(buses_in_zone[z])
    end
 
    gen_to_zone = Dict{String,Int}()
    IJ = vcat(gen_data.I, gen_data.J)
    for g in IJ
        gen_to_zone[g] = bus_to_zone[gen_data.gen_bus_idx[g]]
    end
 
    gens_in_zone = Dict{Int,Vector{String}}(z => String[] for z in Z)
    for (g, z) in gen_to_zone
        push!(gens_in_zone[z], g)
    end
    for z in Z
        sort!(gens_in_zone[z])
    end
 
    MEC = Dict{Tuple{Int,Int}, Float64}()
    if hasproperty(data, :mec_df)
        for row in eachrow(data.mec_df)
            z1, z2 = Int(row.zone_1), Int(row.zone_2)
            mec_val = sigma * Float64(row.mec_MW)
            MEC[(z1, z2)] = mec_val
            MEC[(z2, z1)] = mec_val   # symmetric
        end
    else
        line_df = data.line_order_df
        for row in eachrow(line_df)
            b0 = string(row.bus0)
            b1 = string(row.bus1)
            haskey(net.bus_to_idx, b0) && haskey(net.bus_to_idx, b1) || continue
            z0 = bus_to_zone[net.bus_to_idx[b0]]
            z1 = bus_to_zone[net.bus_to_idx[b1]]
            z0 == z1 && continue
            pair_fwd = (min(z0,z1), max(z0,z1))
            MEC[pair_fwd] = get(MEC, pair_fwd, 0.0) + Float64(row.s_nom)
        end
        for ((z1,z2), val) in collect(MEC)
            MEC[(z1,z2)] = sigma * val
            MEC[(z2,z1)] = sigma * val
        end
    end
 
    return (
        Z             = Z,
        bus_to_zone   = bus_to_zone,
        gen_to_zone   = gen_to_zone,
        buses_in_zone = buses_in_zone,
        gens_in_zone  = gens_in_zone,
        MEC           = MEC,
    )
end