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

    "nl34" => (
        invest_generators   = "nl34_investable_generators.csv",
        legacy_generators   = "nl34_legacy_generators.csv",
        renewable_gens      = "nl34_renewable_generators.csv",
        total_load          = "nl34_total_load_timeseries.csv",
        load_per_bus        = "nl34_load_timeseries_per_bus.csv",
        ptdf                = "nl34_ptdf_matrix.csv",
        bus_order           = "nl34_bus_order.csv",
        line_order          = "nl34_line_order.csv",
        fixed_om            = "fixed_om_costs.csv",
        renewable_cf        = "nl34_renewable_cf_per_generator.csv",
    ),

    "poc" => (
        invest_generators   = "nl20_controllable_generators_poc.csv",
        legacy_generators   = "nl20_controllable_generators_poc_empty.csv", # same as investable gens for poc
        renewable_gens = "nl20_renewable_generators.csv",
        total_load     = "nl20_total_load_timeseries_poc.csv",
        load_per_bus   = "nl20_load_timeseries_per_bus_poc.csv",
        ptdf           = "ptdf_matrix_poc.csv",
        bus_order      = "bus_order_poc.csv",
        line_order     = "line_order_poc.csv",
        fixed_om       = "fixed_om_costs.csv",
        renewable_cf   = "nl20_renewable_cf_per_generator.csv",
    ),
)

function load_data(scenario="nl20"; data_dir="data", overrides...)
    haskey(SCENARIOS, scenario) || error("Unknown scenario \"$scenario\". Known: $(join(keys(SCENARIOS), ", "))")
    cfg = merge(SCENARIOS[scenario], NamedTuple(overrides))
    p(key) = joinpath(data_dir, cfg[key])
    return (
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
end

function build_generators(data, T, bus_to_idx;
    renewable_carriers = Set(["onwind", "offwind-ac", "solar"])
)
    invest_gen_df = data.invest_gen_df
    legacy_gen_df = data.legacy_gen_df
    fixed_om_df   = data.fixed_om_df
    cf_df         = data.cf_df

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

    hours = length(T)
    fc = Dict(row.tech => row.fixed_om_eur_MW_hour * hours for row in eachrow(fixed_om_df))
    for t in unique(values(gi.tech))
        haskey(fc, t) || (fc[t] = 0.0)
    end

    a = Dict{Tuple{Int,String},Float64}()
    for t in T, g in IJ
        a[(t, g)] = g in names(cf_df) ? Float64(cf_df[t, g]) : 1.0
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