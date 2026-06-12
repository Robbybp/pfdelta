ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using CSV
using DataFrames
using PowerPlots
using PGLib
using VegaLite

function plot_bus_types(results_dir, bustypes, domain; title = "Bus")
    # Load objective differences
    df = CSV.read(joinpath(results_dir, "max-error-sweep.csv"), DataFrame)
    pq_df = filter(row -> row.bustype in bustypes && !ismissing(row.objective), df)

    # Take the worst (max abs) objective per PQ bus
    bus_diff = Dict{Int,Float64}()
    for row in eachrow(pq_df)
        bus = Int(row.bus)
        diff = abs(row.objective)
        bus_diff[bus] = haskey(bus_diff, bus) ? max(bus_diff[bus], diff) : diff
    end

    # Prepare PowerModels case and attach the differences to bus records
    case = PGLib.pglib("case14")
    for (bus_id, bus_data) in case["bus"]
        b = parse(Int, bus_id)
        bus_type = get(bus_data, "bus_type", nothing)
        selected = bus_type in bustypes
        bus_data["selected"] = selected
        bus_data["diff"] = selected ? get(bus_diff, b, 0.0) : 0.0
    end

    # This would presumably remove gen/load/shunt nodes, but it also breaks my
    # code below which hard-codes layer indices.
    case["gen"] = Dict{String,Any}()
    case["load"] = Dict{String,Any}()
    case["shunt"] = Dict{String,Any}()

    # Plot with shades of red proportional to the difference
    plt = powerplot(
        case;
        bus = (
            :data => "diff",
            :data_type => "quantitative",
            :color => reverse(PowerPlots.color_schemes[:reds]), # darker for larger values
        ),
        #gen = :data => "ComponentType",   # keep generators distinct if present
        #load = :data => "ComponentType",  # keep loads distinct if present
        gen = (:color => "#d3d3d3"),
        load = (:color => "#d3d3d3"),
        shunt = (:color => "#d3d3d3"),
    )

    # Force non-PQ buses to gray while keeping PQ buses on reversed red scale
    bus_layer = plt.layer[2]
    #color_title = get(bus_layer["encoding"]["color"], "title", title)
    color_title = title
    bus_layer["encoding"]["color"] = Dict(
        "condition" => Dict(
            "test" => "datum.selected",
            "field" => "diff",
            "type" => "quantitative",
            "title" => color_title,
            "scale" => Dict(
                "range" => reverse(PowerPlots.color_schemes[:reds]),
                "domain" => domain,
                "fontSize" => 20,
            ),
            "legend" => Dict(
                "labelFontSize" => 30,      # Legend label font size
                "titleFontSize" => 40,       # Legend title font size
                "gradientLength" => 400,     # Length of the gradient bar
                "gradientThickness" => 20,   # Thickness/width of the gradient bar
            ),
        ),
        "value" => "#d3d3d3",
    )
    #bus_layer["encoding"]["color"]["legend"] = Dict("title" => title)
    plt.layer[1]["layer"][1]["encoding"]["color"]["legend"] = false
    # Can't figure out how to set node sizees...
    bus_layer["encoding"]["size"] = Dict("value" => 1000)

    # Enlarge legends (font, marker, colorbar) by 2x on visible color legends
    #legend_size_updates = Dict(
    #    "labelFontSize" => 20,
    #    "titleFontSize" => 22,
    #    "symbolSize" => 200,
    #    "gradientLength" => 200,
    #    "gradientThickness" => 32,
    #)
    #function apply_legend_size!(layer)
    #    if haskey(layer, "encoding") && haskey(layer["encoding"], "color") && layer["encoding"]["color"] isa Dict
    #        color_enc = layer["encoding"]["color"]
    #        if get(color_enc, "legend", true) != false
    #            legend_dict = get(color_enc, "legend", Dict{String,Any}())
    #            if !(legend_dict isa Dict)
    #                legend_dict = Dict{String,Any}()
    #            end
    #            for (k, v) in legend_size_updates
    #                legend_dict[k] = v
    #            end
    #            color_enc["legend"] = legend_dict
    #        end
    #    end
    #end

    # Apply legend sizing to top-level layers and nested branch layer
    #for layer in plt.layer
    #    apply_legend_size!(layer)
    #    if haskey(layer, "layer") && layer["layer"] isa Vector
    #        for sublayer in layer["layer"]
    #            apply_legend_size!(sublayer)
    #        end
    #    end
    #end

    #for l in (2, 4, 5, 6)
    #    plt.layer[l]["encoding"]["color"]["legend"] = false
    #    pop!(plt.layer[l]["encoding"]["color"], "title")
    #end
    return plt
end

#if length(ARGS) != 1
#    error("Usage: julia plot-max-error.jl RESULTS_DIR")
#end

results_dir = joinpath("results", "20260527-powerup2026-merge")

plt = plot_bus_types(results_dir, [1], [0.005, 0.08], title="Bus V")
VegaLite.save(joinpath(results_dir, "max-error-pq.pdf"), plt)

plt = plot_bus_types(results_dir, [2,3], [0.0, 4.0], title="Bus Q")
VegaLite.save(joinpath(results_dir, "max-error-pv-slack.pdf"), plt)
