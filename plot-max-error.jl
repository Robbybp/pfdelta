ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using CSV
using DataFrames
using PowerPlots
using PGLib
using VegaLite

# Load objective differences
df = CSV.read("max-error-sweep.csv", DataFrame)
pq_df = filter(row -> row.bustype == 1 && !ismissing(row.objective), df)

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
    is_pq = bus_type == 1 || bus_type == "1"
    bus_data["is_pq"] = is_pq
    bus_data["diff"] = is_pq ? get(bus_diff, b, 0.0) : 0.0
end

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
bus_layer = plt.layer[3]
color_title = get(bus_layer["encoding"]["color"], "title", "Bus")
bus_layer["encoding"]["color"] = Dict(
    "condition" => Dict(
        "test" => "datum.is_pq",
        "field" => "diff",
        "type" => "quantitative",
        "title" => color_title,
        "scale" => Dict(
            "range" => reverse(PowerPlots.color_schemes[:reds]),
            "domain" => [0.06, 0.08],
        ),
    ),
    "value" => "#d3d3d3",
)

# Enlarge legends (font, marker, colorbar) by 2x on visible color legends
legend_size_updates = Dict(
    "labelFontSize" => 20,
    "titleFontSize" => 22,
    "symbolSize" => 200,
    "gradientLength" => 200,
    "gradientThickness" => 32,
)
function apply_legend_size!(layer)
    if haskey(layer, "encoding") && haskey(layer["encoding"], "color") && layer["encoding"]["color"] isa Dict
        color_enc = layer["encoding"]["color"]
        if get(color_enc, "legend", true) != false
            legend_dict = get(color_enc, "legend", Dict{String,Any}())
            if !(legend_dict isa Dict)
                legend_dict = Dict{String,Any}()
            end
            for (k, v) in legend_size_updates
                legend_dict[k] = v
            end
            color_enc["legend"] = legend_dict
        end
    end
end

# Apply legend sizing to top-level layers and nested branch layer
for layer in plt.layer
    apply_legend_size!(layer)
    if haskey(layer, "layer") && layer["layer"] isa Vector
        for sublayer in layer["layer"]
            apply_legend_size!(sublayer)
        end
    end
end


for l in (2, 4, 5, 6)
    plt.layer[l]["encoding"]["color"]["legend"] = false
    pop!(plt.layer[l]["encoding"]["color"], "title")
end

#display(plt)
VegaLite.save("max-error-pq.pdf", plt)
