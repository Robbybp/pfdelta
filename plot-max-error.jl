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
        :color => PowerPlots.color_schemes[:reds],
    ),
    #gen = :data => "ComponentType",   # keep generators distinct if present
    #load = :data => "ComponentType",  # keep loads distinct if present
    gen = (:color => "#d3d3d3"),
    load = (:color => "#d3d3d3"),
    shunt = (:color => "#d3d3d3"),
)

for l in (4, 5, 6)
    plt.layer[l]["encoding"]["color"]["legend"] = false
    pop!(plt.layer[l]["encoding"]["color"], "title")
end

## Force non-PQ buses to light gray, keep PQ buses on red scale
#bus_colors = PowerPlots.color_schemes[:reds]
#for layer in plt.layer
#    if layer isa Dict &&
#       haskey(layer, "encoding") &&
#       layer["encoding"] isa Dict &&
#       haskey(layer["encoding"], "color") &&
#       layer["encoding"]["color"] isa Dict &&
#       get(layer["encoding"]["color"], "title", "") == "Bus"
#        layer["encoding"]["color"] = Dict(
#            "condition" => Dict(
#                "test" => "datum.is_pq",
#                "field" => "diff",
#                "type" => "quantitative",
#                "title" => "Bus",
#                "scale" => Dict("range" => bus_colors),
#            ),
#            "value" => "#d3d3d3",
#        )
#    end
#end
#
## Color generators/loads/shunts light gray and hide their legends
#for layer in plt.layer
#    if layer isa Dict && haskey(layer, "encoding") && haskey(layer["encoding"], "color")
#        color_enc = layer["encoding"]["color"]
#        title = get(color_enc, "title", "")
#        if title in ("Gen", "Load", "Shunt", "Storage")
#            color_enc["legend"] = false
#            color_enc["value"] = "#d3d3d3"
#            pop!(color_enc, "field", nothing)
#            pop!(color_enc, "type", nothing)
#            pop!(color_enc, "scale", nothing)
#        end
#    end
#end

#display(plt)
VegaLite.save("max-error-pq.pdf", plt)
