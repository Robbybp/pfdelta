import DataFrames: DataFrame
import CSV
import JSON
import PGLib

include("moai-canos-model.jl")

# min/max refers to the direction of bound that we constraint y_PF to violate
nbus = 14
#sweep_inputs = collect(Iterators.product(1:nbus, ("lower", "upper")))
training_points = collect(0:9)
sweep_inputs = collect(Iterators.product(1:nbus, ("lower",), training_points))
sweep_inputs = reshape(sweep_inputs, *(size(sweep_inputs)...))

pm_data = PGLib.pglib("case14")

points = Any[]
results = Any[]
json_array = Any[]
for (i, (bus, direction, training_point_index)) in enumerate(sweep_inputs)
    bustype = pm_data["bus"]["$bus"]["bus_type"]
    if bustype != 1
        # Only solving for PQ buses (finding a mismatch in voltage magnitude)
        # for now.
        continue
    end
    println()
    msg = "Sweep sample $i"
    println(msg)
    println(repeat("=", length(msg)))
    println("Inputs: bus=$bus, direction=$direction, training_point=$training_point_index")
    point, result = solve_constrained_error(bus, direction; training_point_index)
    println("(Recall the inputs: bus=$bus, direction=$direction, training_point=$training_point_index)")
    println("Sweep sample $i result:")
    display(result)
    result = merge((; bus, direction, training_point_index), result)
    push!(results, result)
    push!(points, point)
    push!(json_array, Dict(
        "pointtype" => "con-error",
        "bus" => bus,
        "direction" => direction,
        "training_point_index" => training_point_index,
        "point" => point,
    ))
end

open("con-error-points.json", "w") do io
    JSON.print(io, json_array, 1)
end
println("Wrote adversarial points to max-error-points.json")

df = DataFrame(results)
CSV.write("con-error-sweep.csv", df)
println(df)
println("Wrote table to con-error-sweep.csv")
# TODO: Save points to a JSON file
# I only plan to use these points if I want to evaluate loss or something
