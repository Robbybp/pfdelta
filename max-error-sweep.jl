import DataFrames: DataFrame
import CSV
import JSON

include("results.jl")
include("moai-canos-model.jl")

# min/max refers to the objective sense of the signed error between
# y_PF and y_NN.
nbus = 14
sweep_inputs = collect(Iterators.product(1:nbus, ("min", "max")))
sweep_inputs = reshape(sweep_inputs, *(size(sweep_inputs)...))

points = Any[]
results = Any[]
for (i, (bus, sense)) in enumerate(sweep_inputs)
    println()
    msg = "Sweep sample $i"
    println(msg)
    println(repeat("=", length(msg)))
    println("Inputs: bus=$bus, sense=$sense")
    point, result = solve_maximum_error(bus, sense)
    # result should contain:
    # - termination status
    # - Solve time/iterations
    # - Objective value
    # - NN output -- this should be from the NN, independent of the optimization variables
    # - PF output
    # - bus type
    println("Sweep sample $i result:")
    display(result)
    result = merge((; bus, sense), result)
    push!(results, result)
    push!(points, point)
end

json_array = [
    Dict("pointtype" => "max-error", "bus" => bus, "sense" => sense, "point" => point)
    for ((bus, sense), point) in zip(sweep_inputs, points)
]
points_path = results_path("max-error-points.json")
open(points_path, "w") do io
    JSON.print(io, json_array, 1)
end
println("Wrote adversarial points to $points_path")

df = DataFrame(results)
sweep_path = results_path("max-error-sweep.csv")
CSV.write(sweep_path, df)
println(df)
println("Wrote table to $sweep_path")
# TODO: Save points to a JSON file
# I only plan to use these points if I want to evaluate loss or something
