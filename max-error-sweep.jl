import DataFrames: DataFrame
import CSV

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

df = DataFrame(results)
CSV.write("max-error-sweep.csv", df)
println(df)
println("Wrote table to max-error-sweep.csv")
# TODO: Save points to a JSON file
# I only plan to use these points if I want to evaluate loss or something
