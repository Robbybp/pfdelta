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
for (bus, sense) in sweep_inputs
    point, result = solve_maximum_error(bus, sense)
    # result should contain:
    # - termination status
    # - Solve time/iterations
    # - Objective value
    # - NN output -- this should be from the NN, independent of the optimization variables
    # - PF output
    # - bus type
    result = merge((; bus, sense), result)
    push!(results, result)
    push!(points, point)
    break
end

df = DataFrame(results)
CSV.write("max-error-sweep.csv", df)
# TODO: Save points to a JSON file
# I only plan to use these points if I want to evaluate loss or something
