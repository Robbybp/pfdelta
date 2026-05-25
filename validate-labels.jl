"""
This script validates the labels on input data from the PFΔ case-14
task 1.1 training dataset. It does this by solving power flow problems with
PowerModels and computing errors between these solutions and the labels.
Because it solves an ACPF problem for each of 48k samples, it takes a bit
of time to run.

Here is the output of this script, as of 20260124:
```
Found 1 samples with error:
Sample 37506, ϵ = 0.06880697997390817
```
"""

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
using JuMP
using Ipopt
using LinearAlgebra
using MathOptInterface
using PythonCall
using Printf
import MathOptAI as MOAI
import PowerModels
import PGLib
import MathProgIncidence as MPIN
import PowerPlots

include("moai-canos-model.jl")

# Python imports
PythonCall.pyimport("sys").path.append(pwd())
VC = PythonCall.pyimport("vectorcanos")
torch = PythonCall.pyimport("torch")

pfdelta_variants = PythonCall.pyimport("core.datasets.pfdelta_variants")
PFDeltaCANOS = pfdelta_variants.PFDeltaCANOS
root_dir = joinpath("data", "pfdelta_data")
dataset = PFDeltaCANOS(
    add_bus_type=true,
    case_name="case14",
    model="CANOS",
    root_dir=root_dir,
    split="train",
    task="1.1",
)
dataset_len = PythonCall.pyconvert(Int, PythonCall.pybuiltins.len(dataset))
tol = 1e-5
errors = []
for i in 0:(dataset_len - 1)
    println("Sample $i / $dataset_len")
    point = dataset[i]
    y_pf = solve_powerflow(point)
    py_y_data = VC.flatten_input_labels(point).numpy()
    y_data = PythonCall.pyconvert(Vector{Float64}, py_y_data)
    diff = y_data .- y_pf
    maxdiff = maximum(abs.(diff))
    if maxdiff > tol
        println("Sample $i, ϵ = $maxdiff")
        push!(errors, (i, maxdiff))
    end
end
println("Found $(length(errors)) samples with error:")
for (i, maxdiff) in errors
    println("Sample $i, ϵ = $maxdiff")
end
