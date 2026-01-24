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

include("powerflow.jl")

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
#for i in 0:(dataset_len - 1)
for i in 0:0
    point = dataset[i]
    y_pf = solve_powerflow(point)
    py_y_data = VC.flatten_input_labels(point).numpy()
    y_data = PythonCall.pyconvert(Vector{Float64}, py_y_data)
    diff = y_data .- y_pf
    maxdiff = maximum(abs.(diff))
    println("Max diff: $maxdiff")
    @assert all(diff .<= 1e-5)
end
