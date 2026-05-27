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
import HSL_jll

# Load file with model-building functions and utilities for collecting
# inputs and outputs
include("moai-canos-model.jl")

# Python imports
PythonCall.pyimport("sys").path.append(pwd())
VC = PythonCall.pyimport("vectorcanos")
torch = PythonCall.pyimport("torch")

# Load CANOS NN model
#modelpath = joinpath("runs", "canos_task_1_1", "canos_k_steps15_hd128_lr5e-4_task_1_1_260116_121912", "model.pt")
modelpath = "vectorcanos-trained.pt"
nn = torch.load(modelpath, map_location="cpu")
predictor = MOAI.PytorchModel(modelpath)

# Load dataset that we will use for target data
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

function print_x_with_bounds(x::AbstractVector, input_bounds::Vector{Tuple{Float64,Float64}}, input_names::Vector{String})
    @assert length(x) == length(input_bounds) == length(input_names)
    println(@sprintf("%4s %20s %14s %14s %14s %10s", "idx", "name", "value", "lb", "ub", "viol"))
    for i in eachindex(x)
        val = x[i]
        lb, ub = input_bounds[i]
        viol = max(0.0, val - ub, lb - val)
        println(@sprintf("%4d %20s %14.6f %14.6f %14.6f %10.3f", i, input_names[i], val, lb, ub, viol))
    end
end

py_x0 = dataset[0]
py_x0_flat = nn.flatten_input(py_x0)
x0 = PythonCall.pyconvert(Vector{Float64}, py_x0_flat)
pglib_data = PGLib.pglib("case14")
load_pfd_into_pm!(pglib_data, py_x0)
pm = PowerModels.instantiate_model(pglib_data, PowerModels.ACPPowerModel, PowerModels.build_opf)

# 1. Add any variables that are necessary
# 2. Load point from dataset, map variables to values from this point, construct objective.
# 3. Deactivate bounds on ACPF outputs.
# 4. Add bound constraining _some_ output to be above its limit

inputs, input_bounds = get_inputs(pm)
outputs, output_names = get_outputs(pm)
input_names = get_input_names(pm)
n_inputs = length(inputs)
n_outputs = length(outputs)

# Make sure our target variable does not violate any bounds
input_lbs = first.(input_bounds)
input_ubs = last.(input_bounds)
@assert all(input_lbs .- 1e-5 .<= x0 .<= input_ubs .+ 1e-5)
print_x_with_bounds(x0, input_bounds, input_names)

# Delete bounds and inequalities from the original model
for var in JuMP.all_variables(pm.model)
    if JuMP.has_lower_bound(var)
        JuMP.delete_lower_bound(var)
    end
    if JuMP.has_upper_bound(var)
        JuMP.delete_upper_bound(var)
    end
end
for con in MPIN.get_inequality_constraints(pm.model)
    JuMP.delete(pm.model, con)
end

nonconst_mask = .!isa.(inputs, Number)
JuMP.@constraint(pm.model, input_lbs[nonconst_mask] .<= inputs[nonconst_mask] .<= input_ubs[nonconst_mask])
#print_x_with_bounds(x0, input_bounds, input_names)

# Minimize 1-norm of difference between inputs and our target inputs.
JuMP.@variable(pm.model, input_slack_pos[1:n_inputs] >= 0.0, start = 0.0)
JuMP.@variable(pm.model, input_slack_neg[1:n_inputs] >= 0.0, start = 0.0)
JuMP.@constraint(pm.model, input_slack_eqn,
    inputs .- x0 .+ input_slack_pos .- input_slack_neg .== 0.0
)
JuMP.@objective(pm.model, Min, sum(input_slack_pos .+ input_slack_neg))

# CANOS constraints
# We add these extra variables as a hacky workaround to make all inputs variables.
@variable(pm.model, moai_inputs[i = 1:n_inputs], start = x0[i])
@constraint(pm.model, moai_input_link, inputs .== moai_inputs)
cuda_available = PythonCall.pyconvert(Bool, torch.cuda.is_available())
device = cuda_available ? "cuda" : "cpu"
println("device = $device")
y, _ = MOAI.add_predictor(pm.model, predictor, moai_inputs; gray_box = true, device)
pm_to_canos = Dict(zip(outputs, y))

# Constraints imposing a voltage mismatch on some bus
vm_pm = PowerModels.var(pm, :vm, 12)
vm_canos = pm_to_canos[vm_pm]
@constraint(pm.model, vm_pm <= 0.90)
@constraint(pm.model, vm_canos >= 0.94)

JuMP.set_optimizer(pm.model, Ipopt.Optimizer)
JuMP.set_optimizer_attributes(pm.model, "linear_solver" => "ma27")
JuMP.optimize!(pm.model)

println()
println("Compare deviations from initial input x0")
println("----------------------------------------")
println(@sprintf(
    "%4s %10s %14s %14s %14s %14s %14s %3s",
    "idx", "name", "value", "target", "error", "lb", "ub", "type",
))
solved_to_x0 = true
for i in 1:n_inputs
    inp = inputs[i]
    val = isa(inp, Number) ? inp : JuMP.value(inp)
    target = x0[i]
    err = abs(val - target)

    if err <= 1e-8
        continue
    else
        # We have some error
        global solved_to_x0 = false
    end
    if isa(inp, JuMP.VariableRef)
        lb = JuMP.has_lower_bound(inp) ? JuMP.lower_bound(inp) : -Inf
        ub = JuMP.has_upper_bound(inp) ? JuMP.upper_bound(inp) : Inf
        println(
            @sprintf(
                "%4d %10s %14.6f %14.6f %14.6f %14.6f %14.6f %3s",
                i, input_names[i], val, target, err, lb, ub, "var",
            )
        )
    else
        kind = isa(inp, Number) ? "const" : "expr"
        println(
            @sprintf(
                "%4d %10s %14.6f %14.6f %14.6f %14s %14s %3s",
                i, input_names[i], val, target, err, "-", "-", kind,
            )
        )
    end
end

x1 = JuMP.value.(inputs)
py_x1 = torch.tensor(x1)
py_y0 = nn(py_x0_flat).detach()
py_y1 = nn(py_x1).detach()
#diff = torch.abs(py_y1 - py_y0)
y1 = PythonCall.pyconvert(Vector{Float64}, py_y1.numpy())

y_pf = JuMP.value.(outputs)
diff = y_pf .- y1

println()
println("Compare output from PowerModels and CANOS model")
println("-----------------------------------------------")
println(@sprintf("%4s %10s %10s %10s %10s", "idx", "var", "pm_out", "canos_out", "err"))
for i in eachindex(y_pf)
    println(@sprintf("%4d %10s %10.3f %10.3f %10.3f", i, outputs[i], y_pf[i], y1[i], diff[i]))
end

# NOTE That this only makes sense if we solved to a training point x0
if solved_to_x0
    println()
    println("Solved to a training point x0. Labels are available")
    py_y_target = VC.flatten_input_labels(py_x0)
    y_target = PythonCall.pyconvert(Vector{Float64}, py_y_target)
    diff = y_pf .- y_target
    println("Compare output from PowerModels and CANOS targets")
    println("-------------------------------------------------")
    println(@sprintf("%4s %10s %10s %13s %10s", "idx", "var", "pm_out", "canos_target", "err"))
    for i in eachindex(y_pf)
        println(@sprintf("%4d %10s %10.3f %10.3f %10.3f", i, outputs[i], y_pf[i], y_target[i], diff[i]))
    end
end
