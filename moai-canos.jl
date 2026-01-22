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
import MathProgIncidence

# Python imports
PythonCall.pyimport("sys").path.append(pwd())
PythonCall.pyimport("vectorcanos")
torch = PythonCall.pyimport("torch")

# Load CANOS NN model
#modelpath = joinpath("runs", "canos_task_1_1", "canos_k_steps15_hd128_lr5e-4_task_1_1_260116_121912", "model.pt")
modelpath = "vectorcanos-trained.pt"
nn = torch.load(modelpath, map_location = "cpu")
predictor = MOAI.PytorchModel(modelpath)

# Load dataset that we will use for target data
pfdelta_variants = PythonCall.pyimport("core.datasets.pfdelta_variants")
PFDeltaCANOS = pfdelta_variants.PFDeltaCANOS
root_dir = joinpath("data", "pfdelta_data")
dataset = PFDeltaCANOS(
    add_bus_type = true,
    case_name = "case14",
    model = "CANOS",
    root_dir = root_dir,
    split = "train",
    task = "1.1",
)

# Map variables to _input_ data
function get_inputs(pm::PowerModels.AbstractPowerModel)
    ref = pm.ref[:it][:pm][:nw][0]
    buskeys = sort(collect(keys(pm.data["bus"])); by = k -> parse(Int, k))
    branchkeys = sort(collect(keys(pm.data["branch"])); by = k -> parse(Int, k))
    # Inputs could be numbers, variables, or expressions
    bus_inputs = Any[]
    pq_inputs = Any[]
    pv_inputs = Any[]
    slack_inputs = Any[]
    branch_inputs = Any[]
    # Bus inputs
    # bus_type == 3 => reference bus
    # bus_type == 2 => generator (PV) bus
    # bus_type == 1 => load (PQ) bus
    for i in buskeys
        idx = parse(Int, i)
        bus_type = pm.data["bus"][i]["bus_type"]
        if bus_type == 1
            pd = sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init = 0.0)
            qd = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][idx]; init = 0.0)
            append!(pq_inputs, [pd, qd])
            append!(bus_inputs, [pd, qd])
        elseif bus_type == 2
            # Since the input calls for total injection/demand, I sum up generator
            # active power variables at each node.
            pg = sum(PowerModels.var(pm, :pg, g) for g in ref[:bus_gens][idx])
            vm = PowerModels.var(pm, :vm, idx)
            append!(pv_inputs, [pg, vm])
            append!(bus_inputs, [pg, vm])
        elseif bus_type == 3
            va = PowerModels.var(pm, :va, idx)
            vm = PowerModels.var(pm, :vm, idx)
            append!(slack_inputs, [va, vm])
            append!(bus_inputs, [va, vm])
        else
            error("Unexpected bus type $(bus_type)")
        end
    end
    for i in branchkeys
        r = pm.data["branch"][i]["br_r"]
        x = pm.data["branch"][i]["br_x"]
        gf = pm.data["branch"][i]["g_fr"]
        bf = pm.data["branch"][i]["b_fr"]
        gt = pm.data["branch"][i]["g_to"]
        bt = pm.data["branch"][i]["b_to"]
        tap = pm.data["branch"][i]["tap"]
        shift = pm.data["branch"][i]["shift"]
        append!(branch_inputs, [r, x, gf, bf, gt, bt, tap, shift])
    end
    inputs = vcat(bus_inputs, pq_inputs, pv_inputs, slack_inputs, branch_inputs)
    return inputs
end

py_x0 = dataset[0]
py_x0_flat = nn.flatten_input(py_x0)
x0_flat = PythonCall.pyconvert(Vector{Float64}, py_x0_flat)
pglib_data = PGLib.pglib("case14")
load_pfd_into_pm!(pglib_data, py_x0)
pm = PowerModels.instantiate_model(pglib_data, PowerModels.ACPPowerModel, PowerModels.build_pf)

# 1. Add any variables that are necessary
# 2. Load point from dataset, map variables to values from this point, construct objective.
# 3. Deactivate bounds on ACPF outputs.
# 4. Add bound constraining _some_ output to be above its limit

inputs = get_inputs(pm)
n_inputs = length(inputs)

# Minimize 1-norm of difference between inputs and our target inputs.
JuMP.@variable(pm.model, input_slack_pos[1:n_inputs] >= 0.0, start = 0.0)
JuMP.@variable(pm.model, input_slack_neg[1:n_inputs] >= 0.0, start = 0.0)
JuMP.@constraint(pm.model, input_slack_eqn,
    inputs .- x0_flat .+ input_slack_pos .- input_slack_neg .== 0.0
)
JuMP.@objective(pm.model, Min, sum(input_slack_pos .+ input_slack_neg))

JuMP.set_optimizer(pm.model, Ipopt.Optimizer)
JuMP.optimize!(pm.model)

# I don't expect to get zero error here unless I've updated parameters in PM.data
#for i in 1:n_inputs
#    lb = JuMP.has_lower_bound
#    println(
#        @sprintf("%10.2f", inputs[i])
#        * @sprintf("%10.2f", x0_flat[i])
#    )
#end
