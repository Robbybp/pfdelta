ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
using JuMP
using Ipopt
using LinearAlgebra
using MathOptInterface
using PythonCall
import MathOptAI as MOAI
import PowerModels
import PGLib
import MathProgIncidence

# Load the CANOS predictor and its saved template/shapes.
PythonCall.pyimport("sys").path.append(pwd())
PythonCall.pyimport("vectorcanos")
torch = PythonCall.pyimport("torch")

#modelpath = joinpath("runs", "canos_task_1_1", "canos_k_steps15_hd128_lr5e-4_task_1_1_260116_121912", "model.pt")
modelpath = "vectorcanos-trained.pt"
nn = torch.load(modelpath, map_location = "cpu")
predictor = MOAI.PytorchModel(modelpath)

pglib_data = PGLib.pglib("case14")
pm = PowerModels.instantiate_model(pglib_data, PowerModels.ACPPowerModel, PowerModels.build_pf)

PythonCall.pyimport("sys").path.append(pwd())
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
py_x0 = dataset[0]

py_x0_flat = nn.flatten_input(py_x0)
x0_flat = PythonCall.pyconvert(Vector{Float64}, py_x0_flat)
# I need to extract loads out of this input data and set them in the PM data
# before building the model

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
        r = data["branch"][i]["br_r"]
        x = data["branch"][i]["br_x"]
        gf = data["branch"][i]["g_fr"]
        bf = data["branch"][i]["b_fr"]
        gt = data["branch"][i]["g_to"]
        bt = data["branch"][i]["b_to"]
        tap = data["branch"][i]["tap"]
        shift = data["branch"][i]["shift"]
        append!(branch_inputs, [r, x, gf, bf, gt, bt, tap, shift])
    end
    inputs = vcat(bus_inputs, pq_inputs, pv_inputs, slack_inputs, branch_inputs)
    return inputs
end

# 1. Add any variables that are necessary
# 2. Load point from dataset, map variables to values from this point, construct objective.
# 3. Deactivate bounds on ACPF outputs.
# 4. Add bound constraining _some_ output to be above its limit

inputs = get_inputs(pm)
