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

# Python imports
PythonCall.pyimport("sys").path.append(pwd())
PythonCall.pyimport("vectorcanos")
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

"""
Load PFΔ input data into a PowerModels data structure.
Anything that is a constant in PowerModels must be set as in this data dict
before building the JuMP model. This includes:
- Loads
- Line parameters
"""
function load_pfd_into_pm!(pm_data::Dict, py_data::Py)
    # Set loads and slack bus params in pm_data
    load_matrix = PythonCall.pyconvert(Matrix{Float64}, py_data["bus"]["bus_demand"].numpy())
    slack_matrix = PythonCall.pyconvert(Matrix{Float64}, py_data["slack"]["x"].numpy())
    # We need to distribute these loads across all loads at each bus.
    loads_by_bus = Dict(b["index"] => Any[] for b in values(pm_data["bus"]))
    nbus = length(loads_by_bus)
    busindices = map(b -> b["index"], values(pm_data["bus"]))
    # Make sure bus indices are contiguous integers
    @assert all(sort(busindices) .== collect(1:nbus))
    for l in values(pm_data["load"])
        b = l["load_bus"]
        push!(loads_by_bus[b], l["index"])
    end
    for i in 1:length(pm_data["bus"])
        # Set vm and va for reference bus
        # Note that it doesn't matter that we have set any potential loads on the
        # reference bus. In fact, this will make it less confusing if we want to compare
        # net generation with actual generator values.
        if pm_data["bus"]["$i"]["bus_type"] == 3
            pm_data["bus"]["$i"]["va"] = slack_matrix[1, 1]
            pm_data["bus"]["$i"]["vm"] = slack_matrix[1, 2]
        end

        pd = load_matrix[i, 1]
        qd = load_matrix[i, 2]
        if pd == 0.0 && qd == 0.0
            continue
        end
        nd = length(loads_by_bus[i])
        if nd == 0
            error("Zero loads attached to a bus that should have some net load")
        end
        p_per_load = pd / nd
        q_per_load = qd / nd
        for l in loads_by_bus[i]
            pm_data["load"]["$l"]["pd"] = p_per_load
            pm_data["load"]["$l"]["qd"] = q_per_load
        end
    end

    # Set line parameters
    branch_matrix = PythonCall.pyconvert(Matrix{Float64}, py_data["bus", "branch", "bus"]["edge_attr"])
    # This doesn't assume branch keys are contiguous integers
    branchkeys = sort(collect(keys(pm_data["branch"])); by=k -> parse(Int, k))
    for (i, k) in enumerate(branchkeys)
        br = pm_data["branch"][k]
        br["br_r"] = branch_matrix[i, 1]
        br["br_x"] = branch_matrix[i, 2]
        br["g_fr"] = branch_matrix[i, 3]
        br["b_fr"] = branch_matrix[i, 4]
        br["g_to"] = branch_matrix[i, 5]
        br["b_to"] = branch_matrix[i, 6]
        br["tap"] = branch_matrix[i, 7]
        br["shift"] = branch_matrix[i, 8]
    end

    return pm_data
end

"""
Get the vector of inputs as expected by CANOS-PF. These inputs can be
JuMP variables, JuMP expressions, or constants.
"""
function get_inputs(pm::PowerModels.AbstractPowerModel)
    # TODO: This function should also get the names of the inputs
    ref = pm.ref[:it][:pm][:nw][0]
    buskeys = sort(collect(keys(pm.data["bus"])); by=k -> parse(Int, k))
    branchkeys = sort(collect(keys(pm.data["branch"])); by=k -> parse(Int, k))
    # Inputs could be numbers, variables, or expressions
    bus_inputs = Any[]
    pq_inputs = Any[]
    pv_inputs = Any[]
    slack_inputs = Any[]
    branch_inputs = Any[]
    bus_bounds = Tuple{Float64,Float64}[]
    pq_bounds = Tuple{Float64,Float64}[]
    pv_bounds = Tuple{Float64,Float64}[]
    slack_bounds = Tuple{Float64,Float64}[]
    branch_bounds = Tuple{Float64,Float64}[]

    # Bus inputs
    # bus_type == 3 => reference bus
    # bus_type == 2 => generator (PV) bus
    # bus_type == 1 => load (PQ) bus
    for i in buskeys
        idx = parse(Int, i)
        bus_type = pm.data["bus"][i]["bus_type"]
        if bus_type == 1
            # Assume no generators live on a load bus
            @assert length(ref[:bus_gens][idx]) == 0
            pd = sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init=0.0)
            qd = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][idx]; init=0.0)
            append!(pq_inputs, [pd, qd])
            append!(bus_inputs, [pd, qd])
            # Add trivial bounds to avoid having to branch later on...
            append!(pq_bounds, [(pd, pd), (qd, qd)])
            append!(bus_bounds, [(pd, pd), (qd, qd)])
        elseif bus_type == 2
            # Since the input calls for total injection/demand, I sum up generator
            # active power variables at each node.
            # Note that PV buses can have loads as well.
            pg = (
                # We don't use init=0 here because there should always be a generator
                sum(PowerModels.var(pm, :pg, g) for g in ref[:bus_gens][idx])
                - sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init = 0.0)
            )
            pgl = (
                sum(ref[:gen][i]["pmin"] for i in ref[:bus_gens][idx])
                - sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init = 0.0)
            )
            pgu = (
                sum(ref[:gen][i]["pmax"] for i in ref[:bus_gens][idx])
                - sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init = 0.0)
            )
            vm = PowerModels.var(pm, :vm, idx)
            vmin = ref[:bus][idx]["vmin"]
            vmax = ref[:bus][idx]["vmax"]
            append!(pv_inputs, [pg, vm])
            append!(bus_inputs, [pg, vm])
            append!(pv_bounds, [(pgl, pgu), (vmin, vmax)])
            append!(bus_bounds, [(pgl, pgu), (vmin, vmax)])
        elseif bus_type == 3
            va = PowerModels.var(pm, :va, idx)
            vm = PowerModels.var(pm, :vm, idx)
            vmin = ref[:bus][idx]["vmin"]
            vmax = ref[:bus][idx]["vmax"]
            append!(slack_inputs, [va, vm])
            append!(bus_inputs, [va, vm])
            append!(slack_bounds, [(-2pi, 2pi), (vmin, vmax)])
            append!(bus_bounds, [(-2pi, 2pi), (vmin, vmax)])
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
        append!(branch_bounds, [(r,r), (x,x), (gf,gf), (bf,bf), (gt,gt), (bt,bt), (tap,tap), (shift,shift)])
    end
    inputs = vcat(bus_inputs, pq_inputs, pv_inputs, slack_inputs, branch_inputs)
    bounds = vcat(bus_bounds, pq_bounds, pv_bounds, slack_bounds, branch_bounds)
    return inputs, bounds
end

function get_outputs(pm::PowerModels.AbstractPowerModel)
    ref = pm.ref[:it][:pm][:nw][0]
    buskeys = sort(collect(keys(pm.data["bus"])); by=k -> parse(Int, k))
    branchkeys = sort(collect(keys(pm.data["branch"])); by=k -> parse(Int, k))

    bus_out = Any[]
    pq_out = Any[]
    pv_out = Any[]
    slack_out = Any[]
    branch_out = Any[]

    for buskey in buskeys
        idx = parse(Int, buskey)
        bus_type = pm.data["bus"][buskey]["bus_type"]

        va = PowerModels.var(pm, :va, idx)
        vm = PowerModels.var(pm, :vm, idx)
        append!(bus_out, [va, vm])

        if bus_type == 1
            append!(pq_out, [va, vm])
        elseif bus_type == 2
            qg = sum(PowerModels.var(pm, :qg, g) for g in ref[:bus_gens][idx]; init=0.0)
            qd = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][idx]; init=0.0)
            net_q = qg - qd
            append!(pv_out, [va, net_q])
        elseif bus_type == 3
            pg = sum(PowerModels.var(pm, :pg, g) for g in ref[:bus_gens][idx]; init=0.0)
            qg = sum(PowerModels.var(pm, :qg, g) for g in ref[:bus_gens][idx]; init=0.0)
            pd = sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init=0.0)
            qd = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][idx]; init=0.0)
            net_p = pg - pd
            net_q = qg - qd
            append!(slack_out, [net_p, net_q])
        else
            error("Unexpected bus type $(bus_type)")
        end
    end

    for branchkey in branchkeys
        idx = parse(Int, branchkey)
        fbus = ref[:branch][idx]["f_bus"]
        tbus = ref[:branch][idx]["t_bus"]
        append!(
            branch_out,
            [
                PowerModels.var(pm, :p, (idx, fbus, tbus)),
                PowerModels.var(pm, :q, (idx, fbus, tbus)),
                PowerModels.var(pm, :p, (idx, tbus, fbus)),
                PowerModels.var(pm, :q, (idx, tbus, fbus)),
            ],
        )
    end

    return vcat(bus_out, pq_out, pv_out, slack_out, branch_out)
end

function get_input_names(pm::PowerModels.AbstractPowerModel)
    ref = pm.ref[:it][:pm][:nw][0]
    buskeys = sort(collect(keys(pm.data["bus"])); by=k -> parse(Int, k))
    branchkeys = sort(collect(keys(pm.data["branch"])); by=k -> parse(Int, k))

    bus_names = String[]
    pq_names = String[]
    pv_names = String[]
    slack_names = String[]
    branch_names = String[]

    for i in buskeys
        idx = parse(Int, i)
        bus_type = pm.data["bus"][i]["bus_type"]
        if bus_type == 1
            append!(bus_names, ["bus_pd[$idx]", "bus_qd[$idx]"])
            append!(pq_names, ["pq_pd[$idx]", "pq_qd[$idx]"])
        elseif bus_type == 2
            append!(bus_names, ["bus_pg[$idx]", "bus_vm[$idx]"])
            append!(pv_names, ["pv_pg[$idx]", "pv_vm[$idx]"])
        elseif bus_type == 3
            append!(bus_names, ["bus_va[$idx]", "bus_vm[$idx]"])
            append!(slack_names, ["slack_va[$idx]", "slack_vm[$idx]"])
        else
            error("Unexpected bus type $(bus_type)")
        end
    end

    for i in branchkeys
        idx = parse(Int, i)
        append!(
            branch_names,
            [
                "br_r[$idx]",
                "br_x[$idx]",
                "g_fr[$idx]",
                "b_fr[$idx]",
                "g_to[$idx]",
                "b_to[$idx]",
                "tap[$idx]",
                "shift[$idx]",
            ],
        )
    end

    return vcat(bus_names, pq_names, pv_names, slack_names, branch_names)
end

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
outputs = get_outputs(pm)
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
y, _ = MOAI.add_predictor(pm.model, predictor, moai_inputs; gray_box = true)
pm_to_canos = Dict(zip(outputs, y))

# Constraints imposing a voltage mismatch on some bus
vm_pm = PowerModels.var(pm, :vm, 12)
vm_canos = pm_to_canos[vm_pm]
#@constraint(pm.model, vm_pm <= 0.90)
#@constraint(pm.model, vm_canos >= 0.94)

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
    py_y_target = nn.flatten_input_labels(py_x0)
    y_target = PythonCall.pyconvert(Vector{Float64}, py_y_target)
    diff = y_pf .- y_target
    println("Compare output from PowerModels and CANOS targets")
    println("-------------------------------------------------")
    println(@sprintf("%4s %10s %10s %13s %10s", "idx", "var", "pm_out", "canos_target", "err"))
    for i in eachindex(y_pf)
        println(@sprintf("%4d %10s %10.3f %10.3f %10.3f", i, outputs[i], y_pf[i], y_target[i], diff[i]))
    end
end
