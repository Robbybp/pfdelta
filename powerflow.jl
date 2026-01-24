ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
using JuMP
using Ipopt
using PythonCall
using PowerModels
using PGLib

"""
    load_pfd_into_pm!(pm_data::Dict, py_data::Py)

Populate a PowerModels data dict with PFDelta constants (loads and branch params)
from a PFDelta HeteroData sample.
"""
function load_pfd_into_pm!(pm_data::Dict, py_data::Py)
    # Set loads in pm_data, distributing evenly across loads per bus
    load_matrix = PythonCall.pyconvert(
        Matrix{Float64},
        py_data["bus"]["bus_demand"].numpy(),
    )
    loads_by_bus = Dict(b["index"] => Any[] for b in values(pm_data["bus"]))
    nbus = length(loads_by_bus)
    busindices = map(b -> b["index"], values(pm_data["bus"]))
    @assert all(sort(busindices) .== collect(1:nbus))
    for l in values(pm_data["load"])
        b = l["load_bus"]
        push!(loads_by_bus[b], l["index"])
    end
    for i in 1:length(pm_data["bus"])
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
    branch_matrix = PythonCall.pyconvert(
        Matrix{Float64},
        py_data["bus", "branch", "bus"]["edge_attr"],
    )
    branchkeys = sort(collect(keys(pm_data["branch"])); by = k -> parse(Int, k))
    for (i, k) in enumerate(branchkeys)
        br = pm_data["branch"][k]
        br["br_r"] = branch_matrix[i, 1]
        br["br_x"] = branch_matrix[i, 2]
        br["g_fr"] = branch_matrix[i, 3]
        br["g_to"] = branch_matrix[i, 4]
        br["b_fr"] = branch_matrix[i, 5]
        br["b_to"] = branch_matrix[i, 6]
        br["tap"] = branch_matrix[i, 7]
        br["shift"] = branch_matrix[i, 8]
    end

    return pm_data
end

"""
    get_outputs(pm::PowerModels.AbstractPowerModel)

Collect CANOS-style output variables/expressions from a PowerModels model, in
the same order as moai-canos.jl vectorization: bus (va, vm), PQ (va, vm),
PV (va, qg), slack (net_p, net_q), branch (pf, qf, pt, qt).
"""
function get_outputs(pm::PowerModels.AbstractPowerModel)
    ref = pm.ref[:it][:pm][:nw][0]
    buskeys = sort(collect(keys(pm.data["bus"])); by = k -> parse(Int, k))
    branchkeys = sort(collect(keys(pm.data["branch"])); by = k -> parse(Int, k))

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
            qg = sum(PowerModels.var(pm, :qg, g) for g in ref[:bus_gens][idx]; init = 0.0)
            append!(pv_out, [va, qg])
        elseif bus_type == 3
            pg = sum(PowerModels.var(pm, :pg, g) for g in ref[:bus_gens][idx]; init = 0.0)
            qg = sum(PowerModels.var(pm, :qg, g) for g in ref[:bus_gens][idx]; init = 0.0)
            pd = sum(ref[:load][l]["pd"] for l in ref[:bus_loads][idx]; init = 0.0)
            qd = sum(ref[:load][l]["qd"] for l in ref[:bus_loads][idx]; init = 0.0)
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

"""
    solve_powerflow(point)::Vector{Float64}

Solve AC power flow for a PFDelta point and return a vector of outputs ordered
like the vectorized CANOS outputs.
"""
function solve_powerflow(point)
    # Build a PowerModels case from PGLib and overwrite constants from PFDelta point
    pm_data = PGLib.pglib("case14")  # assumes case14; adjust if other cases are used
    load_pfd_into_pm!(pm_data, point)

    pm = PowerModels.instantiate_model(pm_data, PowerModels.ACPPowerModel, PowerModels.build_pf)
    ipopt = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "linear_solver" => "ma57")
    JuMP.set_optimizer(pm.model, ipopt)
    JuMP.optimize!(pm.model)

    outputs = get_outputs(pm)
    return JuMP.value.(outputs)
end
