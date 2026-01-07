"""
JuMP + PowerModels + MOAI example:
Find a small perturbation of a base-case PFNet input while PowerModels' ACPF
predicts a voltage magnitude violation at a target bus and PFNet still reports
feasible voltages.

Key elements:
- Uses PowerModels ACPF formulation so generator setpoints, voltages, angles are
  native decision variables (no manual PF equations).
- Adds an MOAI predictor (PFNet via vector-pfnet.pt) with constraints that its
  predicted |V| stay within limits.
- Forces the mechanistic PowerModels solution to violate an upper |V| limit at
  a chosen bus.

Adjust as needed:
- `case_path`: network case.
- `target_bus`: 1-based index in the sorted bus list.
- `base_x`: replace with a real flattened PFNet input matching your trained
  model/template (see pfnet_vector_wrapper.py).
"""

ENV["JULIA_CONDAPKG_BACKEND"] = "Null" # reuse existing Python env

import JuMP
import Ipopt
import LinearAlgebra
import PythonCall
import MathOptAI as MOAI
import PowerModels

# ---------------------------------------------------------------------------
# Load PFNet predictor via MOAI (expects vector-pfnet.pt in cwd)
PythonCall.pyimport("sys").path.append(pwd())
PythonCall.pyimport("pfnet_vector_wrapper") # ensure dependencies are loaded
predictor = MOAI.PytorchModel("vector-pfnet.pt")

N = 184 # input dim for vector-pfnet.pt (case14 template); adjust if different
base_x = zeros(N) # TODO: load your true base-case flattened PFNet input here

# ---------------------------------------------------------------------------
# Load network data and build ACPF model with PowerModels
case_path = "data_generation/pglib/pglib_opf_case14_ieee.m" # edit as needed
data = PowerModels.parse_file(case_path)

# Capture base generator setpoints for regularization
gen_ids = sort([parse(Int, k) for k in keys(data["gen"])])
pg0 = Dict{Int,Float64}()
qg0 = Dict{Int,Float64}()
for gid in gen_ids
    g = data["gen"][string(gid)]
    pg0[gid] = g["pg"]
    qg0[gid] = g["qg"]
end

# Instantiate ACPF JuMP model
pm = PowerModels.instantiate_model(
    data,
    PowerModels.ACPPowerModel,
    PowerModels.build_pf;
)
model = pm.model
JuMP.set_optimizer(model, JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 5))

# Bus ordering helper
bus_ids = sort([parse(Int, k) for k in keys(data["bus"])])
bus_index = Dict(id => i for (i, id) in enumerate(bus_ids))

# Target bus for voltage violation
target_bus = 2 # 1-based index in bus_ids ordering; adjust as desired
vmax_mech = data["bus"][string(bus_ids[target_bus])]["vmax"]
#violation_margin = 0.01
violation_margin = 0.0

# ---------------------------------------------------------------------------
# Add PFNet input variable and objective term
JuMP.@variable(model, x[1:N])

# Regularize generator moves relative to base setpoints
pg_var = PowerModels.var(pm, :pg)
qg_var = PowerModels.var(pm, :qg)

JuMP.@objective(
    model,
    Min,
    sum((x[i] - base_x[i])^2 for i in 1:N) +
    10.0 * sum((pg_var[id] - pg0[id])^2 for id in gen_ids) +
    10.0 * sum((qg_var[id] - qg0[id])^2 for id in gen_ids)
)

# PFNet feasibility proxy: predicted voltages inside limits (assumes |V| in slot 1 of each 6-length bus block)
bus_stride = 6
y, _ = MOAI.add_predictor(model, predictor, x; vector_nonlinear_oracle = true)
JuMP.@constraint(model, [i = 1:length(bus_ids)], y[(i - 1) * bus_stride + 1] <= 1.06)
JuMP.@constraint(model, [i = 1:length(bus_ids)], y[(i - 1) * bus_stride + 1] >= 0.94)

# Mechanistic voltage violation at target bus (PowerModels variable vm)
vm = PowerModels.var(pm, :vm)
JuMP.@constraint(model, vm[bus_ids[target_bus]] >= vmax_mech + violation_margin)

println("Solving...")
JuMP.optimize!(model)
println("Termination status: ", JuMP.termination_status(model))
println("Objective: ", JuMP.objective_value(model))

best_x = JuMP.value.(x)
println("PFNet input deviation (L2): ", LinearAlgebra.norm(best_x .- base_x))
println(
    "Mechanistic |V| at target bus: ",
    JuMP.value(vm[bus_ids[target_bus]]),
    " (limit ",
    vmax_mech,
    ")",
)

# To save the adversarial PFNet input vector, uncomment:
# using DelimitedFiles
# DelimitedFiles.writedlm("pfnet_adversarial_x.csv", best_x)
