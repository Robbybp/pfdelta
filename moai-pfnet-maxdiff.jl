"""
JuMP + PowerModels + MOAI example:
Maximize the total squared deviation between mechanistic bus voltage magnitudes
(from an ACPF solve) and PFNet-predicted voltage magnitudes, while enforcing
that PFNet itself predicts a feasible voltage range.

Objective (maximize):
  sum_i (vm_mech[i] - vm_pfnet[i])^2
  - λx * ||x - base_x||^2
  - λg * (||pg - pg0||^2 + ||qg - qg0||^2)

Adjust:
- `case_path` for the network.
- Regularization weights `lambda_x`, `lambda_g`.
"""

ENV["JULIA_CONDAPKG_BACKEND"] = "Null" # reuse existing Python env

using JuMP
using Ipopt
using LinearAlgebra
using PythonCall
import MathOptAI as MOAI
import PowerModels

# ---------------------------------------------------------------------------
# Load PFNet predictor via MOAI (expects vector-pfnet.pt in cwd)
PythonCall.pyimport("sys").path.append(pwd())
torch = pyimport("torch")
pyimport("pfnet_vector_wrapper") # ensure class is registered for torch.load
predictor = MOAI.PytorchModel("vector-pfnet.pt")

# Get a real base-case PFNet input from the saved wrapper
wrapper = torch.load("vector-pfnet.pt", map_location="cpu")
base_x_py = wrapper.flatten_input_from_data(wrapper.template).detach().cpu().numpy()
base_x = pyconvert(Vector{Float64}, base_x_py)
N = length(base_x)

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
set_optimizer(model, Ipopt.Optimizer)
set_optimizer_attribute(model, "linear_solver", "ma27")
set_optimizer_attribute(model, "print_level", 5)

# Bus ordering helper
bus_ids = sort([parse(Int, k) for k in keys(data["bus"])])

# ---------------------------------------------------------------------------
# Decision variables for PFNet input and link to predictor
@variable(model, x[i=1:N], start = base_x[i])
bus_stride = 6
y, _ = MOAI.add_predictor(model, predictor, x; vector_nonlinear_oracle = true)
nn = torch.load(predictor.filename)
nnout = nn(torch.tensor(base_x_py)).detach().cpu().numpy()
nnout = PythonCall.pyconvert(Vector{Float64}, nnout)
for i in 1:length(y)
    JuMP.set_start_value(y[i], nnout[i])
end

# PFNet feasibility proxy: predicted voltages inside limits (assumes |V| in slot 1 of each 6-length bus block)
@constraint(model, [i = 1:length(bus_ids)], y[(i - 1) * bus_stride + 1] <= 1.06)
@constraint(model, [i = 1:length(bus_ids)], y[(i - 1) * bus_stride + 1] >= 0.94)

# Mechanistic variables from PowerModels
vm = PowerModels.var(pm, :vm)
pg_var = PowerModels.var(pm, :pg)
qg_var = PowerModels.var(pm, :qg)

# Regularization weights
lambda_x = 0.1
lambda_g = 0.1

# Objective: maximize squared deviation between mechanistic vm and PFNet vm
@objective(
    model,
    Max,
    sum((vm[bus_ids[i]] - y[(i - 1) * bus_stride + 1])^2 for i in 1:length(bus_ids)) -
    lambda_x * sum((x[i] - base_x[i])^2 for i in 1:N) -
    lambda_g * (
        sum((pg_var[id] - pg0[id])^2 for id in gen_ids) +
        sum((qg_var[id] - qg0[id])^2 for id in gen_ids)
    )
)

println("Solving vm-gap maximization...")
JuMP.optimize!(model)
println("Termination status: ", JuMP.termination_status(model))
println("Objective: ", JuMP.objective_value(model))

best_x = value.(x)
vm_gap = sum((value(vm[bus_ids[i]]) - value(y[(i - 1) * bus_stride + 1]))^2 for i in 1:length(bus_ids))
println("Squared vm gap: ", vm_gap)
println("PFNet input deviation (L2): ", norm(best_x .- base_x))

# To save the PFNet input vector, uncomment:
# using DelimitedFiles
# DelimitedFiles.writedlm("pfnet_vm_gap_x.csv", best_x)
