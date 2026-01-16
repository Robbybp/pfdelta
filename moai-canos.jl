ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
import JuMP
import Ipopt
import MathOptInterface as MOI
import PythonCall
import MathOptAI as MOAI

# Requires torch and torch_geometric
PythonCall.pyimport("sys").path.append(pwd())
PythonCall.pyimport("vectorcanos")
predictor = MOAI.PytorchModel("vector-canos.pt")

N = 312
ipopt = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6) #, "linear_solver" => "ma57")
model = JuMP.Model(ipopt)

import PowerModels as PM
casefile = joinpath("data_generation", "pglib", "pglib_opf_case14_ieee.m")
casedata = PM.parse_file(casefile)
pm = PM.instantiate_model(casedata, PM.ACPPowerModel, PM.build_pf; jump_model = model)

#JuMP.set_optimizer_attribute(model, "linear_solver", "ma57")
JuMP.@variable(model, x[1:N], start = 0.5)
y, formulation = MOAI.add_predictor(model, predictor, x, gray_box = true)
xref = ones(N)
#JuMP.@objective(model, Min, sum((x .- xref).^2))
# I just chose random numbers here, but we'd want to constrain a voltage or something
# to go over its limit.
JuMP.@constraint(model, y[2] >= 1.0)
JuMP.optimize!(model)
