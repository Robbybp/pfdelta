# Just use whatever python environment we are already in
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
import JuMP
import Ipopt
import MathOptInterface as MOI
import PythonCall
import MathOptAI as MOAI

# Requires torch and torch_geometric
PythonCall.pyimport("sys").path.append(pwd())
PythonCall.pyimport("pfnet_vector_wrapper")
predictor = MOAI.PytorchModel("vector-pfnet.pt")

N = 184
model = JuMP.Model(Ipopt.Optimizer)
JuMP.@variable(model, x[1:N], start = 0.0)
y, formulation = MOAI.add_predictor(model, predictor, x, vector_nonlinear_oracle = true)
xref = ones(N)
JuMP.@objective(model, Min, sum((x .- xref).^2))
# I just chose random numbers here, but we'd want to constrain a voltage or something
# to go over its limit.
JuMP.@constraint(model, y[30] >= 0.5)
JuMP.optimize!(model)
