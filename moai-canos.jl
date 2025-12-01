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
model = JuMP.Model(Ipopt.Optimizer)
JuMP.@variable(model, x[1:N], start = 0.5)
y, formulation = MOAI.add_predictor(model, predictor, x)
xref = ones(N)
JuMP.@objective(model, Min, sum((x .- xref).^2))
# I just chose random numbers here, but we'd want to constrain a voltage or something
# to go over its limit.
JuMP.@constraint(model, y[2] >= 2.0)
JuMP.optimize!(model)
