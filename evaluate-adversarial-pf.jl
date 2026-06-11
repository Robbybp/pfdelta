ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using JSON
using PythonCall

# Reuse power flow helpers
include("moai-canos-model.jl")  # provides load_pfd_into_pm!, solve_powerflow, etc.

# Python imports
PythonCall.pyimport("sys").path.append(pwd())
VC = PythonCall.pyimport("vectorcanos")
torch = PythonCall.pyimport("torch")
pfdelta_variants = PythonCall.pyimport("core.datasets.pfdelta_variants")
PFDeltaCANOS = pfdelta_variants.PFDeltaCANOS
canos_pf = PythonCall.pyimport("core.models.canos_pf")
CANOS_PF = canos_pf.CANOS_PF

# Build dataset and wrapper for unflattening inputs
dataset = PFDeltaCANOS(
    add_bus_type = true,
    case_name = "case14",
    model = "CANOS",
    root_dir = joinpath("data", "pfdelta_data"),
    split = "train",
    task = "1.1",
)
template_sample = dataset[0]
hidden_dim = 128
include_sent_messages = true
k_steps = 15
canos = CANOS_PF(dataset, hidden_dim, include_sent_messages, k_steps)
wrapper = VC.VectorCanos(canos, template_sample)

function point_to_struct(entry)
    x_flat = torch.tensor(entry["point"]; dtype=torch.float32)
    idx = haskey(entry, "training_point_index") ? Int(entry["training_point_index"]) : 0
    template = dataset[idx]
    py_data = VC.overwrite_inputs(
        template,
        x_flat,
        wrapper.input_sizes,
        wrapper.node_input_keys,
        wrapper.edge_input_keys,
        wrapper.input_shapes,
    )
    return py_data
end

function evaluate_file(input_path::String, output_path::String)
    entries = JSON.parsefile(input_path)
    results = Vector{Any}()
    for entry in entries
        if isnothing(entry["point"])
            pf_outputs = missing
        else
            py_point = point_to_struct(entry)
            pf_outputs = solve_powerflow(py_point)
        end
        push!(results, Dict(
            "training_point_index" => get(entry, "training_point_index", missing),
            "bus" => get(entry, "bus", missing),
            "sense" => get(entry, "sense", missing),
            "pf_output" => pf_outputs,
        ))
    end
    JSON.print(open(output_path, "w"), results)
    println("Wrote $(length(results)) labels to $output_path")
end

# TODO: Right input is probably a comma-separated list of input files
RESULTS_DIR = joinpath("results", "20260527-powerup2026-merge")
fnames = [
    ("con-error-points.json", "con-error-labels.json"),
    ("max-error-points.json", "max-error-labels.json"),
]
for (infile, outfile) in fnames
    infile = joinpath(RESULTS_DIR, infile)
    outfile = joinpath(RESULTS_DIR, outfile)
    evaluate_file(infile, outfile)
end
