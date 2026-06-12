ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using PythonCall
using Printf

include("moai-canos-model.jl")

const TRAINING_POINT_INDEX = 4
const PERTURBED_BUS = 6
const PERTURBED_VM = 0.970353

PythonCall.pyimport("sys").path.append(pwd())
pfdelta_variants = PythonCall.pyimport("core.datasets.pfdelta_variants")
PFDeltaCANOS = pfdelta_variants.PFDeltaCANOS

dataset = PFDeltaCANOS(
    add_bus_type = true,
    case_name = "case14",
    model = "CANOS",
    root_dir = joinpath("data", "pfdelta_data"),
    split = "train",
    task = "1.1",
)

builtins = PythonCall.pybuiltins
py_globals = builtins.dict()
builtins.exec(
    """
def make_perturbed_point(dataset, training_point_index, bus_number, vm):
    data = dataset[training_point_index].clone()
    bus_idx = bus_number - 1
    original_vm = float(data["bus"].x[bus_idx, 1])

    data["bus"].x[bus_idx, 1] = vm
    data["bus"].bus_voltages[bus_idx, 1] = vm

    for node_type, edge_type in [
        ("PQ", ("PQ", "PQ_link", "bus")),
        ("PV", ("PV", "PV_link", "bus")),
        ("slack", ("slack", "slack_link", "bus")),
    ]:
        bus_indices = data[edge_type].edge_index[1]
        matches = (bus_indices == bus_idx).nonzero(as_tuple=True)[0]
        if matches.numel() > 0:
            data[node_type].x[matches[0], 1] = vm
            break

    return data, original_vm
""",
    py_globals,
)
make_perturbed_point = py_globals["make_perturbed_point"]

perturbation_result = make_perturbed_point(
    dataset,
    TRAINING_POINT_INDEX,
    PERTURBED_BUS,
    PERTURBED_VM,
)
point = perturbation_result[0]
original_vm = PythonCall.pyconvert(Float64, perturbation_result[1])

pm_data = PGLib.pglib("case14")
load_pfd_into_pm!(pm_data, point)
pm = PowerModels.instantiate_model(
    pm_data,
    PowerModels.ACPPowerModel,
    PowerModels.build_opf,
)
_, output_names, _ = get_outputs(pm)

pf_output = solve_powerflow(point)

println(
    @sprintf(
        "training_point_index=%d  vm[%d] %.6f -> %.6f",
        TRAINING_POINT_INDEX,
        PERTURBED_BUS,
        original_vm,
        PERTURBED_VM,
    )
)
println("AC power-flow voltage magnitudes at PQ buses:")
for (i, name) in enumerate(output_names)
    m = match(r"^pq_vm\[(\d+)\]$", name)
    isnothing(m) && continue
    bus = parse(Int, m.captures[1])
    println(@sprintf("  bus %2d: %.6f", bus, pf_output[i]))
end
