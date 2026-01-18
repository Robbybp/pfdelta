ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
import PythonCall
import PGLib

pglib_data = PGLib.pglib("case14")

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
pfd_data = dataset[0]

function parse_pfd_edges(pyedges)::Vector{NamedTuple}
    incident_buses = PythonCall.pyconvert(Matrix{Int}, pyedges["edge_index"])
    incident_buses .+= 1 # Python buses are 0-indexed
    params = PythonCall.pyconvert(Matrix{Float64}, pyedges["edge_attr"])
    _, nedges = size(incident_buses)
    edges = Vector{NamedTuple}()
    for i in 1:nedges
        edge = (;
            fr = incident_buses[1, i],
            to = incident_buses[2, i],
            r = params[i, 1],
            x = params[i, 2],
            gfr = params[i, 3],
            bfr = params[i, 4],
            gto = params[i, 5],
            bto = params[i, 6],
            tap = params[i, 7],
            shift = params[i, 8],
        )
        push!(edges, edge)
    end
    return edges
end

function parse_pfd_buses(pybuses)::Vector{NamedTuple}
    limits = PythonCall.pyconvert(Matrix{Float64}, pybuses["limits"])
    nbuses, _ = size(limits)
    buses = Vector{NamedTuple}()
    for i in 1:nbuses
        bus = (;
            vmin = limits[i, 1],
            vmax = limits[i, 2],
        )
        push!(buses, bus)
    end
    return buses
end

# Test edges
pfd_edges = parse_pfd_edges(pfd_data["bus", "branch", "bus"])
for (_, branch) in pglib_data["branch"]
    i = branch["index"]
    @assert isapprox(pfd_edges[i].fr, branch["f_bus"]; atol = 1e-6)
    @assert isapprox(pfd_edges[i].to, branch["t_bus"]; atol = 1e-6)
    @assert isapprox(pfd_edges[i].r, branch["br_r"]; atol = 1e-6)
    @assert isapprox(pfd_edges[i].x, branch["br_x"]; atol = 1e-6)
end

# Test buses
pfd_buses = parse_pfd_buses(pfd_data["bus"])
for (_, bus) in pglib_data["bus"]
    i = bus["index"]
    @assert isapprox(pfd_buses[i].vmin, bus["vmin"]; atol = 1e-6)
    @assert isapprox(pfd_buses[i].vmax, bus["vmax"]; atol = 1e-6)
end
