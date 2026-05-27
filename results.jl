using Dates

const RESULTS_REPO_ROOT = @__DIR__

function _results_git_branch()
    branch = try
        readchomp(`git -C $(RESULTS_REPO_ROOT) branch --show-current`)
    catch
        ""
    end
    if isempty(branch)
        branch = try
            "detached-" * readchomp(`git -C $(RESULTS_REPO_ROOT) rev-parse --short HEAD`)
        catch
            "unknown"
        end
    end
    return replace(branch, r"[^A-Za-z0-9._-]" => "-")
end

const RESULTS_RUN_DIR = joinpath(
    RESULTS_REPO_ROOT,
    "results",
    "$(Dates.format(Dates.today(), "yyyymmdd"))-$(_results_git_branch())",
)

function results_path(base_filename::AbstractString)
    path = joinpath(RESULTS_RUN_DIR, base_filename)
    mkpath(dirname(path))
    return path
end
