This fork and branch of the [pfdelta](https://github.com/MOSSLab-MIT/pfdelta) repository
exists to produce results for the paper "Generating adversarial inputs for a graph
neural network model of AC power flow", to be presented at PowerUp 2026.
```bibtex
@misc{parker2026adversarial,
      title={Generating adversarial inputs for a graph neural network model of {AC} power flow}, 
      author={Robert Parker},
      year={2026},
      eprint={2602.17975},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.17975}, 
}
```

## Reproducing the results

The following scripts may be used to produce the results in the paper:
- `download.py` &mdash; A helper script to download the Case-14 Task 1.1 data.
- `test-data.jl` &mdash; Asserts that constant parameters are consistent between
  the PFDelta dataset and PGLib case file for Case-14.
- `validate-labels.jl` &mdash; Solves an ACPF problem for every point in the Case-14
  Task 1.1 data and records any cases where the solution differs from this point's
  label by more than `1e-5` (max norm)
- `canos_mwe.py` &mdash; Runs a single forward pass on the CANOS-OPF model and
  displays the shape of input and output data. This is not very important here
  as all the results are collected on the CANOS-PF model.
- `validate_canos.py` &mdash; Evaluates loss of the trained CANOS-PF model
  on train and test data
- `vectorcanos.py` &mdash; Contains the `VectorCanos` class that wraps the structured
  CANOS-PF NN. When run as a script, passes the first training point through the vectorized
  wrapper and evaluates loss. (Doesn't seem to compare loss to anything else though...)
  This script also writes the `vector-canos.pt` file.
  - This script only operates on an untrained CANOS-PF instance, so it isn't that important
    as a script
- `test_canos_trained.py` &mdash; Loads weights from a `model.pt` file in a hard-coded
  "run directory" and saves to a `vectorcanos-trained.pt` file
- `moai-canos.jl` &mdash; A script for solving a single instance of the constrained-error
  problem. This script is intended to provide a MWE of the research objective: An adversarial
  point where the NN says "feasible" but ACPF reveals that the point is infeasible.

The following files contain helper functions:
- `moai-canos-model.jl` &mdash; Functions for performing JuMP/PowerModels solves with
  input data in the form of a (flattened) PFDelta data point
