#!/usr/bin/env python3
"""
Summarize optimization solves for con-error-sweep.csv and max-error-sweep.csv.

Reports per-file:
  - number of converged problems (primal_status in {FEASIBLE_POINT, NEARLY_FEASIBLE_POINT})
  - average iterations
  - average solve time
"""
import pandas as pd
import argparse
import os
from pathlib import Path

CONVERGED = {"FEASIBLE_POINT", "NEARLY_FEASIBLE_POINT"}

def summarize_file(path: Path) -> dict:
    df = pd.read_csv(path)
    conv = df[df["primal_status"].isin(CONVERGED)]
    return {
        "problem": path.name,
        "converged": len(conv),
        "avg_iter": conv["n_iter"].mean() if not conv.empty else 0.0,
        "avg_solve_time": conv["solve_time"].mean() if not conv.empty else 0.0,
    }


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("results_dir", help="Directory where results files live")
    args = argparser.parse_args()
    files = [
        Path(args.results_dir, "con-error-sweep.csv"),
        Path(args.results_dir, "max-error-sweep.csv"),
    ]
    rows = [summarize_file(p) for p in files]
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", None):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
