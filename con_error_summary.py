#!/usr/bin/env python3
"""
Summarize adversarially-constrained error runs by training point.

For each training_point_index, report:
  - problems converged (primal_status in {FEASIBLE_POINT, NEARLY_FEASIBLE_POINT})
  - average objective (distance)
  - average NN output
  - average PF output
"""
from pathlib import Path
from typing import Iterable, Tuple
import sys

import matplotlib.pyplot as plt

import pandas as pd


CONVERGED_STATUSES: Tuple[str, ...] = ("FEASIBLE_POINT", "NEARLY_FEASIBLE_POINT")


def summarize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    converged = df[df["primal_status"].isin(CONVERGED_STATUSES)].copy()

    grouped = (
        converged.groupby("training_point_index")
        .agg(
            converged=("primal_status", "size"),
            avg_objective=("objective", "mean"),
            avg_nn_output=("nn_output", "mean"),
            avg_pf_output=("pf_output", "mean"),
        )
        .reset_index()
        .sort_values("training_point_index")
    )
    return grouped


def main(argv: Iterable[str]) -> None:
    infile = Path(argv[1]) if len(argv) > 1 else Path("con-error-sweep.csv")
    outfile = Path(argv[2]) if len(argv) > 2 else Path("distance-histogram.pdf")

    summary = summarize(infile)
    if summary.empty:
        print("No converged rows found.")
        return

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Histogram over all converged objective values (drop missing)
    df = pd.read_csv(infile)
    conv_obj = df[df["primal_status"].isin(CONVERGED_STATUSES)]["objective"].dropna()
    if conv_obj.empty:
        print("No objective values to plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(conv_obj, bins=30, color="#4C72B0", edgecolor="white")
    plt.xlabel("Objective")
    plt.ylabel("Count")
    plt.title("Objective Histogram (Converged)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, transparent=True)
    print(f"Saved histogram to {outfile}")


if __name__ == "__main__":
    main(sys.argv)
