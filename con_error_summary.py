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
import numpy as np

import pandas as pd

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

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

    vmin, vmax = conv_obj.min(), conv_obj.max()
    if vmax <= 7 or vmin >= 3:
        fig = plt.figure(figsize=(8, 4))
        plt.hist(conv_obj, bins=30, color="#4C72B0", edgecolor="white")
        plt.xlabel("Objective")
        plt.ylabel("Count")
        #fig.supxlabel("Objective", y=0.02)
        fig.tight_layout()
        fig.savefig(outfile, dpi=200, transparent=True)
        print(f"Saved histogram to {outfile}")
        return

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5, 4), sharey=True, gridspec_kw={"width_ratios": [3, 2]}
    )
    bins = np.linspace(vmin, vmax, 30)
    ax1.hist(conv_obj, bins=bins, color="#4C72B0", edgecolor="white")
    ax2.hist(conv_obj, bins=bins, color="#4C72B0", edgecolor="white")

    ax1.set_xlim(vmin, 2.5)
    ax2.set_xlim(7, vmax)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    fig.suptitle("Perturbations required to satisfy adversarial constraints", x=0.53)
    ax1.set_ylabel("Count")
    #ax1.set_xlabel("Objective")
    #ax2.set_xlabel("Objective")
    fig.supxlabel("$\\left\\| x - x_0 \\right\\|_1$", y=0.06, x = 0.55)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, transparent=True)
    print(f"Saved histogram with broken x-axis to {outfile}")


if __name__ == "__main__":
    main(sys.argv)
