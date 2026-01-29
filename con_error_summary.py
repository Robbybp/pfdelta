#!/usr/bin/env python3
"""
Summarize adversarially-constrained error runs by training point.

For each training_point_index, report:
  - problems converged (primal_status in {FEASIBLE_POINT, NEARLY_FEASIBLE_POINT})
  - average 1-norm distance between adversarial and training point (computed directly)
  - average support (number of coords differing above 1e-4)
  - average NN output
  - average PF output
"""
import json
import os
from pathlib import Path
from typing import Iterable, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

CONVERGED_STATUSES: Tuple[str, ...] = ("FEASIBLE_POINT", "NEARLY_FEASIBLE_POINT")

def load_dataset():
    from core.datasets.pfdelta_variants import PFDeltaCANOS
    return PFDeltaCANOS(
        add_bus_type=True,
        case_name="case14",
        model="CANOS",
        root_dir=os.path.join("data", "pfdelta_data"),
        split="train",
        task="1.1",
    )

def flatten_training_point(sample):
    from vectorcanos import flatten_input
    return flatten_input(sample).detach().cpu().numpy()

def compute_distances(csv_path: Path, points_path: Path) -> pd.DataFrame:
    pts = json.loads(points_path.read_text())
    pts_lookup = {
        (int(p["training_point_index"]), int(p["bus"])): np.asarray(p["point"], float)
        for p in pts
    }

    df = pd.read_csv(csv_path)
    converged = df[df["primal_status"].isin(CONVERGED_STATUSES)].copy()

    dataset = load_dataset()

    records = []
    for _, row in converged.iterrows():
        idx = int(row["training_point_index"])
        bus = int(row["bus"])
        key = (idx, bus)
        if key not in pts_lookup:
            continue
        adv = pts_lookup[key]
        train = flatten_training_point(dataset[idx])
        if adv.shape != train.shape:
            continue
        diff = adv - train
        l1 = float(np.linalg.norm(diff, ord=1))
        support = int(np.count_nonzero(np.abs(diff) > 1e-4))
        records.append(
            {
                "training_point_index": idx,
                "bus": bus,
                "distance_l1": l1,
                "support": support,
                "nn_output": row["nn_output"],
                "pf_output": row["pf_output"],
            }
        )
    return pd.DataFrame.from_records(records)


def summarize(dist_df: pd.DataFrame) -> pd.DataFrame:
    if dist_df.empty:
        return dist_df
    grouped = (
        dist_df.groupby("training_point_index")
        .agg(
            converged=("bus", "size"),
            avg_distance_l1=("distance_l1", "mean"),
            avg_support=("support", "mean"),
            avg_nn_output=("nn_output", "mean"),
            avg_pf_output=("pf_output", "mean"),
        )
        .reset_index()
        .sort_values("training_point_index")
    )
    # HACK: Bus input values are duplicated in the input vector,
    # so we divide the support by 2 to get the actual number of
    # different "unique physical quantities"
    grouped["avg_support"] /= 2.0
    # For consistency, we divide this by two as well...
    grouped["avg_distance_l1"] /= 2.0
    grouped["avg_support"] = grouped["avg_support"].round()
    return grouped


def main(argv: Iterable[str]) -> None:
    infile = Path(argv[1]) if len(argv) > 1 else Path("con-error-sweep.csv")
    points_path = Path(argv[2]) if len(argv) > 2 else Path("con-error-points.json")
    outfile = Path(argv[3]) if len(argv) > 3 else Path("distance-histogram.pdf")

    dist_df = compute_distances(infile, points_path)
    summary = summarize(dist_df)
    if summary.empty:
        print("No converged rows found.")
        return

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # HACK: We divide distance by two because all bus inputs are duplicated.
    # We have to do this after computing the summary because we divide by
    # two after grouping that function as well (at the same time that we divide
    # the average support by two)..........
    dist_df["distance_l1"] /= 2.0
    # Histogram over all converged L1 distances
    conv_obj = dist_df["distance_l1"].dropna()
    if conv_obj.empty:
        print("No distance values to plot.")
        return

    vmin, vmax = conv_obj.min(), conv_obj.max()
    #if vmax <= 7 or vmin >= 3:
    #    fig = plt.figure(figsize=(8, 4))
    #    plt.hist(conv_obj, bins=30, color="#4C72B0", edgecolor="white")
    #    plt.xlabel("Objective")
    #    plt.ylabel("Count")
    #    #fig.supxlabel("Objective", y=0.02)
    #    fig.tight_layout()
    #    fig.savefig(outfile, dpi=200, transparent=True)
    #    print(f"Saved histogram to {outfile}")
    #    return

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5, 3), sharey=True, gridspec_kw={"width_ratios": [3, 1]}
    )
    bins = np.linspace(vmin, vmax, 30)
    ax1.hist(conv_obj, bins=bins, color="#4C72B0", edgecolor="white")
    ax2.hist(conv_obj, bins=bins, color="#4C72B0", edgecolor="white")

    ax1.set_xlim(0.0, 0.5)
    ax2.set_xlim(0.9, 1.1)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.tick_params(left=False, right=False)

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    #fig.suptitle("Perturbations required to satisfy adversarial constraints", x=0.53, fontsize=14)
    ax1.set_ylabel("Count", fontsize=14)
    #ax1.set_xlabel("Objective")
    #ax2.set_xlabel("Objective")
    fig.supxlabel("$\\left\\| x - x_0 \\right\\|_1$", y=0.06, x = 0.55, fontsize=14)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, transparent=True)
    print(f"Saved histogram with broken x-axis to {outfile}")


if __name__ == "__main__":
    main(sys.argv)
