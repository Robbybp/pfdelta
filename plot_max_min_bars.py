#!/usr/bin/env python3
"""
Plot grouped bar charts of |min| and |max| objective differences per bus.

Creates two subplots: one for PQ (bustype==1) and one for PV+slack (bustype in {2,3}).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"


def build_bus_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to get |objective| for min/max sense per bus."""
    df = df.copy()
    df = df[~df["objective"].isna()]
    df["abs_objective"] = df["objective"].abs()
    df["bustype"] = df["bustype"].astype(int)
    pivot = (
        df.pivot_table(
            index=["bus", "bustype"],
            columns="sense",
            values="abs_objective",
            aggfunc="max",
        )
        .reset_index()
    )
    # Ensure both columns exist
    for col in ("min", "max"):
        if col not in pivot.columns:
            pivot[col] = 0.0
    return pivot


def plot_group(ax, data: pd.DataFrame, title: str, ylim=None, ylabel=None):
    """Render grouped bars for a subset of buses."""
    if data.empty:
        ax.set_visible(False)
        return

    buses = data["bus"].astype(int).tolist()
    min_vals = data.get("min", pd.Series([0] * len(buses))).fillna(0).tolist()
    max_vals = data.get("max", pd.Series([0] * len(buses))).fillna(0).tolist()

    x = np.arange(len(buses))
    width = 0.35
    ax.bar(x - width / 2, min_vals, width, label=r"$\left|\min\left(y_{\textrm{NN},i}-y_{\textrm{PF},i}\right)\right|$", color="#5DA5DA")
    ax.bar(x + width / 2, max_vals, width, label=r"$\left|\max\left(y_{\textrm{NN},i}-y_{\textrm{PF},i}\right)\right|$", color="#F15854")

    ax.set_xticks(x)
    ax.set_xticklabels(buses)
    ax.set_xlabel("Bus", fontsize=14)
    if ylabel is None:
        ax.set_ylabel("Error", fontsize=14)
    else:
        ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend()


def main():
    infile = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("max-error-sweep.csv")
    outfile = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("max-min-objectives.pdf")

    df = pd.read_csv(infile)
    pivot = build_bus_frame(df)

    pq = pivot[pivot["bustype"] == 1].sort_values("bus")
    pv_slack = pivot[pivot["bustype"].isin([2, 3])].sort_values("bus")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharey=False)
    plot_group(ax1, pq, "PQ Buses", ylabel="$v$ error (per-unit)")
    plot_group(ax2, pv_slack, "PV + Reference Buses", ylabel="$q$ error (per-unit)")

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, transparent=True)
    print(f"Saved bar charts to {outfile}")


if __name__ == "__main__":
    main()
