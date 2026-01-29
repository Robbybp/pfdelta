#!/usr/bin/env python3
"""
Compare adversarial points to their source training points.

Outputs a table with:
  - training_point_index
  - bus
  - number of coordinates that differ
  - the differing coordinates (index: train -> adversarial)
  - 1-norm of the difference
"""
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from core.datasets.pfdelta_variants import PFDeltaCANOS
from vectorcanos import flatten_input, get_flattened_input_names


def load_dataset() -> PFDeltaCANOS:
    return PFDeltaCANOS(
        add_bus_type=True,
        case_name="case14",
        model="CANOS",
        root_dir=os.path.join("data", "pfdelta_data"),
        split="train",
        task="1.1",
    )


def summarize_differences(points_path: Path, dataset: PFDeltaCANOS) -> pd.DataFrame:
    name_list = get_flattened_input_names(dataset[0])
    data = json.loads(points_path.read_text())
    records = []
    for entry in data:
        idx = int(entry["training_point_index"])
        bus = int(entry["bus"])
        if entry["point"] is None:
            continue
        adv_vec = np.asarray(entry["point"], dtype=float)

        train_sample = dataset[idx]
        train_vec = flatten_input(train_sample).detach().cpu().numpy()

        if adv_vec.shape[0] != train_vec.shape[0]:
            raise ValueError(f"Vector length mismatch for idx {idx}: adv {adv_vec.shape[0]} vs train {train_vec.shape[0]}")

        diff = adv_vec - train_vec
        mask = ~np.isclose(adv_vec, train_vec, atol=1e-4)
        differing_indices = [i for i in np.nonzero(mask)[0] if i < len(name_list)]
        num_diff = int(len(differing_indices))

        coord_strs = [
            f"{name_list[i]}: {train_vec[i]:.6g} -> {adv_vec[i]:.6g}"
            for i in differing_indices
        ]
        l1_norm = float(np.linalg.norm(diff[differing_indices], ord=1))

        records.append(
            {
                "training_point_index": idx,
                "bus": bus,
                "num_diff_coords": num_diff,
                "l1_norm": l1_norm,
                "diff_coords": "; ".join(coord_strs),
            }
        )

    df = pd.DataFrame.from_records(records)
    df.sort_values(["training_point_index", "bus"], inplace=True)
    return df


def main(argv: Iterable[str]) -> None:
    points_path = Path(argv[1]) if len(argv) > 1 else Path("con-error-points.json")
    dataset = load_dataset()
    df = summarize_differences(points_path, dataset)
    if df.empty:
        print("No records found.")
        return
    with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
        print("All perturbations:")
        print(df.to_string(index=False))

        subset_order = [(4, 12), (0, 4), (8, 7), (9, 9), (2, 4)]
        subset = pd.concat(
            [
                df[(df["training_point_index"] == ti) & (df["bus"] == b)]
                for (ti, b) in subset_order
            ],
            axis=0,
        )
        print("\nSelected perturbations:")
        print(subset.to_string(index=False))

    outfile = "perturbation-summary.csv"
    df.to_csv(outfile)
    print(f"Wrote summary to {outfile}")


if __name__ == "__main__":
    import sys

    main(sys.argv)
