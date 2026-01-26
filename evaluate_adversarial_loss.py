#!/usr/bin/env python3
"""
Evaluate losses on adversarial points from con-error-points.json and max-error-points.json.

Reports mean/std/max for:
  - CANOS MSE loss
  - Power balance loss
"""
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from core.datasets.pfdelta_variants import PFDeltaCANOS
from core.models.canos_pf import CANOS_PF
from core.utils.pf_losses_utils import CANOS_PF_MSE, PowerBalanceLoss
from vectorcanos import VectorCanos


def load_dataset() -> PFDeltaCANOS:
    return PFDeltaCANOS(
        add_bus_type=True,
        case_name="case14",
        model="CANOS",
        root_dir=os.path.join("data", "pfdelta_data"),
        split="train",
        task="1.1",
    )


def load_model(dataset: PFDeltaCANOS, model_path: Path) -> VectorCanos:
    sample = dataset[0]
    hidden_dim = 128
    include_sent_messages = True
    k_steps = 15
    canos = CANOS_PF(dataset, hidden_dim, include_sent_messages, k_steps)
    state = torch.load(model_path, map_location="cpu")
    canos.load_state_dict(state)
    canos.eval()
    return VectorCanos(canos, sample)


def iter_points_with_labels(point_path: Path, label_path: Path):
    point_entries = json.loads(point_path.read_text())
    label_entries = json.loads(label_path.read_text())
    if len(point_entries) != len(label_entries):
        raise ValueError(f"Length mismatch: {point_path} has {len(point_entries)} points, but {label_path} has {len(label_entries)} labels")
    for p_entry, l_entry in zip(point_entries, label_entries):
        vec = p_entry.get("point")
        label_vec = l_entry.get("pf_output")
        if vec is None or label_vec is None:
            continue
        yield vec, label_vec


def summarize(name: str, values: List[float]):
    t = torch.tensor(values)
    mean = t.mean().item()
    std = t.std(unbiased=False).item()
    maxv = t.max().item()
    print(f"{name:18s} mean={mean:.6f}  std={std:.6f}  max={maxv:.6f}")


def prepare_batches(data):
    # Set batch vectors so CANOS sees a single graph
    for key in ["bus", "PQ", "PV", "slack"]:
        if key in data:
            node = data[key]
            n = node.x.shape[0] if hasattr(node, "x") else getattr(node, "num_nodes", None)
            if n is not None:
                node.batch = torch.zeros(n, dtype=torch.long)


def augment_aux_fields(data, labels):
    """
    Reconstruct auxiliary fields (bus_gen, bus_demand, bus_voltages) from inputs/labels.
    These cannot be copied from the template because adversarial points change inputs.
    """
    device = data["bus"].x.device
    nbus = data["bus"].num_nodes
    bus_gen = torch.zeros(nbus, 2, device=device)
    bus_demand = torch.zeros(nbus, 2, device=device)
    bus_voltages = torch.zeros(nbus, 2, device=device)

    edge_dict = data.edge_index_dict if hasattr(data, "edge_index_dict") else data._edge_index_dict

    pq_bus_idx = edge_dict.get(("PQ", "PQ_link", "bus"), torch.empty(2, 0, dtype=torch.long, device=device))[1]
    pv_bus_idx = edge_dict.get(("PV", "PV_link", "bus"), torch.empty(2, 0, dtype=torch.long, device=device))[1]
    slack_bus_idx = edge_dict.get(("slack", "slack_link", "bus"), torch.empty(2, 0, dtype=torch.long, device=device))[1]

    if pq_bus_idx.numel() > 0:
        bus_demand[pq_bus_idx] = data["PQ"].x
        bus_voltages[pq_bus_idx] = labels["PQ"]

    if pv_bus_idx.numel() > 0:
        vm = data["PV"].x[:, 1]
        va = labels["PV"][:, 1]
        bus_voltages[pv_bus_idx, 0] = va
        bus_voltages[pv_bus_idx, 1] = vm

    if slack_bus_idx.numel() > 0:
        bus_voltages[slack_bus_idx] = data["slack"].x

    data["bus"].bus_gen = bus_gen
    data["bus"].bus_demand = bus_demand
    data["bus"].bus_voltages = bus_voltages


def main(argv: Iterable[str]) -> None:
    model_path = (
        Path(argv[1])
        if len(argv) > 1
        else Path(
            "runs/canos_task_1_1/canos_k_steps15_hd128_lr5e-4_task_1_1_260116_121912/model.pt"
        )
    )
    con_path = Path(argv[2]) if len(argv) > 2 else Path("con-error-points.json")
    max_path = Path(argv[3]) if len(argv) > 3 else Path("max-error-points.json")
    con_label_path = Path(argv[4]) if len(argv) > 4 else Path("con-error-labels.json")
    max_label_path = Path(argv[5]) if len(argv) > 5 else Path("max-error-labels.json")
    point_sets: List[Tuple[Path, Path]] = [
        (con_path, con_label_path),
        (max_path, max_label_path),
    ]

    dataset = load_dataset()
    wrapper = load_model(dataset, model_path)

    mse_loss_fn = CANOS_PF_MSE()
    pb_loss_fn = PowerBalanceLoss("CANOS")

    mse_vals: List[float] = []
    pb_vals: List[float] = []

    with torch.no_grad():
        for p_path, l_path in point_sets:
            for vec, label_vec in iter_points_with_labels(p_path, l_path):
                x_flat = torch.tensor(vec, dtype=torch.float32)
                y_flat = torch.tensor(label_vec, dtype=torch.float32)
                data = wrapper.unflatten_input(x_flat)
                labels = wrapper.unflatten_output(y_flat)
                # Attach labels
                data["bus"].y = labels["bus"]
                data["PQ"].y = labels["PQ"]
                data["PV"].y = labels["PV"]
                data["slack"].y = labels["slack"]
                data[("bus", "branch", "bus")].edge_label = labels["edge_preds"]
                augment_aux_fields(data, labels)

                nbus, _ = data["bus"].x.shape
                data["bus"].batch = torch.tensor([0]*nbus)
                #prepare_batches(data)

                out = wrapper.model(data)
                mse_vals.append(float(mse_loss_fn(out, data)))
                pb_vals.append(float(pb_loss_fn(out, data)))

    if not mse_vals:
        print("No points to evaluate.")
        return

    print(f"Evaluated {len(mse_vals)} points from {[str(p) for p, _ in point_sets]}")
    summarize("CANOS MSE", mse_vals)
    summarize("Power balance", pb_vals)


if __name__ == "__main__":
    import sys

    main(sys.argv)
