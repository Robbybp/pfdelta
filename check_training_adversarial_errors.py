#!/usr/bin/env python3
"""Check selected CANOS training-label errors for adversarial candidates."""
import argparse
import os
from pathlib import Path
from typing import Iterable

import torch
from torch_geometric.loader import DataLoader

from core.datasets.pfdelta_variants import PFDeltaCANOS
from core.models.canos_pf import CANOS_PF


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report max CANOS training-set errors for selected outputs."
    )
    parser.add_argument(
        "model_file",
        type=Path,
        help=".pt file containing trained CANOS weights",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(list(argv))


def load_dataset() -> PFDeltaCANOS:
    return PFDeltaCANOS(
        add_bus_type=True,
        case_name="case14",
        model="CANOS",
        root_dir=os.path.join("data", "pfdelta_data"),
        split="train",
        task="1.1",
    )


def load_model(dataset: PFDeltaCANOS, model_path: Path, device: torch.device) -> CANOS_PF:
    model = CANOS_PF(
        dataset=dataset,
        hidden_dim=128,
        include_sent_messages=True,
        k_steps=15,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def empty_record(name: str) -> dict:
    return {
        "name": name,
        "error": -1.0,
        "prediction": None,
        "label": None,
        "training_point_index": None,
        "bus": None,
    }


def local_bus_numbers(batch, node_type: str, edge_type) -> torch.Tensor:
    graph_idx = batch[node_type].batch
    global_bus_idx = batch[edge_type].edge_index[1]
    return global_bus_idx - batch["bus"].ptr[graph_idx] + 1


def update_record(
    record: dict,
    errors: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    graph_indices: torch.Tensor,
    bus_numbers: torch.Tensor,
    batch_start: int,
) -> None:
    if errors.numel() == 0:
        return

    error, row = torch.max(errors.detach().cpu(), dim=0)
    error = float(error)
    if error <= record["error"]:
        return

    row = int(row)
    graph_idx = int(graph_indices[row].detach().cpu())
    record.update(
        error=error,
        prediction=float(predictions[row].detach().cpu()),
        label=float(labels[row].detach().cpu()),
        training_point_index=batch_start + graph_idx,
        bus=int(bus_numbers[row].detach().cpu()),
    )


def print_record(record: dict) -> None:
    print(
        f"{record['name']:16s} "
        f"max_abs_error={record['error']:.8f}  "
        f"prediction={record['prediction']:.8f}  "
        f"label={record['label']:.8f}  "
        f"training_point_index={record['training_point_index']}  "
        f"bus={record['bus']}"
    )


def matching_training_points(
    pq_pred: torch.Tensor,
    pq_label: torch.Tensor,
    pq_graph_indices: torch.Tensor,
    batch_start: int,
) -> set[int]:
    mask = (pq_pred >= 0.94) & (pq_label <= 0.9)
    graph_indices = pq_graph_indices[mask].detach().cpu().unique()
    return {batch_start + int(i) for i in graph_indices.tolist()}


def main(argv: Iterable[str]) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    dataset = load_dataset()
    model = load_model(dataset, args.model_file, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    records = {
        "pv_q": empty_record("PV reactive"),
        "slack_q": empty_record("slack reactive"),
        "pq_vm": empty_record("PQ voltage"),
    }
    constrained_training_points: set[int] = set()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_start = batch_idx * args.batch_size
            batch = batch.to(device)
            output = model(batch)

            pv_pred = output["PV"][:, 0]
            pv_label = batch["PV"].y[:, 0]
            pv_bus = local_bus_numbers(batch, "PV", ("PV", "PV_link", "bus"))
            update_record(
                records["pv_q"],
                torch.abs(pv_pred - pv_label),
                pv_pred,
                pv_label,
                batch["PV"].batch,
                pv_bus,
                batch_start,
            )

            slack_pred = output["slack"][:, 1]
            slack_label = batch["slack"].y[:, 1]
            slack_bus = local_bus_numbers(batch, "slack", ("slack", "slack_link", "bus"))
            update_record(
                records["slack_q"],
                torch.abs(slack_pred - slack_label),
                slack_pred,
                slack_label,
                batch["slack"].batch,
                slack_bus,
                batch_start,
            )

            pq_pred = output["PQ"][:, 1]
            pq_label = batch["PQ"].y[:, 1]
            pq_bus = local_bus_numbers(batch, "PQ", ("PQ", "PQ_link", "bus"))
            constrained_training_points.update(
                matching_training_points(
                    pq_pred,
                    pq_label,
                    batch["PQ"].batch,
                    batch_start,
                )
            )
            update_record(
                records["pq_vm"],
                torch.abs(pq_pred - pq_label),
                pq_pred,
                pq_label,
                batch["PQ"].batch,
                pq_bus,
                batch_start,
            )

    reactive_records = [records["pv_q"], records["slack_q"]]
    reactive_max = max(reactive_records, key=lambda r: r["error"])

    print(f"Checked {len(dataset)} training points.")
    print_record(records["pv_q"])
    print_record(records["slack_q"])
    print_record(records["pq_vm"])
    print()
    print_record({**reactive_max, "name": "reactive overall"})
    print()
    print(
        "Training points with any PQ bus satisfying "
        f"predicted vm >= 0.94 and label vm <= 0.90: "
        f"{len(constrained_training_points)}"
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
