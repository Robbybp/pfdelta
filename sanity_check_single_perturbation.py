#!/usr/bin/env python3
"""Apply one hand-picked CANOS perturbation and print PQ voltage predictions."""
import argparse
import os
from pathlib import Path
from typing import Iterable

import torch

from core.datasets.pfdelta_variants import PFDeltaCANOS
from core.models.canos_pf import CANOS_PF


DEFAULT_MODEL = Path(
    "runs/canos_task_1_1/"
    "canos_k_steps15_hd128_lr5e-4_task_1_1_260527_121647/model.pt"
)
TRAINING_POINT_INDEX = 4
PERTURBED_BUS = 6
PERTURBED_VM = 0.970353


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity check one adversarial perturbation on CANOS."
    )
    parser.add_argument(
        "model_file",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL,
        help="CANOS .pt weights file",
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


def load_model(dataset: PFDeltaCANOS, model_path: Path) -> CANOS_PF:
    model = CANOS_PF(
        dataset=dataset,
        hidden_dim=128,
        include_sent_messages=True,
        k_steps=15,
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def apply_vm_perturbation(data, bus_number: int, vm: float) -> None:
    bus_idx = bus_number - 1
    data["bus"].x[bus_idx, 1] = vm

    import pdb; pdb.set_trace()
    for node_type, edge_type in [
        ("PQ", ("PQ", "PQ_link", "bus")),
        ("PV", ("PV", "PV_link", "bus")),
        ("slack", ("slack", "slack_link", "bus")),
    ]:
        bus_indices = data[edge_type].edge_index[1]
        matches = (bus_indices == bus_idx).nonzero(as_tuple=True)[0]
        if matches.numel() > 0:
            data[node_type].x[matches[0], 1] = vm
            return

    raise ValueError(f"Bus {bus_number} was not found in PQ/PV/slack links")


def main(argv: Iterable[str]) -> None:
    args = parse_args(argv)
    dataset = load_dataset()
    data = dataset[TRAINING_POINT_INDEX].clone()
    original_vm = float(data["bus"].x[PERTURBED_BUS - 1, 1])
    apply_vm_perturbation(data, PERTURBED_BUS, 1.0)

    model = load_model(dataset, args.model_file)
    with torch.no_grad():
        output = model(data)

    pq_bus_indices = data["PQ", "PQ_link", "bus"].edge_index[1]
    pq_vms = output["PQ"][:, 1]

    print(
        f"training_point_index={TRAINING_POINT_INDEX}  "
        f"vm[{PERTURBED_BUS}] {original_vm:.6f} -> {PERTURBED_VM:.6f}"
    )
    print("PQ bus predicted voltage magnitudes:")
    for bus_idx, vm in zip(pq_bus_indices.tolist(), pq_vms.tolist()):
        print(f"  bus {bus_idx + 1:2d}: {vm:.6f}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
