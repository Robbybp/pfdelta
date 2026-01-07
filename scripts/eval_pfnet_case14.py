#!/usr/bin/env python3
"""
Evaluate a trained PFNet model on 100 validation samples for the case14 PFDelta task.

Usage:
  python scripts/eval_pfnet_case14.py \
    --run runs/pfnet_case14_quick/pfnet_case14_quick_260107_133423 \
    --device cpu \
    --samples 100
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from torch_geometric.loader import DataLoader

# Make repository importable when running from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils.main_utils import load_registry
from core.models.powerflownet import PowerFlowNet
from core.datasets.pfdelta_dataset import PFDeltaPFNet


def masked_mse(pred, target, mask):
    # mask is 0/1; avoid division by zero
    diff = (pred - target) * mask
    denom = mask.sum()
    if denom == 0:
        return torch.tensor(0.0, device=pred.device)
    return (diff.pow(2).sum() / denom).detach()


def evaluate(run_dir, device, num_samples):
    # Load run config
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    ds_cfg = cfg["dataset"]["datasets"][0]

    load_registry()  # ensure PowerFlowNet is registered if registry is used elsewhere

    # Build model and load weights
    model = PowerFlowNet(
        nfeature_dim=model_cfg["nfeature_dim"],
        efeature_dim=model_cfg["efeature_dim"],
        output_dim=model_cfg["output_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_gnn_layers=model_cfg["n_gnn_layers"],
        K=model_cfg["K"],
        dropout_rate=model_cfg["dropout_rate"],
    ).to(device)
    state = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Dataset and loader
    dataset = PFDeltaPFNet(
        root_dir=ds_cfg["root_dir"],
        case_name=ds_cfg["case_name"],
        split="val",
        model=ds_cfg["model"],
        task=ds_cfg["task"],
        transform=ds_cfg.get("transform", None),
        add_bus_type=ds_cfg.get("add_bus_type", False),
    )
    if num_samples > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    total_masked_mse = 0.0
    total_mse = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch["bus"].y
            mask = batch["bus"].x[:, -model_cfg["output_dim"] :]

            total_mse += torch.mean((pred - target).pow(2)).item()
            total_masked_mse += masked_mse(pred, target, mask).item()
            count += 1

    print(f"Evaluated batches: {count}")
    print(f"Mean MSE over batches: {total_mse / count:.6f}")
    print(f"Mean Masked MSE over batches: {total_masked_mse / count:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to run directory containing model.pt and config.yaml")
    parser.add_argument("--device", default="cpu", help="Device to run eval on (cpu or cuda)")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to evaluate (use -1 for all)")
    args = parser.parse_args()

    device = torch.device(args.device)
    evaluate(args.run, device, args.samples)


if __name__ == "__main__":
    main()
