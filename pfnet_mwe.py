#!/usr/bin/env python3
"""
Minimal working example for PFNet (PowerFlowNet) on real PFDelta data.
- Resolves repo_root at runtime so imports work regardless of CWD.
- Loads a small real sample (case14) via PFDeltaPFNet.
- Instantiates PowerFlowNet with appropriate dims.
- Runs a forward pass and prints shapes and simple stats.

Notes:
- This script downloads/uses the PFDelta dataset from Hugging Face via
  torch_geometric's download utilities on first run, storing files under
  datasets/pfdelta-data/.
- Keeps everything on CPU.
"""
import os

from core.datasets.pfdelta_dataset import PFDeltaPFNet
from core.models.powerflownet import PowerFlowNet

# Dataset: real PFDelta case14, PFNet preprocessed representation
ds = PFDeltaPFNet(
    root_dir=os.path.join("data"),
    case_name='case14',
    split='train',
    model='PFNet',
    task=1.1,
    add_bus_type=False,
    transform=None,
    pre_transform=None,
    force_reload=False,
)
print(f"Loaded PFDeltaPFNet: len={len(ds)}  processed_dir={ds.processed_dir}")

data = ds[0]
num_buses = data['bus'].x.size(0)
nfeature_dim = 6
efeature_dim = data['bus','branch','bus'].edge_attr.size(-1)
print(f"bus.x shape={tuple(data['bus'].x.shape)}  edge_attr shape={tuple(data['bus','branch','bus'].edge_attr.shape)}")

# Instantiate untrained model
model = PowerFlowNet(
    nfeature_dim=nfeature_dim,
    efeature_dim=efeature_dim,
    output_dim=6,
    hidden_dim=64,
    n_gnn_layers=3,
    K=4,
    dropout_rate=0.0,
)
# Output is just a matrix. Rows are buses, columns are features
# (voltages, injections, withdrawals?)
out = model(data)
print(f"Output tensor shape={tuple(out.shape)}  dtype={out.dtype}")
print(f"Output stats: min={float(out.min()):.6f} max={float(out.max()):.6f} mean={float(out.mean()):.6f}")
