#!/usr/bin/env python3
"""
PFNet vector wrapper
- Builds metadata from one PFDeltaPFNet sample.
- Defines a wrapper that accepts a 1-D input vector (node features and edge attrs)
  and outputs a 1-D vector (flattened model outputs).
"""
import torch
from torch import nn

from core.datasets.pfdelta_variants import PFDeltaPFNet
from core.models.powerflownet import PowerFlowNet


class PFNetVectorWrapper(nn.Module):
    """Wrap PowerFlowNet to expose flat 1-D input/output vectors.

    Input vector layout (per template sample):
      - bus.x[:, 4:10] (the 6 PFNet input features per bus; bus one-hot is in 0:4)
      - (bus, branch, bus).edge_attr (E, fe)

    Static graph pieces retained from template:
      - edge_index (bus, branch, bus)
      - bus.x one-hot type encodings (cols 0:4)
      - pred_mask (cols 10:16) implicitly kept as part of bus.x

    Output vector layout:
      - Flattened PowerFlowNet output tensor (num_buses * 6).
    """

    def __init__(self, model: nn.Module, template):
        super().__init__()
        self.model = model
        self.template = template

        bx = template['bus'].x
        self.num_buses = bx.size(0)
        # PFNet bus.x has 16 dims: 4 one-hot, 6 input features, 6 pred mask
        self.bus_feat_start = 4
        self.bus_feat_len = 6
        assert bx.size(1) >= self.bus_feat_start + self.bus_feat_len, "Unexpected PFNet bus.x layout"

        self.edge_attr_shape = tuple(template['bus','branch','bus'].edge_attr.shape)
        self.efeat_len = self.edge_attr_shape[1]
        self.num_edges = self.edge_attr_shape[0]

        self.input_dim = self.num_buses * self.bus_feat_len + self.num_edges * self.efeat_len

        # Determine output size via dry run
        with torch.no_grad():
            out = model(template)
        self.out_shape = tuple(out.shape)
        self.output_dim = out.numel()

    def flatten_input_from_data(self, data) -> torch.Tensor:
        bx = data['bus'].x[:, self.bus_feat_start:self.bus_feat_start+self.bus_feat_len]
        ea = data['bus','branch','bus'].edge_attr
        return torch.cat([bx.reshape(-1), ea.reshape(-1)], dim=0)

    def unflatten_to_data(self, vec: torch.Tensor):
        vec = vec.to(dtype=torch.float32)
        n_bus = self.num_buses
        n_e = self.num_edges
        fb = self.bus_feat_len * n_bus
        fe = self.efeat_len * n_e
        assert vec.numel() == fb + fe, f"Expected vec length {fb+fe}, got {vec.numel()}"

        data = self.template.clone()
        # restore bus.x 6 input features; keep one-hot and pred mask from template
        bus_feats = vec[:fb].view(n_bus, self.bus_feat_len)
        data['bus'].x[:, self.bus_feat_start:self.bus_feat_start+self.bus_feat_len] = bus_feats

        # restore edge_attr
        edge_feats = vec[fb:fb+fe].view(n_e, self.efeat_len)
        data['bus','branch','bus'].edge_attr = edge_feats
        return data

    def flatten_output(self, out: torch.Tensor) -> torch.Tensor:
        return out.reshape(-1)

    def forward(self, vec_in: torch.Tensor) -> torch.Tensor:
        # dtype/device alignment
        p = next(self.model.parameters())
        vec_in = vec_in.to(dtype=p.dtype, device=p.device)
        data = self.unflatten_to_data(vec_in)
        with torch.no_grad():
            out = self.model(data)
        return self.flatten_output(out)


torch.manual_seed(0)

# Load real PFDelta sample
ds = PFDeltaPFNet(
    root_dir="data",
    case_name='case14',
    split='train',
    model='PFNet',
    task=1.1,
    add_bus_type=False,
    transform=None,
    pre_transform=None,
    force_reload=False,
)
data = ds[0]

n_buses = data['bus'].x.size(0)
eattr = data['bus','branch','bus'].edge_attr
fe = eattr.size(-1)

# Model
model = PowerFlowNet(
    nfeature_dim=6,
    efeature_dim=fe,
    output_dim=6,
    hidden_dim=64,
    n_gnn_layers=3,
    K=4,
    dropout_rate=0.0,
)

wrapper = PFNetVectorWrapper(model, data)

# Build input vector from the real sample
x_vec = wrapper.flatten_input_from_data(data)
y_vec = wrapper(x_vec)

print(f"input_dim={wrapper.input_dim} output_dim={wrapper.output_dim}")
print(f"x_vec.shape={tuple(x_vec.shape)} y_vec.shape={tuple(y_vec.shape)}")
print(f"x[min,max]=({float(x_vec.min())}, {float(x_vec.max())})  y[min,max,mean]=({float(y_vec.min())}, {float(y_vec.max())}, {float(y_vec.mean())})")

torch.save(wrapper, "vector-pfnet.pt")
