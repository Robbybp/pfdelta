import os
import torch
from torch import nn
from core.models.canos_pf import CANOS_PF
from core.datasets.pfdelta_variants import PFDeltaCANOS
from torch_geometric.data import HeteroData


class VectorCanos(nn.Module):
    """
    The input vector contains:
      - bus features
      - PV-bus features
      - PQ-bus features
      - slack bus features
      - (bus, branch, bus).edge_attr

    The output vector contains:
      - out["bus"], out["pq"], out["pv"], out["slack"], out["edge_preds"]

    """

    def __init__(self, model: nn.Module, template):
        super().__init__()
        self.model = model
        self.template = template

        self.node_input_keys = ["bus", "PQ", "PV", "slack"]
        self.edge_input_keys = [("bus", "branch", "bus")]

        # Shapes used for slicing/unflattening
        node_input_shapes = [tuple(template[k]["x"].shape) for k in self.node_input_keys]
        edge_input_shapes = [tuple(template[k]["edge_attr"].shape) for k in self.edge_input_keys]
        self.input_shapes = node_input_shapes + edge_input_shapes
        self.input_sizes = [s[0] * s[1] for s in self.input_shapes]
        self.input_dim = sum(self.input_sizes)

        # Output shapes determined via a dry run
        out = model(template)
        self.output_keys = ["bus", "PQ", "PV", "slack", "edge_preds"]
        self.output_shapes = [tuple(out[k].shape) for k in self.output_keys]
        self.output_sizes = [s[0] * s[1] for s in self.output_shapes]
        self.output_dim = sum(self.output_sizes)

    def flatten_input(self, data) -> torch.Tensor:
        node_inputs = [data[k]["x"].reshape(-1) for k in self.node_input_keys]
        edge_inputs = [data[k]["edge_attr"].reshape(-1) for k in self.edge_input_keys]
        parts = node_inputs + edge_inputs
        return torch.cat(parts, dim=0)

    def unflatten_input(self, x_flat: torch.Tensor):
        x_flat = x_flat.to(dtype=torch.float32)
        chunks = torch.split(x_flat, self.input_sizes)

        data = self.template.clone()
        node_keys = [(k, "x") for k in self.node_input_keys]
        edge_keys = [(k, "edge_attr") for k in self.edge_input_keys]
        keys = node_keys + edge_keys
        for i, k in enumerate(keys):
            data[k[0]][k[1]] = chunks[i].view(*self.input_shapes[i])

        # "Other keys" that we preserve from template
        # ... not that these are used by CANOS... maybe used in the loss?
        # keys_to_preserve = [
        #    ("bus", "limits"),
        #    ("pv", "generation"),
        #    ("pv", "demand"),
        #    ("slack", "generation"),
        #    ("slack", "demand"),
        #    (("bus", "branch", "bus"), "edge_index"),
        #    (("bus", "branch", "bus"), "edge_label"),
        #    (("bus", "branch", "bus"), "edge_limits"),
        # ]
        # for k in keys_to_preserve:
        #    data[k[0]][k[1]] = self.template[k[0]][k[1]].clone()
        return data

    def flatten_output(self, out: dict) -> torch.Tensor:
        parts = [out[k].reshape(-1) for k in self.output_keys]
        return torch.cat(parts, dim=0)

    def unflatten_output(self, y_flat):
        chunks = torch.split(y_flat, self.output_sizes)
        y = dict()
        for i, k in enumerate(self.output_keys):
            y[k] = chunks[i].view(*self.output_shapes[i])
        return y

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        data = self.unflatten_input(x_flat)
        out = self.model(data)
        return self.flatten_output(out)


torch.manual_seed(48)
dataset = PFDeltaCANOS(
    add_bus_type=True,
    case_name="case14",
    model="CANOS",
    root_dir=os.path.join("data", "pfdelta_data"),
    split="train",
    task="1.1",
)
sample = dataset[0]
hidden_dim = 128
include_sent_messages = False
k_steps = 15
canos = CANOS_PF(
    dataset=dataset,
    hidden_dim=hidden_dim,
    include_sent_messages=include_sent_messages,
    k_steps=k_steps,
)
wrapper = VectorCanos(canos, sample)
x_flat = wrapper.flatten_input(sample)
y_flat = wrapper(x_flat)
print(f"input_dim={wrapper.input_dim}  output_dim={wrapper.output_dim}")
torch.save(wrapper, "vector-canos.pt")
