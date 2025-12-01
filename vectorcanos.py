import os
import torch
from torch import nn
from core.datasets.opfdata import OPFData
from core.models.canos_opf import CANOS_OPF


class VectorCanos(nn.Module):
    """
    The input vector contains:
      - bus features
      - generator features
      - load features
      - shunt features
      - (bus, ac_line, bus).edge_attr
      - (bus, transformer, bus).edge_attr

    The output vector contains:
      - out["bus"], out["generator"], out["edge_preds"]
    """

    def __init__(self, model: nn.Module, template):
        super().__init__()
        self.model = model
        self.template = template

        # Shapes used for slicing/unflattening
        self.bus_x_shape = tuple(template["bus"]["x"].shape)
        self.gen_x_shape = tuple(template["generator"]["x"].shape)
        self.load_x_shape = tuple(template["load"]["x"].shape)
        self.shunt_x_shape = tuple(template["shunt"]["x"].shape)
        self.line_ea_shape = tuple(template["bus", "ac_line", "bus"]["edge_attr"].shape)
        self.tr_ea_shape = tuple(template["bus", "transformer", "bus"]["edge_attr"].shape)

        self.sizes = [
            self.bus_x_shape[0] * self.bus_x_shape[1],
            self.gen_x_shape[0] * self.gen_x_shape[1],
            self.load_x_shape[0] * self.load_x_shape[1],
            self.shunt_x_shape[0] * self.shunt_x_shape[1],
            self.line_ea_shape[0] * self.line_ea_shape[1],
            self.tr_ea_shape[0] * self.tr_ea_shape[1],
        ]
        self.input_dim = sum(self.sizes)

        # Output shapes determined via a dry run
        out = model(template)
        self.out_bus_shape = tuple(out["bus"].shape)
        self.out_gen_shape = tuple(out["generator"].shape)
        self.out_edge_shape = tuple(out["edge_preds"].shape)
        self.output_dim = (
            self.out_bus_shape[0] * self.out_bus_shape[1]
            + self.out_gen_shape[0] * self.out_gen_shape[1]
            + self.out_edge_shape[0] * self.out_edge_shape[1]
        )

    def flatten_input(self, data) -> torch.Tensor:
        parts = [
            data["bus"]["x"].reshape(-1),
            data["generator"]["x"].reshape(-1),
            data["load"]["x"].reshape(-1),
            data["shunt"]["x"].reshape(-1),
            data["bus", "ac_line", "bus"]["edge_attr"].reshape(-1),
            data["bus", "transformer", "bus"]["edge_attr"].reshape(-1),
        ]
        return torch.cat(parts, dim=0)

    def unflatten_input(self, x_flat: torch.Tensor):
        x_flat = x_flat.to(dtype=torch.float32)
        chunks = torch.split(x_flat, self.sizes)

        data = self.template.clone()
        b0, b1 = self.bus_x_shape
        g0, g1 = self.gen_x_shape
        l0, l1 = self.load_x_shape
        s0, s1 = self.shunt_x_shape
        e0, e1 = self.line_ea_shape
        t0, t1 = self.tr_ea_shape

        data["bus"]["x"] = chunks[0].view(b0, b1)
        data["generator"]["x"] = chunks[1].view(g0, g1)
        data["load"]["x"] = chunks[2].view(l0, l1)
        data["shunt"]["x"] = chunks[3].view(s0, s1)
        data["bus", "ac_line", "bus"]["edge_attr"] = chunks[4].view(e0, e1)
        data["bus", "transformer", "bus"]["edge_attr"] = chunks[5].view(t0, t1)

        # Keep limits from template; set branch_vals = edge_attr
        data["bus"]["v_lims"] = self.template["bus"]["v_lims"].clone()
        data["generator"]["p_lims"] = self.template["generator"]["p_lims"].clone()
        data["generator"]["q_lims"] = self.template["generator"]["q_lims"].clone()
        data["bus", "ac_line", "bus"]["branch_vals"] = data["bus", "ac_line", "bus"]["edge_attr"]
        data["bus", "transformer", "bus"]["branch_vals"] = data["bus", "transformer", "bus"]["edge_attr"]
        return data

    def flatten_output(self, out: dict) -> torch.Tensor:
        parts = [
            out["bus"].reshape(-1),
            out["generator"].reshape(-1),
            out["edge_preds"].reshape(-1),
        ]
        return torch.cat(parts, dim=0)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        data = self.unflatten_input(x_flat)
        out = self.model(data)
        return self.flatten_output(out)


dataset = OPFData(
    split="train",
    case_name="pglib_opf_case14_ieee",
    num_groups=1,
    topological_perturbations=True,
    pre_transform="mean_zero_variance_one",
    root=os.path.join("data", "opfdata"),
)
sample = dataset[0]
canos = CANOS_OPF(dataset=dataset, hidden_dim=64, include_sent_messages=False, k_steps=3)
wrapper = VectorCanos(canos, sample)
x_flat = wrapper.flatten_input(sample)
y_flat = wrapper(x_flat)
print(f"input_dim={wrapper.input_dim}  output_dim={wrapper.output_dim}")
torch.save(wrapper, "vector-canos.pt")
