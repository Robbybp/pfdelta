import os
import copy

import torch
from core.datasets.pfdelta_dataset import PFDeltaDataset
from core.models.canos_pf import CANOS_PF
from core.datasets.opfdata import opfdata_mean0_var1
from core.utils.registry import registry
from core.datasets.pfdelta_variants import PFDeltaCANOS

#class PFDeltaCANOS(PFDeltaDataset):
#    """
#    Simplified PFDelta dataset variant for CANOS model.
#    
#    Prunes the heterogeneous graph to include only the node types
#    required by CANOS: bus, PV, PQ, and slack.
#    """
#
#    def __init__(
#        self,
#        root_dir="data",
#        case_name="",
#        split="train",
#        model="CANOS",
#        task=1.1,
#        add_bus_type=True,
#        force_reload=False,
#    ):
#        # Initialize parent class with CANOS defaults
#        super().__init__(
#            root_dir=root_dir,
#            case_name=case_name,
#            split=split,
#            model=model,
#            task=task,
#            add_bus_type=add_bus_type,
#            force_reload=force_reload,
#        )
#
#    def build_heterodata(self, pm_case: dict, is_cpf_sample: bool = False):
#        """
#        Build a CANOS-compatible HeteroData graph with pruned node types.
#        
#        Parameters
#        ----------
#        pm_case : dict
#            PowerModels.jl case dictionary with bus, branch, gen, and load data
#        is_cpf_sample : bool
#            Whether this is a continuation power flow sample
#            
#        Returns
#        -------
#        data : HeteroData
#            Processed graph with only bus, PV, PQ, and slack nodes
#        """
#        # Build the full heterogeneous graph using parent method
#        data = super().build_heterodata(pm_case, is_cpf_sample=is_cpf_sample)
#
#        # Prune to keep only CANOS-required node types
#        keep_nodes = {"bus", "PV", "PQ", "slack"}
#
#        # Remove unwanted node types
#        for node_type in list(data.node_types):
#            if node_type not in keep_nodes:
#                del data[node_type]
#
#        # Remove edges connected to deleted node types
#        for edge_type in list(data.edge_types):
#            src, _, dst = edge_type
#            if src not in keep_nodes or dst not in keep_nodes:
#                del data[edge_type]
#
#        return data

#dataset = PFDeltaCANOS(
#    case_name="case14",
#    task=1.1,
#    root_dir=os.path.join("data", "pfdelta_data"),
#    model="CANOS",
#    add_bus_type=True,
#    force_reload=False,
#)

dataset_name = "pfdeltaCANOS"
print(f"Dataset name:  {dataset_name}")
dataset_inputs = dict(
    add_bus_type=True,
    case_name="case14",
    model="CANOS",
    root_dir=os.path.join("data", "pfdelta_data"),
    split="train",
    task="1.1",
)
dataset = PFDeltaCANOS(**dataset_inputs)

hidden_dim = 128
include_sent_messages = False # "self-edges" in inter-layer connections?
k_steps = 15 # Number of "message passing" steps
canos = CANOS_PF(dataset, hidden_dim, include_sent_messages, k_steps)
nparam = sum(p.numel() for p in canos.parameters if p.requires_grad)
print(f"N. parameters: {nparam}")

modelpath = os.path.join("runs", "canos_task_1_1", "canos_k_steps15_hd128_lr5e-4_task_1_1_260116_121912", "model.pt")
canos_state = torch.load(modelpath, map_location="cpu")
canos.load_state_dict(canos_state)
canos.eval()

# CANOS expects HeterData with:
# - bus
# - gen
# - load
# - a bunch of edge keys

#x = {
#    "bus": torch.zeros(2),
#    "gen": torch.tensor([]),
#    "load": torch.tensor([]),
#}

data = dataset[0]
for node_type in data.num_node_features.keys():
    print(f"{node_type}: {data.num_node_features[node_type]}")
    print(data[node_type])
    print(data[node_type].x)
    # It appears data[node_type].x needs to be added somewhere...

#projected_nodes = {
#    node_type: canos.encoder.node_projections[node_type](data[node_type].x)
#    for node_type in data.num_node_features.keys()
#}

#stats = dict(mean=0.0, std=1.0)
#newdata = opfdata_mean0_var1(stats, data)
