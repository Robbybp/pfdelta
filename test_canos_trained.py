import os
import torch
from core.models.canos_pf import CANOS_PF
from core.datasets.pfdelta_variants import PFDeltaCANOS

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
nparam = sum(p.numel() for p in canos.parameters() if p.requires_grad)
print(f"N. parameters: {nparam}")

modelpath = os.path.join("runs", "canos_task_1_1", "canos_k_steps15_hd128_lr5e-4_task_1_1_260116_121912", "model.pt")
canos_state = torch.load(modelpath, map_location="cpu")
canos.load_state_dict(canos_state)
canos.eval()

data = dataset[0]
for node_type in data.num_node_features.keys():
    print(f"{node_type}: {data.num_node_features[node_type]}")
    print(data[node_type])
    print(data[node_type].x)
    # It appears data[node_type].x needs to be added somewhere...

print(data)
print(canos(data))
