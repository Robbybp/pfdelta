import os
import torch
from torch import nn
from core.models.canos_pf import CANOS_PF
from core.datasets.pfdelta_variants import PFDeltaCANOS
from vectorcanos import VectorCanos

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
wrapper = VectorCanos(
    canos,
    sample,
)

x = dataset[101]
y = canos(x)
x_flat = wrapper.flatten_input(x)
y_flat = wrapper(x_flat)
y_unflat = wrapper.unflatten_output(y_flat)
print("Errors in y")
print("-----------")
for k in y_unflat.keys():
    print(k)
    diff = y[k] - y_unflat[k]
    print(diff)
    print()
