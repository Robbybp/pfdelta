from core.datasets.pfdelta_dataset import PFDeltaDataset
from core.datasets.pfdelta_variants import PFDeltaCANOS
import torch

from core.models.canos_pf import CANOS_PF
#from core.datasets.opfdata import opfdata_mean0_var1

dataset = PFDeltaCANOS(
    case_name="case118",
    task=1.1,
    root_dir="data",
    model="CANOS",
    add_bus_type=True,
    force_reload=False,
)

hidden_dim = 384
include_sent_messages = True # "self-edges" in inter-layer connections?
k_steps = 15 # Number of "message passing" steps
canos = CANOS_PF(dataset, hidden_dim, include_sent_messages, k_steps)
point = dataset[0]
nparam = sum(p.numel() for p in canos.parameters() if p.requires_grad)
print(f"N. parameters: {nparam}")

nparam = sum(p.numel() for p in canos.encoder.parameters() if p.requires_grad)
print(f"N. encoder parameters: {nparam}")

nparam = sum(p.numel() for p in canos.decoder.parameters() if p.requires_grad)
print(f"N. decoder parameters: {nparam}")
