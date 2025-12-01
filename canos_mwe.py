import os
from core.datasets.opfdata import OPFData
from core.models.canos_opf import CANOS_OPF
dataset = OPFData(
    split="test",
    case_name="pglib_opf_case14_ieee",
    num_groups=1,
    topological_perturbations=True,
    pre_transform="mean_zero_variance_one",
    root=os.path.join("data", "opfdata"),
)
data = dataset[0]
canos = CANOS_OPF(
    dataset=dataset,
    hidden_dim=64,
    include_sent_messages=False,
    k_steps=3,
)
out = canos(data)
print("\nInput data:")
print(data)
print("\nCANOS GNN:")
print(canos)
print("\nOutput data:")
print(out)
