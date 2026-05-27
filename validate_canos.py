"""validate_canos.py

This script evaluates loss on the trained CANOS model.
Note that this process gets killed (OOM?) if I try to evaluate loss for every
point individually.

Here is the output of this script running on the first 1000 test points:
```
Dataset loss summary:
MSE          mean=7.202885  max=58.739437  std=7.956676
Constraint   mean=3.127100  max=5.625901  std=0.929861
PB           mean=0.694869  max=1.839052  std=0.408954
```
And on the first 1000 training points:
```
Dataset loss summary:
MSE          mean=0.000856  max=0.021331  std=0.000719
Constraint   mean=1.405252  max=2.275985  std=0.166036
PB           mean=0.038077  max=0.169933  std=0.006189
```

"""
import os
import torch
from torch_geometric.loader import DataLoader
from core.models.canos_pf import CANOS_PF
from core.datasets.pfdelta_variants import PFDeltaCANOS
from core.utils.pf_losses_utils import (
    CANOS_PF_MSE,
    constraint_violations_loss_pf,
    PowerBalanceLoss,
)

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
include_sent_messages = True # "self-edges" in inter-layer connections?
k_steps = 15 # Number of "message passing" steps
canos = CANOS_PF(dataset, hidden_dim, include_sent_messages, k_steps)
nparam = sum(p.numel() for p in canos.parameters() if p.requires_grad)
print(f"N. parameters: {nparam}")

# This is the old model (on Selene, I believe)
#modelpath = os.path.join("runs", "canos_task_1_1", "canos_k_steps15_hd128_lr5e-4_task_1_1_260116_133938", "model.pt")
# This is the model after the PFDelta bugfix, on Selene
modelpath = os.path.join("runs", "canos_task_1_1", "canos_k_steps15_hd128_lr5e-4_task_1_1_260527_121647", "model.pt")
canos_state = torch.load(modelpath, map_location="cpu")
canos.load_state_dict(canos_state)
canos.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
canos.to(device)

batch_size = 512


def evaluate_loader(loader, device):
    pbl = PowerBalanceLoss("CANOS")
    mse_vals, con_vals, pb_vals = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"Batch {i}")
            batch = batch.to(device)
            output = canos(batch)
            mse_loss = CANOS_PF_MSE()(output, batch)
            con_loss = constraint_violations_loss_pf()(output, batch)
            pb_loss = pbl(output, batch)
            mse_vals.append(float(mse_loss))
            con_vals.append(float(con_loss))
            pb_vals.append(float(pb_loss))
    return mse_vals, con_vals, pb_vals


def summarize(name, vals):
    t = torch.tensor(vals)
    mean = t.mean().item()
    maxv = t.max().item()
    std = t.std(unbiased=False).item()
    print(f"{name:12s} mean={mean:.6f}  max={maxv:.6f}  std={std:.6f}")


print("\nTrain loss summary:")
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
train_mse, train_con, train_pb = evaluate_loader(train_loader, device)
summarize("MSE", train_mse)
summarize("Constraint", train_con)
summarize("PB", train_pb)

print("\nTest loss summary:")
test_dataset = PFDeltaCANOS(**{**dataset_inputs, "split": "test"})
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
test_mse, test_con, test_pb = evaluate_loader(test_loader, device)
summarize("MSE", test_mse)
summarize("Constraint", test_con)
summarize("PB", test_pb)

print("\nTrain loss summary:")
summarize("MSE", train_mse)
summarize("Constraint", train_con)
summarize("PB", train_pb)

print("\nTest loss summary:")
summarize("MSE", test_mse)
summarize("Constraint", test_con)
summarize("PB", test_pb)
