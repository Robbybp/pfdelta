import os
import torch
from core.models.canos_pf import CANOS_PF
from core.datasets.pfdelta_variants import PFDeltaCANOS
from vectorcanos import VectorCanos

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

modelpath = os.path.join("runs", "canos_task_1_1", "canos_k_steps15_hd128_lr5e-4_task_1_1_260116_133938", "model.pt")
canos_state = torch.load(modelpath, map_location="cpu")
canos.load_state_dict(canos_state)
canos.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"

x0 = dataset[0]
x0.to(device)
canos.to(device)
for node_type in x0.num_node_features.keys():
    print(f"{node_type}: {x0.num_node_features[node_type]}")
    print(x0[node_type])
    print(x0[node_type].x)
    # It appears x0[node_type].x needs to be added somewhere...

print(x0)
y = canos(x0)
print(y)

vcanos = VectorCanos(canos, x0)
torch.save(vcanos, "vectorcanos-trained.pt")
print("Saved vectorized wrapper to vectorcanos-trained.pt")

x_flat = vcanos.flatten_input(x0)
y_flat = vcanos(x_flat)

# Quick loss check on a single sample
from core.utils.pf_losses_utils import CANOS_PF_MSE, constraint_violations_loss_pf

#with torch.no_grad():
#    # I am simulating the case where this x comes from another optimization solve.
#    # It will be passed through the unflatten_input function, which deletes keys we
#    # don't need.
#    x = vcanos.unflatten_input(x_flat)
#    y = vcanos.unflatten_output(y_flat)
#    mse_loss = CANOS_PF_MSE()(y, x0)
#    constraint_loss = constraint_violations_loss_pf()(y, x)
#    combined_loss = mse_loss + 0.1 * constraint_loss
#    print(
#        f"mse_loss={mse_loss.item():.6f}  "
#        f"constraint_loss={constraint_loss.item():.6f}  "
#        f"combined(λ=0.1)={combined_loss.item():.6f}"
#    )
