import os
from core.datasets.pfdelta_dataset import PFDeltaDataset

dataset = PFDeltaDataset(
    case_name="case14",
    task=1.1,
    root_dir=os.path.join("data", "pfdelta_data"),
)
