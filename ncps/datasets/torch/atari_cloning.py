from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from ncps.datasets.utils import download_and_unzip


class AtariCloningDataset(Dataset):
    def __init__(self, env_name, split="train", root_dir="."):
        path = Path(root_dir) / "data_atari_seq" / env_name
        if not path.exists():
            print("Downloading data ... ", end="", flush=True)
            download_and_unzip(
                f"https://people.csail.mit.edu/mlechner/datasets/{env_name}.zip"
            )
            print("[done]")

        self.files = list(path.glob(f"{split}_*.npz"))
        if len(self.files) == 0:
            raise RuntimeError("Could not find data")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        x = arr["obs"]
        y = arr["actions"]
        y = torch.from_numpy(y)
        x = np.transpose(x, [0, 3, 1, 2])  # channel first
        x = torch.from_numpy(x.astype(np.float32) / 255)
        return x, y