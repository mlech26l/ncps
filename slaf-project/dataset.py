import pandas as pd

import torch
from torch.utils.data import Dataset


class ESTDataset(Dataset):
    """EST hourly dataset."""
    STATIONS = [
        "AEP",
        "COMED",
        "DAYTON",
        "DEOK",
        "DOM",
        "DUQ",
        "EKPC",
        "FE",
        "NI",
        "PJME",
        "PJMW",
        "PJM_Load"
    ]

    def __init__(self, dataset_file, sequence_length=10, station="AEP"):
        """
        Arguments:
            dataset_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sequence_length = sequence_length
        if station not in self.STATIONS and \
            station.upper() not in self.STATIONS:
            raise ValueError(f"{station} is not found in the dataset.")
        data = pd.read_parquet(dataset_file)[[station]]
        data.index.names = ["datetime"]
        data.rename(columns={station: "value"}, inplace=True)
        data.dropna(inplace=True)

        self.data = data.values  # Convert pandas DataFrame to numpy array

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]
        sequence = torch.tensor(sequence, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return sequence, target
