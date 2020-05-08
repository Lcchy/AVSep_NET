from source.parameters import *
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
import torch


class AVDataset(Dataset):

    def __init__(self, path_to_data):
        self.av_data = pd.read_csv(path_to_data, header=2, quotechar='"', skipinitialspace=True)

    def __len__(self):
        return len(self.av_data)

    def __getitem__(self, index):
        return torch.load(index)


def test():
    test_data = AVDataset(Path("../dataset/balanced_train_segments.csv"))
