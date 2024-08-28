from typing import Tuple, Any

import numpy as np
import pandas as pd
from torch.utils import data
from PIL import Image
import torch
import os
import random

from torch.utils.data import Dataset


class DiscreteDataset(Dataset):

    def __init__(self, train_file, test_file, n_domains, domains):
        subsets = ['train', 'test']
        subset_files = [train_file, test_file]

        self.mode = 'train'

        self.data = {}

        for i, s in enumerate(subsets):
            x, y = self.load_subset(domains, subset_files[i])

            x = np.expand_dims(x, 0)

            x = torch.tensor(x, dtype=torch.float32).squeeze()
            y = torch.tensor(y, dtype=torch.long)

            self.data[s] = (x, y)

    def load_subset(self, domains, path):
        frames = []
        y = []
        for d in domains:
            try:
                df = pd.read_csv(os.path.join(path, '{}.csv'.format(d)), header=0)
                frames.append(df)
                labels = [int(d) for i in range(len(df))]
                y = y + labels
            except pd.errors.EmptyDataError:
                continue

        df = pd.concat(frames)
        x = df.values

        return x, y

    def __len__(self):
        return len(self.data['train'][1])

    def __getitem__(self, idx):
        return self.data['train'][0][idx], self.data['train'][1][idx]


def get_loader(dataset_path, batch_size=16, mode='train', num_workers=1, n_domains=5, domains=None):
    """Build and return a data loader."""
    dataset = DiscreteDataset(os.path.join(dataset_path, 'train'), os.path.join(dataset_path, 'test'), n_domains, domains)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader
