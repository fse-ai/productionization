import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset


class MLP(nn.Module):
    def __init__(self, inference=False):
        super(MLP, self).__init__()
        self.inference = inference
        self.linear1 = nn.Linear(10, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        if self.inference:
            return f.softmax(self.out(x), dim=0)
        return self.out(x)


class MLPDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path).to_numpy()
        self.length = self.data.shape[0]
        self.n_features = len(np.unique(self.data[:, -1]))

    def __getitem__(self, item):
        data = self.data[item]
        features = torch.tensor(data[:-1]).to(torch.float32)
        label = torch.tensor(data[-1]).to(torch.long)
        return label, features

    def __len__(self):
        return self.length
