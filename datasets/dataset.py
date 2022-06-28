import os
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from engine.configs import Datasets


class MultipleDomainDataset(object):
    def __init__(self):
        self.Environments = None
        self.input_shape = None

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


@Datasets.register('whole')
class LocalDataset(object):
    def __init__(self, root):
        super(LocalDataset, self).__init__()
        self.data_pkl = self._load_cache(root)

    def _load_cache(self, cache_path):
        if os.path.exists(cache_path):
            print("load cache from {}...".format(cache_path))
            with open(cache_path, "rb") as fin:
                data_pkl = pickle.load(fin)
        else:
            raise NotImplementedError
        return data_pkl

    def process_dataset(self):
        data = self.data_pkl['data']
        labels = self.data_pkl['label']
        domain_idx = self.data_pkl['domain']
        x = torch.tensor(data).float()
        y = torch.tensor(labels).long()
        idx = torch.tensor(domain_idx).long()
        return TensorDataset(x, y, idx)

    def __len__(self):
        return len(self.data_pkl)
