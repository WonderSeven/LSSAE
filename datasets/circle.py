# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch

from torch.utils.data import TensorDataset

from datasets.dataset import MultipleDomainDataset
from engine.configs import Datasets


class MultipleEnvironmentCircle(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape, num_classes):
        super(MultipleEnvironmentCircle, self).__init__()
        self.Environments = environments
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.normalize = False  # Don't do normalization for DG

        if root is None:
            raise ValueError('Data directory not specified!')

        self.data_pkl = self._load_cache(root)

        if self.normalize:
            data = self.data_pkl['data']
            data_mean = data.mean(0, keepdims=True)
            data_std = data.std(0, keepdims=True)
            self.data_pkl['data'] = (data - data_mean) / data_std  # normalize the raw data

        self.datasets = []
        for domain_id in range(len(environments)):
            idx = self.data_pkl['domain'] == domain_id
            data = self.data_pkl['data'][idx].astype(np.float32)
            labels = self.data_pkl['label'][idx].astype(np.int64)

            self.datasets.append(dataset_transform(data, labels))

    def _load_cache(self, cache_path):
        if os.path.exists(cache_path):
            print("load cache from {}...".format(cache_path))
            with open(cache_path, "rb") as fin:
                data_pkl = pickle.load(fin)
        else:
            raise NotImplementedError
        return data_pkl


@Datasets.register('toycircle')
class ToyCircle(MultipleEnvironmentCircle):
    """
    source domains: 0~14
    intermedia domains: 15~19
    target domains: 20~29
    """

    def __init__(self, root, input_shape, num_classes):
        num_domains = 30
        environments = list(np.arange(num_domains))
        self.normalize_domain = False
        super(ToyCircle, self).__init__(root, environments, self.process_dataset, input_shape, num_classes)

    def process_dataset(self, data, labels):
        x = torch.tensor(data).float()
        y = torch.tensor(labels).long()
        if self.normalize_domain:
            mu, sigma = x.mean(0, keepdims=True), x.std(0, keepdims=True)
            x = (x - mu) / sigma
        return TensorDataset(x, y)