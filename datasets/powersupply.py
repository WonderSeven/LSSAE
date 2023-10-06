# -*- coding: utf-8 -*-
import numpy as np

import torch
from torch.utils.data import TensorDataset

from datasets.dataset import MultipleDomainDataset
from engine.configs import Datasets


class MultipleEnvironmentPowerSupply(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape, num_classes):
        super().__init__()
        num_domains = 30
        # self.Environments = environments
        self.Environments = np.arange(num_domains)
        self.root = root
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.normalize = True

        self.drift_times = [17, 47, 76]
        self.num_data = 29928  # original data number

        self.X, self.Y = self.load_data()
        # normalize
        if self.normalize:
            self.X = self.X / np.max(self.X, axis=0)

        self.datasets = []
        for i in range(len(environments)):
            images = self.X[i::len(environments)]
            labels = self.Y[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels))


    def load_data(self):
        X = []
        Y = []
        with open(self.root) as file:
            i = 0
            for line in file:
                fields = line.strip().split(',')
                label = 1 if int(fields[2]) < 12 else 0

                cur_x = np.array([float(fields[0]), float(fields[1])], np.float32)

                X.append(cur_x)
                Y.append(label)
                i += 1
        assert len(X) == self.num_data
        return np.array(X), np.array(Y, np.int64)


@Datasets.register('powersupply')
class PowerSupply(MultipleEnvironmentPowerSupply):
    def __init__(self, root, input_shape, num_classes):
        num_domains = 30
        environments = list(np.arange(num_domains))

        super(PowerSupply, self).__init__(root, environments, self.process_dataset, input_shape, num_classes)

    def process_dataset(self, data, labels):
        x = torch.tensor(data).float()
        y = torch.tensor(labels).long()
        return TensorDataset(x, y)
