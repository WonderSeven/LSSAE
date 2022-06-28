# -*- coding: utf-8 -*-
import os
import pdb
import math
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from datasets.dataset import MultipleDomainDataset
from engine.configs import Datasets


def Gray_loader(path):
    return Image.open(path).convert('L')


@Datasets.register('portraits')
class MultipleEnvironmentPortraits(MultipleDomainDataset):
    def __init__(self, root, input_shape, num_classes, dataset_transform=None):
        super().__init__()
        num_domains = 34
        # self.Environments = environments
        self.Environments = np.arange(num_domains)
        self.root = root
        self.input_shape = input_shape
        self.num_classes = num_classes

        if dataset_transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.input_shape[-2:]),
                # transforms.RandomResizedCrop(self.input_shape[-2:]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = dataset_transform

        if root is None:
            raise ValueError('Data directory not specified!')

        self.female_dict, self.male_dict, common_keys = self.load_data()

        self.datasets = []

        pre_idx = 0
        for domain_idx in range(num_domains):
            domain_keys = common_keys[pre_idx: pre_idx+3]
            pre_idx += 3
            female_name_list, male_name_list = [], []
            [female_name_list.extend(self.female_dict[year_key]) for year_key in domain_keys]
            [male_name_list.extend(self.male_dict[year_key]) for year_key in domain_keys]
            female_name_list = [os.path.join('F', item) for item in female_name_list]
            male_name_list = [os.path.join('M', item) for item in male_name_list]
            label_list = np.zeros([len(female_name_list)+len(male_name_list)], dtype=np.int64)
            label_list[: len(female_name_list)] = 1
            image_name_list = female_name_list + male_name_list
            # print('Domain idx: {}, Image num:{}, keys:{}'.format(domain_idx, len(image_name_list), domain_keys))
            self.datasets.append(SubPortraits(root, image_name_list, label_list, self.transform))

    def load_data(self):
        female_dict = self.load_name_dict(os.path.join(self.root, 'F'))
        male_dict = self.load_name_dict(os.path.join(self.root, 'M'))
        female_keys = set(female_dict.keys())
        male_keys = set(male_dict.keys())
        common_keys = list(female_keys.intersection(male_keys))[1:]  # 103
        common_keys.sort()
        return female_dict, male_dict, common_keys

    def load_name_dict(self, path):
        name_list = os.listdir(path)
        name_list.sort()
        name_dict = {}

        for name in name_list:
            year = int(name[:4])
            if year not in name_dict:
                name_dict[year] = []
            name_dict[year].append(name)
        return name_dict


class SubPortraits(Dataset):
    def __init__(self, root, image_name_list, image_labels, dataset_transform, loader=Gray_loader):
        super(SubPortraits, self).__init__()
        self.root = root
        self.image_name_list = image_name_list
        self.image_labels = image_labels
        self.transform = dataset_transform
        self.loader = loader

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_name_list[idx])
        label = self.image_labels[idx]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_name_list)
