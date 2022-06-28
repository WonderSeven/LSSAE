import os
import pdb
import numpy as np
from PIL import Image
from scipy.io import loadmat

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from datasets.dataset import MultipleDomainDataset
from engine.configs import Datasets


def RGB_loader(path):
    return Image.open(path).convert('RGB')


@Datasets.register('caltran')
class MultipleEnvironmentCalTran(MultipleDomainDataset):
    def __init__(self, root, input_shape, num_classes, dataset_transform=None):
        super().__init__()
        num_domains = 34
        self.Environments = np.arange(num_domains)
        self.input_shape = input_shape
        self.num_classes = num_classes

        if dataset_transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.input_shape[-2:]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = dataset_transform

        if root is None:
            raise ValueError('Data directory not specified!')
        dict_path = os.path.join(root, 'caltran_dataset_labels.mat')
        data_dict = loadmat(dict_path)

        img_names = data_dict['names']
        img_labels = data_dict['labels'][0]
        img_labels = [0 if item <= 0 else 1 for item in img_labels]
        img_names = [np.array2string(item)[9:25]+'.jpg' for item in img_names]

        self.datasets = []

        pre_idx = 0

        for i in range(2, len(self.Environments) + 2):
            data_idx = i // 3
            part_idx = i % 3

            cur_day_image_list = list(filter(lambda k: k.startswith('2013-03-{:02}'.format(4+data_idx)), img_names))
            if part_idx == 0:
                cur_image_list = list(filter(lambda k: k[11:-4] <= '08-00', cur_day_image_list))
            elif part_idx == 1:
                cur_image_list = list(filter(lambda k: '08-00' < k[11:-4] < '16-00', cur_day_image_list))
            else:
                cur_image_list = list(filter(lambda k: k[11:-4] >= '16-00', cur_day_image_list))
            cur_labels = img_labels[pre_idx: pre_idx+len(cur_image_list)]
            # print('Domain idx:{}, image num:{}, label num:{}'.format(i-2, len(cur_image_list), len(cur_labels)))
            pre_idx += len(cur_image_list)
            self.datasets.append(SubCalTran(root, cur_image_list, cur_labels, self.transform))


class SubCalTran(Dataset):
    def __init__(self, root, image_name_list, image_labels, dataset_transform, loader=RGB_loader):
        super(SubCalTran, self).__init__()
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


if __name__ == '__main__':
    data_root = '/data/qtx/DataSets/CalTran'
    ds = MultipleEnvironmentCalTran(data_root, [3, 84, 84], 2)
