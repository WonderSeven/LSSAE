import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import TensorDataset
from torchvision.transforms.functional import rotate

from datasets.dataset import MultipleDomainDataset
from engine.configs import Datasets


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape, num_classes):
        super().__init__()
        self.Environments = environments
        self.input_shape = input_shape
        self.num_classes = num_classes

        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = datasets.MNIST(root, train=True, download=True)
        original_dataset_te = datasets.MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))


@Datasets.register('rmnist')
class RotatedMNIST(MultipleEnvironmentMNIST):
    def __init__(self, root, input_shape, num_classes):
        # environments = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        environments = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        # environments = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
        #                 210, 220, 230, 240, 250, 260, 270]
        super(RotatedMNIST, self).__init__(root, environments, self.rotate_dataset, input_shape, num_classes)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), *self.input_shape)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)

