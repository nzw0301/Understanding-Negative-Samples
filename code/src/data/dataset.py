import numpy as np
import torch
import torchvision

from .ag_news import IntAGNEWS


class DownstreamDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        assert len(data) == len(targets)

        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> tuple:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)


class BoundCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, fax: np.ndarray, train=True, transform=None, target_transform=None,
                 download=False):
        """
        DataSet class for CIFAR-10 to evaluate bounds.

        :param root: Same to the original
        :param fax: Averaged feature representation over data augmentations per sample
        :param train: Same to the original
        :param transform: Same to the original
        :param target_transform: Same to the original
        :param download: Same to the original
        """
        super(BoundCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.fax = fax.astype(np.float32)
        if len(fax) != len(self.data):
            raise ValueError("`fax` must have the the same number of samples")

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, torch.tensor(self.fax[index]), target


class BoundCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, fax: torch.FloatTensor, train=True, transform=None, target_transform=None,
                 download=False):
        """
        DataSet class for CIFAR-100 to evaluate bounds.

        :param root: Same to the original
        :param fax: Averaged feature representation over data augmentations per sample
        :param train: Same to the original
        :param transform: Same to the original
        :param target_transform: Same to the original
        :param download: Same to the original
        """
        super(BoundCIFAR100, self).__init__(root, train, transform, target_transform, download)
        self.fax = fax
        if len(fax) != len(self.data):
            raise ValueError("`fax` must have the the same number of samples")

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, self.fax[index], target


class BoundAGNEWS(torch.utils.data.Dataset):
    def __init__(self, fax: torch.FloatTensor, int_ag_news: IntAGNEWS) -> None:
        self.fax = fax
        self.data = int_ag_news
        if len(fax) != len(self.data):
            raise ValueError("`fax` must have the the same number of samples")

    def __getitem__(self, index) -> tuple:
        words, target = self.data.__getitem__(index)
        return words, self.fax[index], target

    def __len__(self) -> int:
        return len(self.data)
