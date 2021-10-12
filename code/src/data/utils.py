import torch
import torchvision
from torch.utils.data import DataLoader


def create_data_loaders(num_workers: int, batch_size: int, training_dataset=None, validation_dataset=None,
                        train_drop_last: bool = True, distributed: bool = True):
    """
    :param num_workers: the number of workers for data loader
    :param batch_size: the mini-batch size
    :param training_dataset: Dataset instance for training dataset
    :param validation_dataset: Dataset instance for validation dataset
    :param train_drop_last: whether drop the a part of mini-batches of the training data or not.
    :param distributed: whether use DistributedSampler for DDP training or not.

    :return: list of DataLoaders
    """
    data_loaders = []

    if training_dataset is not None:

        sampler = None

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=True)

        training_data_loader = DataLoader(dataset=training_dataset, sampler=sampler, num_workers=num_workers,
                                          batch_size=batch_size, pin_memory=True, drop_last=train_drop_last,
                                          )
        data_loaders.append(training_data_loader)

    if validation_dataset is not None:

        validation_sampler = None

        if distributed:
            validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset, shuffle=False)

        validation_data_loader = DataLoader(dataset=validation_dataset, sampler=validation_sampler,
                                            num_workers=num_workers, batch_size=batch_size, pin_memory=True,
                                            drop_last=False,
                                            )
        data_loaders.append(validation_data_loader)

    return data_loaders


def fetch_dataset(name: str, train_transform, val_transform, include_val: bool = True):
    """
    :param name: The name of dataset to fetch
    :param train_transform: torchVision's transform for training dataset
    :param val_transform: torchVision's transform for validation dataset
    :param include_val: Whether include validation set or not. This value
        might be `True` for self-supervised training since it doesn't need validation set.
    :return: training dataset or list of training and validation datasets.
    """

    root = "~/pytorch_datasets"

    if name == "cifar10":
        dataset = torchvision.datasets.CIFAR10
    elif "cifar100" in name:
        dataset = torchvision.datasets.CIFAR100
    else:
        ValueError(f"{name} is unsupported.")

    training_dataset = dataset(
        root=root, train=True, download=True, transform=train_transform
    )

    if not include_val:
        return training_dataset
    else:
        val_dataset = dataset(
            root=root, train=False, download=True, transform=val_transform
        )
        return training_dataset, val_dataset


def get_num_classes(dataset_name: str) -> int:
    """
    Get the number of supervised loss given a dataset name.
    :param dataset_name: dataset name
    :return: number of supervised class.
    """
    if dataset_name == "cifar10":
        return 10
    elif "cifar100" in dataset_name:
        return 100
    else:
        raise ValueError("Supported datasets are only original CIFAR10/100")
