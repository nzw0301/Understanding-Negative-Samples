import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision
import yaml
from omegaconf import OmegaConf

from src.data.utils import fetch_dataset, create_data_loaders
from src.data.transforms import create_simclr_data_augmentation
from src.model import ContrastiveModel


def convert_vectors(data_loader: torch.utils.data.DataLoader, model: ContrastiveModel, device: torch.device) -> tuple:
    """
    Convert images to feature representations.

    :param data_loader: Data loader of the dataset.
    :param model: Pre-trained instance.
    :param device: PyTorch's device instance.
    :return: Tuple of numpy array and labels.
    """

    model.eval()
    new_X = []
    new_y = []
    with torch.no_grad():
        for x_batches, y_batches in data_loader:
            new_X.append(model(x_batches.to(device)))
            new_y.append(y_batches)

    X = torch.cat(new_X).cpu()
    y = torch.cat(new_y).cpu()

    return X.numpy(), y.numpy()


@hydra.main(config_path="conf", config_name="analysis_config")
def main(cfg: OmegaConf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ""
    logger.addHandler(stream_handler)

    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = cfg["experiment"]["use_cuda"] and torch.cuda.is_available()

    if use_cuda:
        device_id = cfg["experiment"]["gpu_id"] % torch.cuda.device_count()
        device = torch.device(device_id)
    else:
        device = torch.device("cpu")
    logger.info("Using {}".format(device))

    dataset_name = cfg["dataset"]["name"]
    is_cifar = "cifar" in dataset_name

    # initialise data loaders
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

    training_dataset, validation_dataset = fetch_dataset(dataset_name, transform, transform, include_val=True)
    training_data_loader, validation_data_loader = create_data_loaders(
        num_workers=cfg["experiment"]["num_workers"], batch_size=cfg["experiment"]["batches"],
        training_dataset=training_dataset, validation_dataset=validation_dataset, train_drop_last=False,
        distributed=False
    )

    weights_path = Path(cfg["experiment"]["target_weight_file"])
    key = weights_path.name

    logger.info("Save features extracted by using {}".format(key))

    self_sup_config_path = weights_path.parent / ".hydra" / "config.yaml"
    with open(self_sup_config_path) as f:
        self_sup_conf = yaml.load(f, Loader=yaml.FullLoader)

        model = ContrastiveModel(
            base_cnn=self_sup_conf["architecture"]["base_cnn"], d=self_sup_conf["parameter"]["d"],
            is_cifar=is_cifar
        )

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    state_dict = torch.load(weights_path)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if use_cuda:
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False, map_location=device)

    # remove projection head or not
    if not cfg["experiment"]["use_projection_head"]:
        model.g = torch.nn.Identity()

    X_train, y_train = convert_vectors(training_data_loader, model, device)
    X_val, y_val = convert_vectors(validation_data_loader, model, device)

    fname = "{}.feature.train.npy".format(key)
    np.save(fname, X_train)
    fname = "{}.label.train.npy".format(key)
    np.save(fname, y_train)
    fname = "{}.feature.val.npy".format(key)
    np.save(fname, X_val)
    fname = "{}.label.val.npy".format(key)
    np.save(fname, y_val)

    # with data-augmentation
    transform = create_simclr_data_augmentation(
        self_sup_conf["dataset"]["strength"], self_sup_conf["dataset"]["size"]
    )

    training_dataset, validation_dataset = fetch_dataset(dataset_name, transform, transform, include_val=True)
    training_data_loader, validation_data_loader = create_data_loaders(
        num_workers=cfg["experiment"]["num_workers"], batch_size=cfg["experiment"]["batches"],
        training_dataset=training_dataset, validation_dataset=validation_dataset, train_drop_last=False,
        distributed=False
    )

    for a in range(2):
        X_train, _ = convert_vectors(training_data_loader, model, device)
        X_val, _ = convert_vectors(validation_data_loader, model, device)

        fname = "{}.feature.{}.train.npy".format(key, a)
        np.save(fname, X_train)
        fname = "{}.feature.{}.val.npy".format(key, a)
        np.save(fname, X_val)


if __name__ == "__main__":
    main()
