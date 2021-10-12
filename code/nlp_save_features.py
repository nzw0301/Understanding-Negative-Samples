import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data import ag_news
from src.model import ContrastiveFastText


def convert_vectors(data_loader: torch.utils.data.DataLoader, model: ContrastiveFastText,
                    device: torch.device) -> tuple:
    """
    Convert images to feature representations.

    :param data_loader: Data loader of the dataset.
    :param model: Pre-trained instance.
    :param device: PyTorch's device instance.
    :return: Tuple of numpy; data and labels.
    """

    model.eval()
    new_X = []
    new_y = []
    with torch.no_grad():
        for x_batches, y_batches, offsets in data_loader:
            new_X.append(
                model(x_batches.to(device), offsets.to(device))
            )
            new_y.append(y_batches)

    X = torch.cat(new_X).cpu()
    y = torch.cat(new_y).cpu()

    return X.numpy(), y.numpy()


@hydra.main(config_path="conf", config_name="nlp_analysis_config")
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

    # initialise data loaders
    training_dataset, validation_dataset = ag_news.get_train_val_datasets(
        root=Path.home() / "pytorch_datasets",
        min_freq=cfg["dataset"]["min_freq"],
    )

    training_data_loader = DataLoader(
        training_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=False,
        collate_fn=ag_news.collate_eval_batch,
        drop_last=False
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=False,
        collate_fn=ag_news.collate_eval_batch,
        drop_last=False
    )

    weights_path = Path(cfg["experiment"]["target_weight_file"])
    key = weights_path.name
    vocab_size = training_dataset.vocab_size

    logger.info("Save features extracted by using {}".format(key))

    self_sup_config_path = weights_path.parent / ".hydra" / "config.yaml"
    with open(self_sup_config_path) as f:
        self_sup_conf = yaml.load(f, Loader=yaml.FullLoader)

    model = ContrastiveFastText(
        num_embeddings=vocab_size,
        embedding_dim=self_sup_conf["architecture"]["embedding_dim"],
        num_last_hidden_units=self_sup_conf["architecture"]["embedding_dim"],
        with_projection_head=True
    ).to(device)

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

    vocab_size = training_dataset.vocab_size
    augmentation_type = self_sup_conf["dataset"]["augmentation_type"]
    if augmentation_type == "erase":
        replace_data = None
    else:
        replace_data = np.load(self_sup_conf["dataset"]["replace_data"])
        assert len(replace_data) == vocab_size

    # with data-augmentation
    training_data_loader = DataLoader(
        training_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=False,
        collate_fn=ag_news.CollateSupervised(
            self_sup_conf["dataset"]["mask_ratio"], replace_data, np.random.RandomState(seed), augmentation_type),
        drop_last=False
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=False,
        collate_fn=ag_news.CollateSupervised(
            self_sup_conf["dataset"]["mask_ratio"], replace_data, np.random.RandomState(seed), augmentation_type),
        drop_last=False
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
