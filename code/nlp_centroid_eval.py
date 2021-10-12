import json
import logging
from pathlib import Path
from typing import Union

import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data import ag_news
from src.data.dataset import DownstreamDataset
from src.eval_utils import centroid_eval
from src.model import CentroidClassifier, ContrastiveFastText, SupervisedFastText


def convert_vectors(
        data_loader: torch.utils.data.DataLoader, model: Union[ContrastiveFastText, SupervisedFastText],
        device: torch.device, normalized: bool
) -> tuple:
    """
    Convert experiment to feature representations.
    :param data_loader: Tata loader for raw experiment.
    :param model: Pre-trained model.
    :param device: PyTorch's device instance.
    :param normalized: Whether normalize the feature representation or not.

    :return: Tuple of tensors: features and labels.
    """

    new_X = []
    new_y = []
    model.eval()

    with torch.no_grad():
        for x_batches, y_batches, offsets in data_loader:
            fs = model(x_batches.to(device), offsets.to(device))
            if normalized:
                fs = torch.nn.functional.normalize(fs, p=2, dim=1)
            new_X.append(fs)
            new_y.append(y_batches)

    X = torch.cat(new_X)
    y = torch.cat(new_y)

    return X, y


@hydra.main(config_path="conf", config_name="nlp_eval_config")
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

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device_id = cfg["experiment"]["gpu_id"] % torch.cuda.device_count()
        device = torch.device(device_id)
    else:
        device = torch.device("cpu")

    logger_line = "Using {}".format(device)
    logger.info(logger_line)

    # load self-sup training's config
    weights_path = Path(cfg["experiment"]["target_weight_file"])
    weight_name = weights_path.name
    logger.info("Evaluation by using {}".format(weight_name))

    pre_train_config_path = weights_path.parent / ".hydra" / "config.yaml"
    with open(pre_train_config_path) as f:
        pre_train_conf = yaml.load(f, Loader=yaml.FullLoader)

    # initialise data loaders
    training_dataset, validation_dataset = ag_news.get_train_val_datasets(
        root=Path.home() / "pytorch_datasets",
        min_freq=cfg["dataset"]["min_freq"],
    )
    vocab_size = training_dataset.vocab_size
    mask_ratio = pre_train_conf["dataset"]["mask_ratio"]
    aug_type = pre_train_conf["dataset"]["augmentation_type"]

    if aug_type == "erase":
        replace_data = None
    else:
        replace_data = np.load(pre_train_conf["dataset"]["replace_data"])
        assert len(replace_data) == vocab_size

    training_data_loader = DataLoader(
        training_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=False,
        collate_fn=ag_news.CollateSupervised(mask_ratio, replace_data, np.random.RandomState(seed), aug_type),
        drop_last=False
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=False,
        collate_fn=ag_news.collate_eval_batch,
        drop_last=False
    )

    num_classes = len(np.unique(validation_dataset.targets))

    logger.info("#train: {}, #val: {}".format(len(training_dataset), len(validation_dataset)))

    assert pre_train_conf["parameter"]["algorithm"] in ["self_supervised", "supervised"]
    is_self_supervised_pretrain = pre_train_conf["parameter"]["algorithm"] == "self_supervised"
    d = pre_train_conf["architecture"]["embedding_dim"]
    if is_self_supervised_pretrain:
        model = ContrastiveFastText(
            num_embeddings=vocab_size,
            embedding_dim=d,
            num_last_hidden_units=d,
            with_projection_head=True
        ).to(device)
    else:
        model = SupervisedFastText(
            num_embeddings=vocab_size,
            embedding_dim=d
        ).to(device)


    state_dict = torch.load(weights_path)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # load weights trained on self-supervised task
    if use_cuda:
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False, map_location=device)

    # remove projection head or not
    if not cfg["experiment"]["use_projection_head"]:
        if is_self_supervised_pretrain:
            model.g = torch.nn.Identity()
        else:
            model.f.fc = torch.nn.Identity()

    # create feature representations and classifier
    x, y = convert_vectors(training_data_loader, model, device, normalized=True)
    downstream_training_dataset = DownstreamDataset(x, y)

    classifier = CentroidClassifier(
        weights=CentroidClassifier.create_weights(downstream_training_dataset, num_classes=num_classes).to(device)
    )

    # create data_loader for centroids classifier's input
    x, y = convert_vectors(validation_data_loader, model, device, normalized=True)
    downstream_val_dataset = DownstreamDataset(x, y)

    downstream_training_data_loader = DataLoader(
        dataset=downstream_training_dataset, batch_size=cfg["experiment"]["batches"], shuffle=False,
    )
    downstream_val_data_loader = DataLoader(
        dataset=downstream_val_dataset, batch_size=cfg["experiment"]["batches"], shuffle=False,
    )

    top_k = min(cfg["experiment"]["top_k"], num_classes)
    train_acc, train_top_k_acc = centroid_eval(downstream_training_data_loader, device, classifier, top_k)
    val_acc, val_top_k_acc = centroid_eval(downstream_val_data_loader, device, classifier, top_k)

    classification_results = {}
    classification_results[weight_name] = {
        "train_acc": train_acc,
        "train_top_{}_acc".format(top_k): train_top_k_acc,
        "val_acc": val_acc,
        "val_top_{}_acc".format(top_k): val_top_k_acc
    }
    logger.info("train acc: {}, val acc: {}".format(train_acc, val_acc))

    # save evaluation metric
    fname = cfg["experiment"]["classification_results_json_fname"]

    with open(fname, "w") as f:
        json.dump(classification_results, f)


if __name__ == "__main__":
    main()
