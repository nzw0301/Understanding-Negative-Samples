import json
import logging
from pathlib import Path

import hydra
import torch
import torchvision
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.check_hydra_conf import check_hydra_conf
from src.data.dataset import DownstreamDataset
from src.data.transforms import create_simclr_data_augmentation
from src.data.utils import create_data_loaders
from src.data.utils import fetch_dataset
from src.data.utils import get_num_classes
from src.eval_utils import convert_vectors, centroid_eval
from src.model import CentroidClassifier
from src.model import ContrastiveModel


@hydra.main(config_path="conf", config_name="eval_config")
def main(cfg: OmegaConf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ""
    logger.addHandler(stream_handler)

    check_hydra_conf(cfg)

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

    logger_line = "Using {}".format(device)
    logger.info(logger_line)

    # load self-sup training's config
    weights_path = Path(cfg["experiment"]["target_weight_file"])
    weight_name = weights_path.name
    logger.info("Evaluation by using {}".format(weight_name))

    self_sup_config_path = weights_path.parent / ".hydra" / "config.yaml"
    with open(self_sup_config_path) as f:
        self_sup_conf = yaml.load(f, Loader=yaml.FullLoader)

    # initialise data loaders to create centroids representations
    dataset_name = cfg["dataset"]["name"]
    num_classes = get_num_classes(cfg["dataset"]["name"])
    is_cifar = "cifar" in cfg["dataset"]["name"]

    training_transform = create_simclr_data_augmentation(
        self_sup_conf["dataset"]["strength"], self_sup_conf["dataset"]["size"]
    )
    val_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

    training_dataset, validation_dataset = fetch_dataset(dataset_name, training_transform, val_transform,
                                                         include_val=True)

    training_data_loader, validation_data_loader = create_data_loaders(
        num_workers=cfg["experiment"]["num_workers"], batch_size=cfg["experiment"]["batches"],
        training_dataset=training_dataset, validation_dataset=validation_dataset, train_drop_last=False,
        distributed=False
    )

    logger.info("#train: {}, #val: {}".format(len(training_dataset), len(validation_dataset)))

    model = ContrastiveModel(
        base_cnn=self_sup_conf["architecture"]["base_cnn"], d=self_sup_conf["parameter"]["d"], is_cifar=is_cifar
    )

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    state_dict = torch.load(weights_path)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # load weights trained on self-supervised task
    if use_cuda:
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False, map_location=device)

    # remove projection head or not
    if not cfg["experiment"]["use_projection_head"]:
        model.g = torch.nn.Identity()

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
