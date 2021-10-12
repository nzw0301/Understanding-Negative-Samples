import json
import logging
from pathlib import Path

import hydra
import torch
import torchvision
import yaml
from omegaconf import OmegaConf

from src.check_hydra_conf import check_hydra_conf
from src.data.transforms import create_simclr_data_augmentation
from src.data.utils import create_data_loaders
from src.data.utils import fetch_dataset
from src.data.utils import get_num_classes
from src.distributed_utils import init_ddp
from src.eval_utils import learnable_eval
from src.model import ContrastiveModel
from src.model import LinearClassifier


@hydra.main(config_path="conf", config_name="linear_eval_config")
def main(cfg: OmegaConf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ""
    logger.addHandler(stream_handler)

    init_ddp(cfg)
    check_hydra_conf(cfg)

    # to reproduce results
    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rank = cfg["distributed"]["local_rank"]
    use_cuda = cfg["experiment"]["use_cuda"] and torch.cuda.is_available()
    logger.info("{}".format(rank))

    # load pre-trained model
    weights_path = Path(cfg["experiment"]["target_weight_file"])
    weight_name = weights_path.name
    self_sup_config_path = weights_path.parent / ".hydra" / "config.yaml"
    with open(self_sup_config_path) as f:
        self_sup_conf = yaml.load(f, Loader=yaml.FullLoader)

    dataset_name = cfg["dataset"]["name"]
    num_classes = get_num_classes(cfg["dataset"]["name"].lower())
    is_cifar = "cifar" in cfg["dataset"]["name"]
    top_k = min(cfg["experiment"]["top_k"], num_classes)

    # initialise data loaders
    training_transform = create_simclr_data_augmentation(self_sup_conf["dataset"]["strength"],
                                                         self_sup_conf["dataset"]["size"])
    val_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

    training_dataset, validation_dataset = fetch_dataset(dataset_name, training_transform, val_transform,
                                                         include_val=True)

    training_data_loader, validation_data_loader = create_data_loaders(
        num_workers=cfg["experiment"]["num_workers"],
        batch_size=cfg["experiment"]["batches"],
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        train_drop_last=True,
        distributed=True
    )

    model = ContrastiveModel(
        base_cnn=self_sup_conf["architecture"]["base_cnn"], d=self_sup_conf["parameter"]["d"], is_cifar=is_cifar
    )

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(rank)

    state_dict = torch.load(weights_path)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if use_cuda:
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False, map_location=rank)

    # get the dimensionality of the representation
    if cfg["experiment"]["use_projection_head"]:
        num_last_units = model.g.projection_head.linear2.out_features
    else:
        num_last_units = model.g.projection_head.linear1.in_features
        model.g = torch.nn.Identity()

    if rank == 0:
        logger.info("#train: {}, #val: {}".format(len(training_dataset), len(validation_dataset)))
        logger.info("Evaluation by using {}".format(weight_name))

    # initialise linear classifier
    # NOTE: the weights are not normalize
    classifier = LinearClassifier(num_last_units, num_classes, normalize=False).to(rank)
    classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[rank])

    # execute linear evaluation protocol
    train_accuracies, train_top_k_accuracies, train_losses, val_accuracies, val_top_k_accuracies, val_losses = \
        learnable_eval(
            cfg, classifier, model, training_data_loader, validation_data_loader, top_k
        )

    if rank == 0:
        classification_results = {}
        classification_results[weight_name] = {
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_top_{}_accuracies".format(top_k): train_top_k_accuracies,
            "val_top_{}_accuracies".format(top_k): val_top_k_accuracies,
            "lowest_val_loss": min(val_losses),
            "highest_val_acc": max(val_accuracies),
            "highest_val_top_k_acc": max(val_top_k_accuracies)
        }

        logger.info("train acc: {}, val acc: {}".format(max(train_accuracies), max(val_accuracies)))

        fname = cfg["experiment"]["classification_results_json_fname"]

        with open(fname, "w") as f:
            json.dump(classification_results, f)


if __name__ == "__main__":
    main()
