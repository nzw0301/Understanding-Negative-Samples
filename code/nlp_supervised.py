import json
import logging
import os
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.ag_news import (CollateSupervised, collate_eval_batch,
                              get_train_val_datasets)
from src.lr_utils import calculate_initial_lr, calculate_warmup_lr
from src.model import SupervisedFastText


def validation(validation_data_loader: torch.utils.data.DataLoader, model: SupervisedFastText,
               device: torch.device) -> Tuple[float, float]:
    """
    :param validation_data_loader: Validation data loader
    :param model: ResNet based classifier.
    :param device: pytorch's device instance

    :return: tuple of validation loss and accuracy
    """

    model.eval()

    sum_loss = 0.
    num_corrects = 0.

    with torch.no_grad():
        for data, targets, offsets in validation_data_loader:
            data, targets, offsets = data.to(device), targets.to(device), offsets.to(device)
            unnormalized_features = model(data, offsets)
            loss = torch.nn.functional.cross_entropy(unnormalized_features, targets, reduction="sum")

            predicted = torch.max(unnormalized_features.data, 1)[1]

            sum_loss += loss.item()
            num_corrects += (predicted == targets).sum().item()

    num_val_samples = len(validation_data_loader.dataset)
    return sum_loss / num_val_samples, num_corrects / num_val_samples


def learning(
        cfg: OmegaConf,
        training_data_loader: torch.utils.data.DataLoader,
        validation_data_loader: torch.utils.data.DataLoader,
        model: SupervisedFastText,
        device: torch.device,
) -> None:
    """
    Learning function including evaluation

    :param cfg: Hydra's config instance
    :param training_data_loader: Training data loader
    :param validation_data_loader: Validation data loader
    :param model: `SupervisedFastText`'s instance.
    :param device: pytorch's device instance.
    :return: None
    """

    epochs = cfg["experiment"]["epochs"]
    steps_per_epoch = len(training_data_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = cfg["optimizer"]["warmup_epochs"] * steps_per_epoch
    current_step = 0
    num_training_samples = len(training_data_loader.dataset)

    validation_losses = []
    validation_accuracies = []
    best_metric = np.finfo(np.float64).max

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=calculate_initial_lr(cfg),
        momentum=0.9,
        nesterov=False,
        weight_decay=0.
    )

    # https://github.com/google-research/simclr/blob/master/lars_optimizer.py#L26
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    cos_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.optim,
        T_max=total_steps - warmup_steps,
    )

    for epoch in range(1, epochs + 1):

        # training
        model.train()

        for (data, targets, offsets) in training_data_loader:
            if current_step <= warmup_steps:
                lr = calculate_warmup_lr(cfg, warmup_steps, current_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            optimizer.zero_grad()
            data, targets, offsets = data.to(device), targets.to(device), offsets.to(device)
            loss = torch.nn.functional.cross_entropy(model(data, offsets), targets)
            loss.backward()
            optimizer.step()

            # adjust learning rate by applying cosine annealing
            if current_step > warmup_steps:
                cos_lr_scheduler.step()

            current_step += 1

        # end of training loop

        logger_line = "Epoch:{}/{} progress:{:.3f} loss:{:.3f}, lr:{:.7f}".format(
            epoch, epochs, epoch / epochs, loss.item(), optimizer.param_groups[0]["lr"]
        )

        validation_loss, validation_acc = validation(validation_data_loader, model, device)

        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_acc)

        if cfg["parameter"]["metric"] == "loss":
            metric = validation_loss
        else:
            # store metric as risk: 1 - accuracy
            metric = 1. - validation_acc

        if metric <= best_metric:
            # delete old checkpoint file
            if "save_fname" in locals():
                if os.path.exists(save_fname):
                    os.remove(save_fname)

            save_fname = cfg["experiment"]["output_model_name"]
            torch.save(model.state_dict(), save_fname)
            best_metric = metric

        logging.info(
            logger_line + " val loss:{:.3f}, val acc:{:.2f}%".format(validation_loss, validation_acc * 100.)
        )

        if cfg["parameter"]["metric"] == "loss":
            logging_line = "best val loss:{:.7f}%".format(best_metric)
        else:
            logging_line = "best val acc:{:.2f}%".format((1. - best_metric) * 100)

        logging.info(logging_line)

    # save validation metrics and both of best metrics
    supervised_results = {
        "validation": {
            "losses": validation_losses,
            "accuracies": validation_accuracies,
            "lowest_loss": min(validation_losses),
            "highest_accuracy": max(validation_accuracies),
        }
    }
    fname = cfg["parameter"]["classification_results_json_fname"]
    with open(fname, "w") as f:
        json.dump(supervised_results, f)


@hydra.main(config_path="conf", config_name="supervised_nlp_config")
def main(cfg: OmegaConf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ""
    logger.addHandler(stream_handler)

    # fix seed
    seed = cfg["experiment"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        gpu_id = cfg["experiment"]["gpu_id"] % torch.cuda.device_count()  # NOTE: GPU's id is one origin

        device = torch.device(gpu_id)

    else:
        device = torch.device("cpu")

    logger.info("Using {}".format(device))

    # initialize data loaders
    training_dataset, validation_dataset = get_train_val_datasets(
        root=Path.home() / "pytorch_datasets",
        min_freq=cfg["dataset"]["min_freq"],
    )
    vocab_size = training_dataset.vocab_size

    if cfg["dataset"]["augmentation_type"] == "erase":
        replace_data = None
    else:
        replace_data = np.load(cfg["dataset"]["replace_data"])
        assert len(replace_data) == vocab_size

    training_data_loader = DataLoader(
        training_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=True,
        collate_fn=CollateSupervised(
            mask_ratio=cfg["dataset"]["mask_ratio"], replace_data=replace_data, rnd=np.random.RandomState(seed),
            augmentation_type=cfg["dataset"]["augmentation_type"]),
        drop_last=True
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=False,
        collate_fn=collate_eval_batch,
        drop_last=False
    )

    num_classes = len(np.unique(validation_dataset.targets))
    logger.info(
        f"#train: {len(training_dataset)}, #val: {len(validation_dataset)}, #classes: {num_classes}, vocab size: {vocab_size}")

    model = SupervisedFastText(
        num_embeddings=vocab_size,
        embedding_dim=cfg["architecture"]["embedding_dim"],
        num_classes=num_classes,
    ).to(device)

    learning(cfg, training_data_loader, validation_data_loader, model, device)


if __name__ == "__main__":
    """
    To run this code,
    `python nlp_supervised.py`
    """
    main()
