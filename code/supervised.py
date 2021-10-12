import json
import logging
import os

import hydra
import numpy as np
import torch
import torchvision
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.check_hydra_conf import check_hydra_conf
from src.data.transforms import create_simclr_data_augmentation
from src.data.utils import fetch_dataset, create_data_loaders
from src.data.utils import get_num_classes
from src.distributed_utils import init_ddp
from src.eval_utils import make_two_vector_for_confusion_matrix
from src.lr_utils import calculate_warmup_lr, calculate_initial_lr
from src.model import SupervisedModel


def validation(validation_data_loader: torch.utils.data.DataLoader, model: SupervisedModel, local_rank: int) -> tuple:
    """
    :param validation_data_loader: Validation data loader
    :param model: ResNet based classifier.
    :param local_rank: local rank.
    :return: validation loss, the number of corrected samples, and the size of samples on a local
    """

    model.eval()

    sum_loss = torch.tensor([0.]).to(local_rank)
    num_corrects = torch.tensor([0.]).to(local_rank)

    with torch.no_grad():
        for data, targets in validation_data_loader:
            data, targets = data.to(local_rank), targets.to(local_rank)
            unnormalized_features = model(data)
            loss = torch.nn.functional.cross_entropy(unnormalized_features, targets, reduction="sum")

            predicted = torch.max(unnormalized_features.data, 1)[1]

            sum_loss += loss.item()
            num_corrects += (predicted == targets).sum()

    return sum_loss, num_corrects


def learning(
        cfg: OmegaConf,
        training_data_loader: torch.utils.data.DataLoader,
        validation_data_loader: torch.utils.data.DataLoader,
        model: SupervisedModel,
) -> None:
    """
    Learning function including evaluation

    :param cfg: Hydra's config instance
    :param training_data_loader: Training data loader
    :param validation_data_loader: Validation data loader
    :param model: `SupervisedModel`'s instance.
    :return: None
    """

    local_rank = cfg["distributed"]["local_rank"]
    num_gpus = cfg["distributed"]["world_size"]
    epochs = cfg["experiment"]["epochs"]
    num_training_samples = len(training_data_loader.dataset)
    num_val_samples = len(validation_data_loader.dataset)
    steps_per_epoch = len(training_data_loader)  # because the drop=True
    total_steps = epochs * steps_per_epoch
    warmup_steps = cfg["optimizer"]["warmup_epochs"] * steps_per_epoch
    current_step = 0

    validation_losses = []
    validation_accuracies = []
    best_metric = np.finfo(np.float64).max

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=calculate_initial_lr(cfg),
        momentum=cfg["optimizer"]["momentum"],
        nesterov=False,
        weight_decay=cfg["optimizer"]["decay"]
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
        training_data_loader.sampler.set_epoch(epoch)

        for data, targets in training_data_loader:
            # adjust learning rate by applying linear warming
            if current_step <= warmup_steps:
                lr = calculate_warmup_lr(cfg, warmup_steps, current_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            optimizer.zero_grad()
            data, targets = data.to(local_rank), targets.to(local_rank)
            loss = torch.nn.functional.cross_entropy(model(data), targets)
            loss.backward()
            optimizer.step()

            # adjust learning rate by applying cosine annealing
            if current_step > warmup_steps:
                cos_lr_scheduler.step()

            current_step += 1

        if local_rank == 0:
            logger_line = "Epoch:{}/{} progress:{:.3f} loss:{:.3f}, lr:{:.7f}".format(
                epoch, epochs, epoch / epochs, loss.item(), optimizer.param_groups[0]["lr"]
            )

        sum_val_loss, num_val_corrects = validation(validation_data_loader, model, local_rank)

        torch.distributed.barrier()
        torch.distributed.reduce(sum_val_loss, dst=0)
        torch.distributed.reduce(num_val_corrects, dst=0)

        # logging and save checkpoint
        if local_rank == 0:
            validation_loss = sum_val_loss.item() / num_val_samples
            validation_acc = num_val_corrects.item() / num_val_samples
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

    if local_rank == 0:
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

    torch.distributed.barrier()
    if cfg["parameter"]["save_predictions"]:
        model.load_state_dict(
            torch.load(cfg["experiment"]["output_model_name"], map_location={'cuda:%d' % 0: 'cuda:%d' % local_rank}))

        ys, ys_pred = make_two_vector_for_confusion_matrix(validation_data_loader, local_rank, model, None,
                                                           normalized=False)

        np.save(file=f"y_{local_rank}.npy", arr=np.array(ys))
        np.save(file=f"y_{local_rank}_pred.npy", arr=np.array(ys_pred))


@hydra.main(config_path="conf", config_name="supervised_config")
def main(cfg: OmegaConf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ""
    logger.addHandler(stream_handler)

    check_hydra_conf(cfg)
    init_ddp(cfg)

    # fix seed
    seed = cfg["experiment"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rank = cfg["distributed"]["local_rank"]
    logger.info("Using cuda:{}".format(rank))

    # initialise data loaders
    dataset_name = cfg["dataset"]["name"]
    num_classes = get_num_classes(cfg["dataset"]["name"])
    is_cifar = "cifar" in cfg["dataset"]["name"]

    training_transform = create_simclr_data_augmentation(cfg["dataset"]["strength"], size=cfg["dataset"]["size"])
    val_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])

    training_dataset, validation_dataset = fetch_dataset(dataset_name, training_transform, val_transform)
    training_data_loader, validation_data_loader = create_data_loaders(
        num_workers=cfg["experiment"]["num_workers"],
        batch_size=cfg["experiment"]["batches"],
        training_dataset=training_dataset,
        validation_dataset=validation_dataset
    )

    if rank == 0:
        logger.info("#train: {}, #val: {}".format(len(training_dataset), len(validation_dataset)))

    model = SupervisedModel(base_cnn=cfg["architecture"]["base_cnn"], num_classes=num_classes, is_cifar=is_cifar)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    learning(cfg, training_data_loader, validation_data_loader, model)


if __name__ == "__main__":
    """
    To run this code,
    `python launch.py --nproc_per_node={The number of GPUs on a single machine} supervised.py`
    """
    main()
