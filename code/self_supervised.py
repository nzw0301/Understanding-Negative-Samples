import logging

import hydra
import numpy as np
import torch
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.check_hydra_conf import check_hydra_conf
from src.data.transforms import SimCLRTransforms
from src.data.utils import create_data_loaders
from src.data.utils import fetch_dataset
from src.distributed_utils import init_ddp
from src.loss import NT_Xent
from src.lr_utils import calculate_warmup_lr, calculate_initial_lr
from src.model import ContrastiveModel


def exclude_from_wt_decay(named_params, weight_decay, skip_list=("bias", "bn")) -> tuple:
    """
    :param named_params: Model's named_params.
    :param weight_decay: weight_decay's parameter.
    :param skip_list: list of names to exclude weight decay.
    :return: dictionaries
    """
    # https://github.com/nzw0301/pytorch-lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L90-L105
    # https://github.com/google-research/simclr/blob/3fb622131d1b6dee76d0d5f6aac67db84dab3800/model_util.py#L99

    params = []
    excluded_params = []

    for name, param in named_params:

        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)

    return {"params": params, "weight_decay": weight_decay}, {"params": excluded_params, "weight_decay": 0.}


def train(cfg: OmegaConf, training_data_loader: torch.utils.data.DataLoader, model: ContrastiveModel,) -> None:
    """
    Training function.

    :param cfg: Hydra's config instance.
    :param training_data_loader: Training data loader for contrastive learning.
    :param model: Self-supervised model.
    :return: None
    """
    local_rank = cfg["distributed"]["local_rank"]
    epochs = cfg["experiment"]["epochs"]
    # because the drop=True in data loader,
    steps_per_epoch = len(training_data_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = cfg["optimizer"]["warmup_epochs"] * steps_per_epoch
    current_step = 0

    model.train()
    simclr_loss_function = NT_Xent(
        temperature=cfg["loss"]["temperature"], device=local_rank
    )

    optimizer = torch.optim.SGD(
        params=exclude_from_wt_decay(model.named_parameters(), weight_decay=cfg["optimizer"]["decay"]),
        lr=calculate_initial_lr(cfg),
        momentum=cfg["optimizer"]["momentum"],
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
        training_data_loader.sampler.set_epoch(epoch)

        for views, _ in training_data_loader:
            # adjust learning rate by applying linear warming
            if current_step <= warmup_steps:
                lr = calculate_warmup_lr(cfg, warmup_steps, current_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            optimizer.zero_grad()
            zs = [model(view.to(local_rank)) for view in views]
            loss = simclr_loss_function(zs)

            loss.backward()
            optimizer.step()

            # adjust learning rate by applying cosine annealing
            if current_step > warmup_steps:
                cos_lr_scheduler.step()

            current_step += 1

        if local_rank == 0:
            logging.info("Epoch:{}/{} progress:{:.3f} loss:{:.3f}, lr:{:.7f}".format(
                epoch, epochs, epoch / epochs, loss.item(), optimizer.param_groups[0]["lr"]
            ))

            if epoch % cfg["experiment"]["save_model_epoch"] == 0 or epoch == epochs:
                save_fname = "epoch_{}-{}".format(epoch, cfg["experiment"]["output_model_name"])
                torch.save(model.state_dict(), save_fname)


@hydra.main(config_path="conf", config_name="simclr_config")
def main(cfg: OmegaConf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ""
    logger.addHandler(stream_handler)

    init_ddp(cfg)
    check_hydra_conf(cfg)

    seed = cfg["experiment"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rank = cfg["distributed"]["local_rank"]
    logger.info("Using {}".format(rank))

    transform = SimCLRTransforms(strength=cfg["dataset"]["strength"], size=cfg["dataset"]["size"],
                                 num_views=cfg["dataset"]["num_views"])

    dataset_name = cfg["dataset"]["name"].lower()

    training_dataset = fetch_dataset(dataset_name, transform, None, include_val=False)

    training_data_loader = create_data_loaders(
        num_workers=cfg["experiment"]["num_workers"],
        batch_size=cfg["experiment"]["batches"],
        training_dataset=training_dataset,
        validation_dataset=None
    )[0]

    is_cifar = "cifar" in cfg["dataset"]["name"]

    if rank == 0:
        logger.info("#train: {}".format(len(training_data_loader.dataset)))

    model = ContrastiveModel(base_cnn=cfg["architecture"]["base_cnn"], d=cfg["parameter"]["d"], is_cifar=is_cifar)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train(cfg, training_data_loader, model)


if __name__ == "__main__":
    main()
