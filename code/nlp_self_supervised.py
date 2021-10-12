import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from apex.parallel.LARC import LARC
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.ag_news import CollateSelfSupervised, get_train_val_datasets
from src.loss import NT_Xent
from src.lr_utils import calculate_initial_lr, calculate_warmup_lr
from src.model import ContrastiveFastText


def train(cfg: OmegaConf, training_data_loader: torch.utils.data.DataLoader, model: ContrastiveFastText,
          device: torch.device) -> None:
    """
    Training function.

    :param cfg: Hydra's config instance.
    :param training_data_loader: Training data loader for contrastive learning.
    :param model: Self-supervised model.
    :return: None
    """
    epochs = cfg["experiment"]["epochs"]
    steps_per_epoch = len(training_data_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = cfg["optimizer"]["warmup_epochs"] * steps_per_epoch
    current_step = 0

    model.train()
    simclr_loss_function = NT_Xent(
        temperature=cfg["loss"]["temperature"], device=device
    )

    if cfg["optimizer"]["name"] == "sgd":
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=calculate_initial_lr(cfg),
            momentum=cfg["optimizer"]["momentum"],
            nesterov=False,
            weight_decay=cfg["optimizer"]["decay"])

        # https://github.com/google-research/simclr/blob/master/lars_optimizer.py#L26
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

        cos_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.optim,
            T_max=total_steps - warmup_steps,
        )

    elif cfg["optimizer"]["name"] == "adamW":
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=calculate_initial_lr(cfg),
            weight_decay=cfg["optimizer"]["decay"]
        )

        cos_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
        )

    else:
        raise ValueError("Unsupported optimizer: {}. Must be either `sgd` or `adamW`".format(cfg["optimizer"]["name"]))

    for epoch in range(1, epochs + 1):
        training_loss = 0.
        for views_with_offsets in training_data_loader:
            # adjust learning rate by applying linear warming
            if current_step <= warmup_steps:
                lr = calculate_warmup_lr(cfg, warmup_steps, current_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            optimizer.zero_grad()
            zs = [model(view.to(device), off_set.to(device)) for view, off_set in views_with_offsets]
            loss = simclr_loss_function(zs)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            # adjust learning rate by applying cosine annealing
            if current_step > warmup_steps:
                cos_lr_scheduler.step()

            current_step += 1

        training_loss = training_loss / len(training_data_loader) / training_data_loader.batch_size

        logging.info("Epoch:{}/{} progress:{:.3f} loss:{:.5f}, lr:{:.5f}".format(
            epoch, epochs, epoch / epochs, training_loss, optimizer.param_groups[0]["lr"]
        ))

        if epoch % cfg["experiment"]["save_model_epoch"] == 0 or epoch == epochs:
            save_fname = "epoch_{}-{}".format(epoch, cfg["experiment"]["output_model_name"])
            torch.save(model.state_dict(), save_fname)


@hydra.main(config_path="conf", config_name="simclr_nlp_config")
def main(cfg: OmegaConf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ""
    logger.addHandler(stream_handler)

    seed = cfg["experiment"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpu_id = cfg["experiment"][
                     "gpu_id"] % torch.cuda.device_count()  # NOTE: GPU's id is one origin when we use gnu-parallel.
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    logger.info("Using {}".format(device))

    training_dataset, _ = get_train_val_datasets(
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
        collate_fn=CollateSelfSupervised(cfg["dataset"]["mask_ratio"], replace_data, np.random.RandomState(seed),
                                         cfg["dataset"]["augmentation_type"]),
        drop_last=True
    )

    logger.info("#train: {}, vocab size: {}".format(len(training_dataset), vocab_size))

    model = ContrastiveFastText(
        num_embeddings=vocab_size,
        embedding_dim=cfg["architecture"]["embedding_dim"],
        num_last_hidden_units=cfg["architecture"]["embedding_dim"],
        with_projection_head=True
    ).to(device)

    train(cfg, training_data_loader, model, device)


if __name__ == "__main__":
    main()
