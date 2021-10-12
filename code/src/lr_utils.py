import numpy as np
from omegaconf import OmegaConf


def calculate_initial_lr(cfg: OmegaConf) -> float:
    """
    Proposed initial learning rates by SimCLR paper.

    Note: SimCLR paper says squared learning rate is better when the size of mini-batches is small.

    :param cfg: Hydra's config.
    :return: Initial learning rate whose type is float.
    """

    if cfg["optimizer"]["linear_schedule"]:
        scaled_lr = cfg["optimizer"]["lr"] * cfg["experiment"]["batches"] / 256.
    else:
        scaled_lr = cfg["optimizer"]["lr"] * np.sqrt(cfg["experiment"]["batches"])

    return scaled_lr


def calculate_warmup_lr(cfg: OmegaConf, warmup_steps: int, current_step: int) -> float:
    """
    Calculate a learning rate during warmup period given a current step.
    :param cfg: Hydra's config file.
    :param warmup_steps: The number of steps for warmup.
    :param current_step: the current step.
    :return: learning rate value.
    """

    initial_lr = calculate_initial_lr(cfg)

    if warmup_steps > 0.:
        learning_rate = current_step / warmup_steps * initial_lr
    else:
        learning_rate = initial_lr

    return learning_rate
