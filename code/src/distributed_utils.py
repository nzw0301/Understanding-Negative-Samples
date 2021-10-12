import torch
import torch.distributed as dist
from omegaconf import OmegaConf


def init_ddp(conf: OmegaConf) -> None:

    rank = conf["distributed"]["local_rank"]
    world_size = conf["distributed"]["world_size"]
    dist_url = conf["distributed"]["dist_url"]

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()
