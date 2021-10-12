from omegaconf import OmegaConf


def check_hydra_conf(cfg: OmegaConf) -> None:
    if "architecture" in cfg:
        if cfg["architecture"]["base_cnn"] not in {"resnet18", "resnet50"}:
            raise ValueError

    if "parameter" in cfg:
        if "d" in cfg["parameter"]:
            d = cfg["parameter"]["d"]
            if d < 0:
                raise ValueError("the dimensionality `d` must be greater than 0. Not {}".format(d))

    # dataset
    dataset_conf = cfg["dataset"]
    if dataset_conf["name"].lower() not in {"cifar10", "cifar100", "imbalance_cifar100", "sub_sampled_cifar100"}:
        raise ValueError

    if "num_views" in dataset_conf:
        if cfg["dataset"]["num_views"] < 1:
            raise ValueError
