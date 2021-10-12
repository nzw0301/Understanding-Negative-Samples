import json
import logging
from pathlib import Path
import numpy as np

import hydra
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataset import BoundCIFAR10, BoundCIFAR100
from src.data.transforms import SimCLRTransforms
from src.data.utils import fetch_dataset
from src.data.utils import get_num_classes
from src.loss import NT_Xent
from src.model import ContrastiveModel


def create_eval_representations(
        model: ContrastiveModel, training_data_loader: DataLoader,
        validation_data_loader: DataLoader, device: torch.device, num_augmentations: int,
        num_classes: int, dim: int = 128,
) -> tuple:
    """
    :param model: Pre-trained contrastive model.
    :param training_data_loader: Training data loader with `shuffle=False`.
    :param validation_data_loader: Validation data loader with `shuffle=False`.
    :param device: PyTorch's device instance.
    :param num_augmentations: The number of data augmentation per sample. The sample is doubled.
    :param num_classes: The number of supervised classes.
    :param dim: The dimensionality of feature representations.
    :return: Tuple of FloatTensor.
    """

    def convert_vectors(model: ContrastiveModel, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
        """
        :param model: Contrastive model.
        :param data_loader: Data loader contains Dataset with SimCLR's data augmentation.
        :param device: PyTorch's cuda device instance.
        :return: Feature representations. shape is (N, d).
        """

        new_X = []

        with torch.no_grad():
            for list_x_batches, _ in data_loader:
                fs = torch.mean(
                    torch.stack(
                        [torch.nn.functional.normalize(model(xs.to(device)), p=2, dim=1)
                         for xs in list_x_batches]),
                    dim=0
                )  # num-batch-size x dim
                new_X.append(fs)

        return torch.cat(new_X)

    def create_mean_weights(
            features: torch.Tensor, targets: torch.LongTensor, num_classes: int
    ) -> torch.Tensor:
        """
        :param features: FloatTensor of feature representations.  shape is (N, D)
        :param targets: LongTensor contains target labels. Shape is (N,)
        :param num_classes: the number of distinct classes.
        :return: shape (num_classes, D)
        """

        if len(features) != len(targets):
            raise ValueError(
                "The number of features and targets must be same: {} != {}".format(len(features), len(targets)))

        weights = []
        for k in range(num_classes):
            ids = torch.where(targets == k)[0]
            weights.append(torch.mean(features[ids], dim=0))

        return torch.stack(weights, dim=0)

    training_fax = torch.zeros(len(training_data_loader.dataset), dim).to(device)
    validation_fax = torch.zeros(len(validation_data_loader.dataset), dim).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(num_augmentations):
            training_fax += convert_vectors(model, training_data_loader, device)  # (N, D)
            validation_fax += convert_vectors(model, validation_data_loader, device)  # (N, D)

        # take average over different data augmentations
        training_fax = training_fax / num_augmentations
        validation_fax = validation_fax / num_augmentations

        targets = torch.LongTensor(training_data_loader.dataset.targets).to(device)
        training_mean_weights = create_mean_weights(training_fax, targets, num_classes)

        targets = torch.LongTensor(validation_data_loader.dataset.targets).to(device)
        validation_mean_weights = create_mean_weights(validation_fax, targets, num_classes)

    return training_fax, validation_fax, training_mean_weights, validation_mean_weights


def eval_bound(
        model: ContrastiveModel, data_loader: DataLoader, device: torch.device,
        epochs: int, mean_weights: torch.FloatTensor, simclr_loss_function: NT_Xent, num_classes: int, logger,
        calculate_centroids_accuracy: bool = True, log_interval: int = 10
) -> dict:

    num_iters_per_epoch = (len(data_loader.dataset) // data_loader.batch_size)
    total_samples_for_bounds = float(num_iters_per_epoch * data_loader.batch_size * epochs)

    logger.info(f"Epochs: {epochs}. Total samples, {int(total_samples_for_bounds / 1000):}K")

    simclr_loss = 0.
    original_simclr_loss = 0.
    upsilon = 0.
    sup_loss = 0.
    partial_sup_loss = 0.
    conflict_term = 0.

    curl_tau = 0.
    curl_sup_loss = 0.
    curl_partial_sup_loss = 0.

    assumption_value = 0.

    model.eval()
    with torch.no_grad():
        for epoch in range(1, epochs + 1):
            internal_simclr_loss = 0.
            internal_original_simclr_loss = 0.
            internal_upsilon = 0.
            internal_sup_loss = 0.
            internal_partial_sup_loss = 0.

            internal_curl_tau = 0.
            internal_curl_sup_loss = 0.
            internal_curl_partial_sup_loss = 0.
            internal_assumption_value = 0.

            for list_x_batches, mean_zs, labels in data_loader:
                mean_zs = mean_zs.to(device)
                labels = labels.to(device)

                zs_list = [model(xs.to(device)) for xs in list_x_batches]  # (2, N, D)

                # the original SimCLR use 2K+1 class classification,
                # but it does not necessary to use another view of negative samples.
                # so we use view1 as key and view2 as positive / negative samples
                internal_simclr_loss += simclr_loss_function.simclr_forward_with_single_view_negative(
                    zs_list).sum().item()
                # compute original SimCLR loss for reference
                # due to `reduction=sum`, returned value is divided by `2`.
                internal_original_simclr_loss += simclr_loss_function(zs_list).sum().item() / 2.

                # use only view 1
                zs = torch.nn.functional.normalize(zs_list[0], p=2, dim=-1)

                # check assumption value `d`
                inner_product = (zs * (mean_weights[labels] - mean_zs)).sum(dim=1)  # (N, )
                internal_assumption_value += torch.sum(
                    inner_product[inner_product < 0.] / simclr_loss_function.temperature
                ).item()

                sampled_unique_classes, class_frequency = torch.unique(labels, return_counts=True)

                # for CURL bound
                # use only samples that come from unique latent classes in mini-batches for loss function
                non_duplicated_classes = sampled_unique_classes[class_frequency == 1]
                num_non_duplicated_classes = len(non_duplicated_classes)
                internal_curl_tau += data_loader.batch_size - num_non_duplicated_classes
                if num_non_duplicated_classes > 0:
                    used_sample_ids_for_curl = torch.cat([torch.where(labels == c)[0] for c in non_duplicated_classes])

                # observe all supervised class
                if len(sampled_unique_classes) == num_classes:

                    internal_upsilon += data_loader.batch_size
                    _sup_loss = simclr_loss_function.bound_loss(zs, mean_weights, labels)
                    internal_sup_loss += _sup_loss.sum().item()

                    # for curl bound
                    if num_non_duplicated_classes > 0:
                        loss = _sup_loss[used_sample_ids_for_curl].sum().item()

                        internal_curl_partial_sup_loss += loss
                        internal_curl_sup_loss += loss  # this value does not appear in the original paper.

                else:  # observe a part of supervised classes

                    # convert labels -> unique sub-problem
                    partial_labels = torch.cat(
                        [torch.where(sampled_unique_classes == label)[0] for label in labels]).to(device)
                    partial_mean_weights = mean_weights[sampled_unique_classes]
                    _partial_loss = simclr_loss_function.bound_loss(zs, partial_mean_weights, partial_labels)
                    internal_partial_sup_loss += _partial_loss.sum().item()

                    # for curl bound
                    if num_non_duplicated_classes > 0:
                        internal_curl_partial_sup_loss += _partial_loss[used_sample_ids_for_curl].sum().item()

                # conflict term
                # for each sample, conflict term is computed
                # to avoid overflow, each conflict term is divided the the total number of samples at a each epoch
                num_conflicts = class_frequency[class_frequency >= 2]
                conflict_term += torch.sum(
                    num_conflicts * torch.log(
                        num_conflicts.type(torch.FloatTensor).to(device))).item() / total_samples_for_bounds

            simclr_loss += internal_simclr_loss / total_samples_for_bounds
            original_simclr_loss += internal_original_simclr_loss / total_samples_for_bounds
            upsilon += internal_upsilon / total_samples_for_bounds
            sup_loss += internal_sup_loss / total_samples_for_bounds
            partial_sup_loss += internal_partial_sup_loss / total_samples_for_bounds
            # skip conflict term since it has already been divided.

            curl_tau += internal_curl_tau / total_samples_for_bounds
            curl_sup_loss += internal_curl_sup_loss / total_samples_for_bounds
            curl_partial_sup_loss += internal_curl_partial_sup_loss / total_samples_for_bounds

            assumption_value += internal_assumption_value / total_samples_for_bounds

            if (epoch - 1) % log_interval == 0 or epochs == epoch:
                # all loss value is already divided the total number of samples during the whole training
                # thus the following part is to recover the current approximations.
                inverse_progress = epochs / epoch

                _simclr_loss = simclr_loss * inverse_progress
                _original_simclr_loss = original_simclr_loss * inverse_progress
                _upsilon = upsilon * inverse_progress
                _conflict_term = conflict_term * inverse_progress
                _sup_loss = sup_loss * inverse_progress
                _partial_sup_loss = partial_sup_loss * inverse_progress
                _lower_bound = 0.5 * (_sup_loss + _partial_sup_loss + _conflict_term)

                # curl
                _curl_tau = curl_tau * inverse_progress
                _curl_sup_loss = curl_sup_loss * inverse_progress
                _curl_partial_sup_loss = curl_partial_sup_loss * inverse_progress
                _curl_lower_bound = _curl_partial_sup_loss + _conflict_term

                # assumption value
                _assumption_value = assumption_value * inverse_progress

                logger.info(
                    f"SimCLR: {_simclr_loss:.2f}, OriginalSimCLR: {_original_simclr_loss:.2f}, υ: {_upsilon:.2f}, "
                    f"Sup: {_sup_loss:.2f}, Part sup: {_partial_sup_loss:.2f}, Conflict: {_conflict_term:.2f}, "
                    f"Bound: {_lower_bound:.2f}"
                )
                logger.info(
                    f"CURL -- τ: {_curl_tau:.2f}, Sup: {_curl_sup_loss:.2f}, "
                    f"Part sup: {_curl_partial_sup_loss:.2f}, Bound: {_curl_lower_bound:.2f} "
                    f"Assumption value {_assumption_value:.2f}"
                )

    lower_bound = 0.5 * (sup_loss + partial_sup_loss + conflict_term)
    curl_lower_bound = curl_partial_sup_loss + conflict_term

    results = {
        "SimCLR_loss": simclr_loss,
        "Original_SimCLR_loss": original_simclr_loss,
        "upsilon": upsilon,
        "conflict_term": conflict_term,
        "sup_loss": sup_loss,
        "partial_sup_loss": partial_sup_loss,
        "bound": lower_bound,
        "curl_tau": curl_tau,
        "curl_sup_loss": curl_sup_loss,
        "curl_partial_sup_loss": curl_partial_sup_loss,
        "curl_bound": curl_lower_bound,
        "assumption_value": assumption_value
    }

    # compute centroid classifier's supervised performance for reference.
    if calculate_centroids_accuracy:

        with torch.no_grad():
            correct = 0.
            for list_x_batches, _, labels in data_loader:
                zs_list = [model(xs.to(device)) for xs in list_x_batches]  # (2, N, D)
                labels = labels.to(device)

                zs = torch.mean(
                    torch.nn.functional.normalize(torch.stack(zs_list), p=2, dim=-1),
                    dim=0
                )
                output = zs.matmul(mean_weights.t())

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()

        accuracy = correct / (len(data_loader) * data_loader.batch_size) * 100.
        results["centroid_accuracy"] = accuracy

    return results


@hydra.main(config_path="conf", config_name="bound_config")
def main(cfg: OmegaConf):
    # initialise logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ""
    logger.addHandler(stream_handler)

    # to reproduce results
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

    # initialise data loaders
    dataset_name = cfg["dataset"]["name"]
    is_cifar = "cifar" in cfg["dataset"]["name"]
    num_classes = get_num_classes(dataset_name)
    num_workers = cfg["experiment"]["num_workers"]

    weights_path = Path(cfg["experiment"]["target_weight_file"])
    self_sup_config_path = weights_path.parent / ".hydra" / "config.yaml"
    with open(self_sup_config_path) as f:
        self_sup_conf = yaml.load(f, Loader=yaml.FullLoader)

    # data loaders for centroids classifier (mu_c)
    batch_size = self_sup_conf["experiment"]["batches"]
    feature_dim = self_sup_conf["parameter"]["d"]
    transform = SimCLRTransforms(strength=self_sup_conf["dataset"]["strength"], size=self_sup_conf["dataset"]["size"],
                                 num_views=2)

    training_dataset, validation_dataset = fetch_dataset(dataset_name, transform, transform, include_val=True)

    # both datasets are not shuffled and not use `drop_last`
    training_data_loader = DataLoader(dataset=training_dataset, shuffle=False, num_workers=num_workers,
                                      batch_size=1024, pin_memory=True, drop_last=False)
    validation_data_loader = DataLoader(dataset=validation_dataset, shuffle=False, num_workers=num_workers,
                                        batch_size=1024, pin_memory=True, drop_last=False)

    # load pre-trained model
    weight_name = weights_path.name

    model = ContrastiveModel(
        base_cnn=self_sup_conf["architecture"]["base_cnn"], d=feature_dim, is_cifar=is_cifar
    )

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    state_dict = torch.load(weights_path)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if use_cuda:
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False, map_location=device)

    logger.info("#train: {}, #val: {}".format(len(training_dataset), len(validation_dataset)))
    logger.info("Bound for {} with K+1={}".format(weight_name, batch_size))

    epochs = cfg["experiment"]["epochs"]
    num_augmentations = cfg["experiment"]["num_augmentations"]  # num samples to approximate E_{a \sim A} f(a(x))

    # approximate \mu_c per supervised class
    logger.info(f"Approximating mean classifier with {num_augmentations * 2} augmentations per sample")

    # `*_fax` is average representation of normalized features representation per sample:
    # E_{a \sim A} f(a(x))
    # `*_mean_weights` are averaged feature representation of normalized features representation per class:
    # N_c \sum_{x \sim D_c} E_{a \sim A} f(a(x))
    training_fax, validation_fax, training_mean_weights, validation_mean_weights = create_eval_representations(
        model, training_data_loader, validation_data_loader, device, num_augmentations, num_classes=num_classes,
        dim=feature_dim
    )

    np.save("training_fax", training_fax.to("cpu").numpy())
    np.save("validation_fax", validation_fax.to("cpu").numpy())
    np.save("training_mean_weights", training_mean_weights.to("cpu").numpy())
    np.save("validation_mean_weights", validation_mean_weights.to("cpu").numpy())
    del training_fax, validation_mean_weights

    simclr_loss_function = NT_Xent(
        temperature=self_sup_conf["loss"]["temperature"], device=device, reduction="none"
    )

    # create dataset and its dataloader for bound analysis.
    root = "~/pytorch_datasets"
    if dataset_name == "cifar10":
        dataset = BoundCIFAR10
    else:
        dataset = BoundCIFAR100

    # transform is applied to only images -- not `fax`.
    validation_dataset = dataset(root=root, fax=validation_fax.cpu().numpy(), train=False, download=True,
                                 transform=transform)

    # shuffle and use `drop_last` to make the size of mini-batches consistent
    validation_data_loader = DataLoader(dataset=validation_dataset, shuffle=True, num_workers=num_workers,
                                        batch_size=batch_size, pin_memory=True, drop_last=True)

    results = {}

    logger.info("Computing train x val bound")
    _results = eval_bound(model=model, data_loader=validation_data_loader, device=device, epochs=epochs,
                          mean_weights=training_mean_weights, simclr_loss_function=simclr_loss_function,
                          num_classes=num_classes, logger=logger, calculate_centroids_accuracy=True)
    results["training-validation"] = {metric: value for metric, value in _results.items()}

    fname = cfg["experiment"]["bound_json_fname"]

    with open(fname, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
