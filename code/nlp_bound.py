import json
import logging
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from faiss import Kmeans

from src.data import ag_news
from src.data.dataset import BoundAGNEWS
from src.loss import NT_Xent
from src.model import ContrastiveFastText


def create_eval_representations(
        model: ContrastiveFastText, training_data_loader: DataLoader, validation_data_loader: DataLoader,
        device: torch.device, num_augmentations: int, num_classes_per_class: int = 1, dim: int = 128,
        seed: int = 7,
) -> tuple:
    """
    :param model: Pre-trained contrastive model.
    :param training_data_loader: Training data loader with `shuffle=False`.
    :param validation_data_loader: Validation data loader with `shuffle=False`.
    :param device: PyTorch's device instance.
    :param num_augmentations: The number of data augmentation per sample. The sample is doubled.
    :param num_classes_per_class: The number of supervised classes. When the value is greater than 1,
        we peroform the k-mean clustering on the same labeled dataset in the training data,
        then each label is assigned on the basis of the cluster id. Thus the total number of classes becomes
        the original `classes x  num_classes_per_class`.
    :param dim: The dimensionality of feature representations.
    :return: Tuple of FloatTensor.
    """

    def convert_vectors(model: ContrastiveFastText, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
        """
        :param model: Contrastive model.
        :param data_loader: Data loader contains Dataset with SimCLR's data augmentation.
        :param device: PyTorch's cuda device instance.
        :return: Feature representations. shape is (N, d).
        """

        new_X = []

        with torch.no_grad():
            for list_x_batches_with_offsets in data_loader:
                fs = torch.mean(
                    torch.stack(
                        [torch.nn.functional.normalize(model(xs.to(device), offsets.to(device)), p=2, dim=1)
                         for xs, offsets in list_x_batches_with_offsets]),
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

        num_classes = len(np.unique(validation_data_loader.dataset.targets))

        if num_classes_per_class == 1:
            targets = torch.LongTensor(training_data_loader.dataset.targets).to(device)
            training_mean_weights = create_mean_weights(training_fax, targets, num_classes)

            targets = torch.LongTensor(validation_data_loader.dataset.targets).to(device)
            validation_mean_weights = create_mean_weights(validation_fax, targets, num_classes)
        else:
            # for re-assigned labels
            relabeldel_training_targets = np.zeros(len(training_data_loader.dataset.targets), dtype=np.int64)
            relabeldel_val_targets = np.zeros(len(validation_data_loader.dataset.targets), dtype=np.int64)

            training_mean_weights = []
            validation_mean_weights = []

            d = training_fax.size()[1]
            training_labels = np.array(training_data_loader.dataset.targets)
            validation_labels = np.array(validation_data_loader.dataset.targets)
            training_fax_np = training_fax.to("cpu").numpy()
            validation_fax_np = validation_fax.to("cpu").numpy()

            for i in range(num_classes):
                model = Kmeans(d, k=num_classes_per_class, niter=1000, nredo=5, verbose=True,
                               spherical=False, seed=seed + i)
                ids = training_labels == i
                model.train(training_fax_np[ids])  # perform clustering on the training samples whose labels are same
                new_labels = model.index.search(training_fax_np[ids], 1)[1].reshape(-1)  # get new labels
                new_labels += num_classes_per_class * i
                relabeldel_training_targets[ids] = new_labels
                for v in model.centroids:
                    training_mean_weights.append(v)

                ids = validation_labels == i
                new_labels = model.index.search(validation_fax_np[ids], 1)[1].reshape(-1)  # get new labels
                new_labels += num_classes_per_class * i
                relabeldel_val_targets[ids] = new_labels
                for new_label in range(num_classes_per_class):
                    validation_mean_weights.append(np.mean(validation_fax_np[ids][new_labels == new_label], axis=0))

            training_data_loader.dataset.targets = relabeldel_training_targets
            validation_data_loader.dataset.targets = relabeldel_val_targets
            training_mean_weights = torch.tensor(np.stack(training_mean_weights)).to(device)
            validation_mean_weights = torch.tensor(np.stack(validation_mean_weights)).to(device)

    return training_fax, validation_fax, training_mean_weights, validation_mean_weights


def eval_bound(
        model: ContrastiveFastText, data_loader: DataLoader, device: torch.device,
        epochs: int, mean_weights: torch.FloatTensor, simclr_loss_function: NT_Xent, num_classes: int, logger,
        calculate_centroids_accuracy: bool = True, log_interval: int = 10
) -> dict:
    num_iters_per_epoch = len(data_loader)
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

            for list_x_batches_with_offsets, mean_zs, labels in data_loader:
                mean_zs = mean_zs.to(device)
                labels = labels.to(device)

                zs_list = [model(xs.to(device), offsets.to(device)) for xs, offsets in
                           list_x_batches_with_offsets]  # (2, N, D)

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

    return results


@hydra.main(config_path="conf", config_name="nlp_bound_config")
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
    rnd = np.random.RandomState(seed)

    use_cuda = cfg["experiment"]["use_cuda"] and torch.cuda.is_available()

    if use_cuda:
        device_id = cfg["experiment"]["gpu_id"] % torch.cuda.device_count()
        device = torch.device(device_id)
    else:
        device = torch.device("cpu")

    logger_line = "Using {}".format(device)
    logger.info(logger_line)

    # initialise data loaders
    num_workers = cfg["experiment"]["num_workers"]

    weights_path = Path(cfg["experiment"]["target_weight_file"])
    self_sup_config_path = weights_path.parent / ".hydra" / "config.yaml"
    with open(self_sup_config_path) as f:
        self_sup_conf = yaml.load(f, Loader=yaml.FullLoader)

    # data loaders for centroids classifier (mu_c)
    batch_size = self_sup_conf["experiment"]["batches"]
    # initialise data loaders
    training_dataset, validation_dataset = ag_news.get_train_val_datasets(
        root=Path.home() / "pytorch_datasets",
        min_freq=cfg["dataset"]["min_freq"],
    )
    vocab_size = training_dataset.vocab_size

    mask_ratio = self_sup_conf["dataset"]["mask_ratio"]
    aug_type = self_sup_conf["dataset"]["augmentation_type"]
    if aug_type == "erase":
        replace_data = None
    else:
        replace_data = np.load(self_sup_conf["dataset"]["replace_data"])
        assert len(replace_data) == vocab_size

    # both datasets are not shuffled and `drop_last=False`
    training_data_loader = DataLoader(dataset=training_dataset, shuffle=False, num_workers=num_workers,
                                      batch_size=1024, pin_memory=True, drop_last=False,
                                      collate_fn=ag_news.CollateSelfSupervised(mask_ratio, replace_data, rnd, aug_type))
    validation_data_loader = DataLoader(dataset=validation_dataset, shuffle=False, num_workers=num_workers,
                                        batch_size=1024, pin_memory=True, drop_last=False,
                                        collate_fn=ag_news.CollateSelfSupervised(mask_ratio, replace_data, rnd,
                                                                                 aug_type))

    # load pre-trained model
    weight_name = weights_path.name

    model = ContrastiveFastText(
        num_embeddings=vocab_size,
        embedding_dim=self_sup_conf["architecture"]["embedding_dim"],
        num_last_hidden_units=self_sup_conf["architecture"]["embedding_dim"],
        with_projection_head=True
    ).to(device)

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
        model, training_data_loader, validation_data_loader, device, num_augmentations,
        num_classes_per_class=cfg["experiment"]["num_classes_per_class"],
        dim=self_sup_conf["architecture"]["embedding_dim"],
        seed=seed
    )
    # update relabeled resutls
    validation_dataset.targets = validation_data_loader.dataset.targets
    num_classes = len(np.unique(validation_dataset.targets))

    np.save("training_fax", training_fax.to("cpu").numpy())
    np.save("validation_fax", validation_fax.to("cpu").numpy())
    np.save("training_mean_weights", training_mean_weights.to("cpu").numpy())
    np.save("validation_mean_weights", validation_mean_weights.to("cpu").numpy())
    del training_fax, validation_mean_weights

    simclr_loss_function = NT_Xent(
        temperature=self_sup_conf["loss"]["temperature"], device=device, reduction="none"
    )

    # create dataset and its dataloader for bound analysis.

    # transform is applied to only images -- not `fax`.
    validation_dataset = BoundAGNEWS(fax=validation_fax.cpu().numpy(), int_ag_news=validation_dataset)

    # shuffle and use `drop_last` to make the size of mini-batches consistent
    validation_data_loader = DataLoader(dataset=validation_dataset, shuffle=True, num_workers=num_workers,
                                        batch_size=batch_size, pin_memory=True, drop_last=True,
                                        collate_fn=ag_news.CollateBound(mask_ratio, replace_data, rnd, aug_type))

    results = {}

    logger.info("Computing train x val bound")
    _results = eval_bound(model=model, data_loader=validation_data_loader, device=device, epochs=epochs,
                          mean_weights=training_mean_weights, simclr_loss_function=simclr_loss_function,
                          num_classes=num_classes, logger=logger, calculate_centroids_accuracy=True)

    # compute centroid classifier's supervised performance for reference.
    # almost same as ag_news.collate_eval_batch, but dataset returns fax via `item` method.
    # this method ignore it.
    def collate_eval_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        mainly from
        https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#generate-data-batch-and-iterator
        """

        label_list, text_list, offsets = [], [], [0]
        for (int_words, _, label) in batch:
            label_list.append(label)
            text_list.append(int_words)
            offsets.append(len(int_words))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.tensor(np.concatenate(text_list))
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

        return text_list, label_list, offsets

    validation_data_loader = DataLoader(dataset=validation_dataset, shuffle=False, num_workers=num_workers,
                                        batch_size=1024, pin_memory=True, drop_last=False,
                                        collate_fn=collate_eval_batch)
    correct = 0.

    with torch.no_grad():

        for list_x_batches, labels, offsets in validation_data_loader:

            labels = labels.to(device)

            unnormalized_features = model(list_x_batches.to(device), offsets.to(device))
            zs = torch.nn.functional.normalize(unnormalized_features, p=2, dim=-1)
            output = zs.matmul(training_mean_weights.t())

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # revert label transform
            pred = pred // cfg["experiment"]["num_classes_per_class"]
            labels = labels // cfg["experiment"]["num_classes_per_class"]

            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = correct / len(validation_data_loader.dataset) * 100.
    _results["centroid_accuracy"] = accuracy

    results["training-validation"] = {metric: value for metric, value in _results.items()}

    fname = cfg["experiment"]["bound_json_fname"]

    with open(fname, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
