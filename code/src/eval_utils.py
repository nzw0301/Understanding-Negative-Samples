import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .model import CentroidClassifier
from .model import ContrastiveModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.terminator = ""
logger.addHandler(stream_handler)


def centroid_eval(
        data_loader: DataLoader, device: torch.device, classifier: CentroidClassifier, top_k: int = 5
) -> tuple:
    """
    :param data_loader: DataLoader of downstream task.
    :param device: PyTorch's device instance.
    :param classifier: Instance of MeanClassifier.
    :param top_k: The number of top-k to calculate accuracy.
    :return: Tuple of top-1 accuracy and top-k accuracy.
    """

    num_samples = len(data_loader.dataset)
    top_1_correct = 0
    top_k_correct = 0

    classifier.eval()
    with torch.no_grad():
        for x, y in data_loader:
            y = y.to(device)
            pred_top_k = torch.topk(classifier(x.to(device)), dim=1, k=top_k)[1]
            pred_top_1 = pred_top_k[:, 0]

            top_1_correct += pred_top_1.eq(y.view_as(pred_top_1)).sum().item()
            if top_k > 1:
                top_k_correct += (pred_top_k == y.view(len(y), 1)).sum().item()

    return top_1_correct / num_samples, top_k_correct / num_samples


def convert_vectors(
        data_loader: torch.utils.data.DataLoader, model: ContrastiveModel, device: torch.device, normalized: bool
) -> tuple:
    """
    Convert experiment to feature representations.
    :param data_loader: Tata loader for raw experiment.
    :param model: Pre-trained model.
    :param device: PyTorch's device instance.
    :param normalized: Whether normalize the feature representation or not.

    :return: Tuple of tensors: features and labels.
    """

    new_X = []
    new_y = []
    model.eval()

    with torch.no_grad():
        for x_batches, y_batches in data_loader:
            x_batches = x_batches.to(device)
            fs = model(x_batches)
            if normalized:
                fs = torch.nn.functional.normalize(fs, p=2, dim=1)
            new_X.append(fs)
            new_y.append(y_batches)

    X = torch.cat(new_X)
    y = torch.cat(new_y)

    return X, y


def calculate_accuracies_loss(classifier, encoder: ContrastiveModel, data_loader: DataLoader, device: torch.device,
                              top_k: int = 5, normalized=False) -> tuple:
    """
    Auxiliary function to calculate accuracies and loss.
    :param classifier: Instance of classifier. Either linear or nonlinear.
    :param encoder: ContrastiveModel to extract feature representation.
    :param data_loader: Data loader for a downstream task.
    :param device: PyTorch's device instance.
    :param top_k: The number of top-k to calculate accuracy. Note `top_k <= 1` is same to top1.
    :param normalized: Flag whether feature is normalised or not.
    :return: Tuple of num_correct, the number of top_k_corrects, and sum of loss.
    """

    classifier.eval()
    total_loss = 0.
    top_1_correct = 0
    top_k_correct = 0

    with torch.no_grad():
        for x, y in data_loader:
            y = y.to(device)
            rep = encoder(x.to(device))

            if normalized:
                rep = torch.nn.functional.normalize(rep, p=2, dim=1)

            outputs = classifier(rep)
            total_loss += torch.nn.functional.cross_entropy(outputs, y, reduction="sum")

            pred_top_k = torch.topk(outputs, dim=1, k=top_k)[1]
            pred_top_1 = pred_top_k[:, 0]

            top_1_correct += pred_top_1.eq(y.view_as(pred_top_1)).sum()
            if top_k > 1:
                top_k_correct += (pred_top_k == y.view(len(y), 1)).sum()

    if top_k == 1:
        top_k_correct = top_1_correct

    return top_1_correct, top_k_correct, total_loss


def learnable_eval(
        cfg: OmegaConf, classifier, encoder: ContrastiveModel, training_data_loader: DataLoader,
        val_data_loader: DataLoader, top_k: int,
) -> tuple:
    """
    :param cfg: Hydra's config instance.
    :param classifier: Instance of classifier with learnable parameters.
    :param encoder: feature extractor trained on self-supervised method.
    :param training_data_loader: Training data loader for a downstream task.
    :param val_data_loader: Validation data loader for a downstream task.
    :param top_k: The number of top-k for evaluation.

    :return: tuple of train acc, train top-k acc, train loss, val acc, val top-k acc, and val loss.
    """

    local_rank = cfg["distributed"]["local_rank"]
    epochs = cfg["experiment"]["epochs"]
    normalized = cfg["experiment"]["normalize"]

    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        params=classifier.parameters(),
        lr=cfg["optimizer"]["lr"],
        momentum=cfg["optimizer"]["momentum"],
        nesterov=True,
        weight_decay=cfg["optimizer"]["decay"]
    )

    cos_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.)

    train_accuracies = []
    train_top_k_accuracies = []
    val_accuracies = []
    val_top_k_accuracies = []
    train_losses = []
    val_losses = []

    num_train = len(training_data_loader.dataset)
    num_val = len(val_data_loader.dataset)

    highest_val_acc = 0.

    encoder.eval()
    for epoch in range(1, epochs + 1):
        classifier.train()
        training_data_loader.sampler.set_epoch(epoch)  # to shuffle dataset

        for x, y in training_data_loader:
            optimizer.zero_grad()

            with torch.no_grad():
                rep = encoder(x.to(local_rank))
                if normalized:
                    rep = torch.nn.functional.normalize(rep, p=2, dim=1)

            # t is not used
            outputs = classifier(rep)

            loss = cross_entropy_loss(outputs, y.to(local_rank))

            loss.backward()
            optimizer.step()


        cos_lr_scheduler.step()

        # train and val metrics
        train_acc, train_top_k_acc, train_loss = calculate_accuracies_loss(
            classifier, encoder, training_data_loader, local_rank, top_k=top_k, normalized=normalized
        )

        torch.distributed.barrier()
        torch.distributed.reduce(train_acc, dst=0)
        torch.distributed.reduce(train_top_k_acc, dst=0)
        torch.distributed.reduce(train_loss, dst=0)

        val_acc, val_top_k_acc, val_loss = calculate_accuracies_loss(
            classifier, encoder, val_data_loader, local_rank, top_k=top_k, normalized=normalized
        )

        torch.distributed.barrier()
        torch.distributed.reduce(val_acc, dst=0)
        torch.distributed.reduce(val_top_k_acc, dst=0)
        torch.distributed.reduce(val_loss, dst=0)

        if local_rank == 0:
            # NOTE: since drop=True, num_train is not approximate value
            train_losses.append(train_loss.item() / num_train)
            train_acc = train_acc.item() / num_train
            train_accuracies.append(train_acc)
            train_top_k_accuracies.append(train_top_k_acc.item() / num_train)

            val_losses.append(val_loss.item() / num_val)
            val_acc = val_acc.item() / num_val
            val_accuracies.append(val_acc)
            val_top_k_accuracies.append(val_top_k_acc.item() / num_val)

            current_lr = optimizer.param_groups[0]["lr"]
            current_progress = epoch / epochs
            logging.info(f"Epoch:{epoch}/{epochs} progress:{current_progress:.2f}, train acc.:{train_acc * 100.:.1f} "
                         f"val acc.:{val_acc * 100.:.1f} lr:{current_lr:.4f}")

        if highest_val_acc < val_acc and local_rank == 0:
            # save best linear classifier on validation dataset
            highest_val_acc = val_acc

            # delete old checkpoint file
            if "save_fname" in locals():
                if os.path.exists(save_fname):
                    os.remove(save_fname)

            save_fname = "epoch_{}-{}".format(epoch, cfg["experiment"]["output_model_name"])
            torch.save(classifier.state_dict(), save_fname)

    return train_accuracies, train_top_k_accuracies, train_losses, val_accuracies, val_top_k_accuracies, val_losses


def make_two_vector_for_confusion_matrix(
        data_loader: DataLoader, device: torch.device, classifier: CentroidClassifier, encoder: ContrastiveModel = None,
        normalized: bool = True
) -> tuple:
    """
    Create two categorical vectors to plot confusion matrix for classifier predictions.

    :param data_loader: DataLoader of downstream task.
    :param device: PyTorch's device instance.
    :param classifier: Instance of MeanClassifier.
    :param encoder: pre-trained encoder
    :param normalized: whether or not to perform normalization just after encoder
    :return: Tuple of `np.ndarray`. The first one contains true labels and the second one contains predicted labels.
    """

    classifier.eval()
    if encoder is not None:
        encoder.eval()

    y_true_vec = []
    y_pred_vec = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            if encoder is not None:
                x = encoder(x)
                if normalized:
                    x = torch.nn.functional.normalize(x, p=2, dim=1)

            pred = torch.argmax(classifier(x), dim=1)  # 1-D tensor

            # save ys
            y_true_vec.extend(y.tolist())
            y_pred_vec.extend(pred.tolist())

    return np.array(y_true_vec), np.array(y_pred_vec)
