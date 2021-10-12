import json
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data import ag_news
from src.model import ContrastiveFastText, LinearClassifier


def calculate_accuracies_loss(classifier, encoder: ContrastiveFastText, data_loader: DataLoader, device: torch.device,
                              normalized=False) -> tuple:
    """
    Auxiliary function to calculate accuracies and loss.
    :param classifier: Instance of classifier. Either linear or nonlinear.
    :param encoder: ContrastiveModel to extract feature representation.
    :param data_loader: Data loader for a downstream task.
    :param device: PyTorch's device instance.
    :param normalized: Flag whether feature is normalised or not.
    :return: Tuple of num_correct, the number of top_k_corrects, and sum of loss.
    """

    classifier.eval()
    total_loss = 0.
    top_1_correct = 0

    with torch.no_grad():
        for x, y, offsets in data_loader:
            y = y.to(device)
            rep = encoder(x.to(device), offsets.to(device))

            if normalized:
                rep = torch.nn.functional.normalize(rep, p=2, dim=1)

            outputs = classifier(rep)
            total_loss += torch.nn.functional.cross_entropy(outputs, y, reduction="sum").item()

            pred_top_k = torch.topk(outputs, dim=1, k=2)[1]
            pred_top_1 = pred_top_k[:, 0]

            top_1_correct += pred_top_1.eq(y.view_as(pred_top_1)).sum().item()

    if data_loader.drop_last:
        num_samples = len(data_loader) * data_loader.batch_size
    else:
        num_samples = len(data_loader.dataset)

    return top_1_correct / num_samples, total_loss / num_samples


def learnable_eval(
        cfg: OmegaConf, classifier, encoder: ContrastiveFastText, training_data_loader: DataLoader,
        val_data_loader: DataLoader, device: torch.device
) -> tuple:
    """
    :param cfg: Hydra's config instance.
    :param classifier: Instance of classifier with learnable parameters.
    :param encoder: feature extractor trained on self-supervised method.
    :param training_data_loader: Training data loader for a downstream task.
    :param val_data_loader: Validation data loader for a downstream task.

    :return: tuple of train acc, train top-k acc, train loss, val acc, val top-k acc, and val loss.
    """

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
    val_accuracies = []
    train_losses = []
    val_losses = []
    highest_val_acc = 0.

    encoder.eval()
    for epoch in range(1, epochs + 1):
        classifier.train()

        for x, y, offsets in training_data_loader:
            optimizer.zero_grad()

            with torch.no_grad():
                rep = encoder(x.to(device), offsets.to(device))
                if normalized:
                    rep = torch.nn.functional.normalize(rep, p=2, dim=1)

            # t is not used
            outputs = classifier(rep)

            loss = cross_entropy_loss(outputs, y.to(device))

            loss.backward()
            optimizer.step()

        cos_lr_scheduler.step()

        # train and val metrics
        train_acc, train_loss = calculate_accuracies_loss(
            classifier, encoder, training_data_loader, device, normalized=normalized
        )

        val_acc, val_loss = calculate_accuracies_loss(
            classifier, encoder, val_data_loader, device, normalized=normalized
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        current_progress = epoch / epochs
        logging.info(f"Epoch:{epoch}/{epochs} progress:{current_progress:.2f}, train acc.:{train_acc * 100.:.1f} "
                     f"val acc.:{val_acc * 100.:.1f} lr:{current_lr:.4f}")

        if highest_val_acc < val_acc:
            # save best linear classifier on validation dataset
            highest_val_acc = val_acc

            # delete old checkpoint file
            if "save_fname" in locals():
                if os.path.exists(save_fname):
                    os.remove(save_fname)

            save_fname = "epoch_{}-{}".format(epoch, cfg["experiment"]["output_model_name"])
            torch.save(classifier.state_dict(), save_fname)

    return train_accuracies, train_losses, val_accuracies, val_losses


@hydra.main(config_path="conf", config_name="nlp_linear_eval_config")
def main(cfg: OmegaConf):
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

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device_id = cfg["experiment"]["gpu_id"] % torch.cuda.device_count()
        device = torch.device(device_id)
    else:
        device = torch.device("cpu")

    logger_line = "Using {}".format(device)
    logger.info(logger_line)

    # load pre-trained model
    weights_path = Path(cfg["experiment"]["target_weight_file"])
    weight_name = weights_path.name
    self_sup_config_path = weights_path.parent / ".hydra" / "config.yaml"
    with open(self_sup_config_path) as f:
        self_sup_conf = yaml.load(f, Loader=yaml.FullLoader)

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

    training_data_loader = DataLoader(
        training_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=True,
        collate_fn=ag_news.CollateSupervised(mask_ratio, replace_data, np.random.RandomState(seed), aug_type),
        drop_last=True
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=cfg["experiment"]["batches"],
        shuffle=False,
        collate_fn=ag_news.collate_eval_batch,
        drop_last=False
    )

    num_classes = len(np.unique(validation_dataset.targets))

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

    num_last_units = self_sup_conf["architecture"]["embedding_dim"]
    # get the dimensionality of the representation
    if not cfg["experiment"]["use_projection_head"]:
        model.g = torch.nn.Identity()

    logger.info("#train: {}, #val: {}".format(len(training_dataset), len(validation_dataset)))
    logger.info("Evaluation by using {}".format(weight_name))

    # initialise linear classifier
    # NOTE: the weights are not normalize
    classifier = LinearClassifier(num_last_units, num_classes, normalize=False).to(device)

    # execute linear evaluation protocol
    train_accuracies, train_losses, val_accuracies, val_losses = learnable_eval(
        cfg, classifier, model, training_data_loader, validation_data_loader, device
    )

    classification_results = {}
    classification_results[weight_name] = {
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lowest_val_loss": min(val_losses),
        "highest_val_acc": max(val_accuracies),
    }

    logger.info("train acc: {}, val acc: {}".format(max(train_accuracies), max(val_accuracies)))

    fname = cfg["experiment"]["classification_results_json_fname"]

    with open(fname, "w") as f:
        json.dump(classification_results, f)


if __name__ == "__main__":
    main()
