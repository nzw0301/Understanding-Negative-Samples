import pathlib
import yaml
import numpy as np
import json
import pandas as pd


def get_weight_path_in_current_system(weight_path: str) -> str:
    """
    :param weight_path: Give a weight path created for another computation node,
        replace the path the current system's path by
    :return: the current system's path.
    """
    in_result_dir = False
    path_elem = [".."]
    weight_path = weight_path.replace("//", "/")

    for elem in weight_path.split("/"):
        if not in_result_dir and elem == "results":
            in_result_dir = True
        if in_result_dir:
            path_elem.append(elem)

    return "/".join(path_elem)


def load_features(sub_sampled=False) -> dict:
    """
    :param sub_sampled: if True, use only 2,000 x num_samples for debugging.
    :return: dict instance that stores feature vectors.
    """
    datasets = ("cifar10", "cifar100")
    epoch = 500

    features = {}
    for dataset in datasets:
        features[dataset] = {}

        base_dir = pathlib.Path("../results/{}/analysis/save_unnormalised_feature/".format(dataset))

        for config_path in base_dir.glob("**/config.yaml"):

            with open(config_path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                seed = config["experiment"]["seed"]

                if config["experiment"]["use_projection_head"]:
                    extractor = "Head"
                else:
                    extractor = "Without Head"

            self_sup_path = pathlib.Path(
                get_weight_path_in_current_system(config["experiment"]["target_weight_file"])).parent
            with open(self_sup_path / ".hydra" / "config.yaml") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                num_mini_batches = config["experiment"]["batches"]

            path = config_path.parent.parent

            X_train = np.load(path / "epoch_{}-cifar.pt.feature.train.npy".format(epoch))
            y_train = np.load(path / "epoch_{}-cifar.pt.label.train.npy".format(epoch))

            X_eval = np.load(path / "epoch_{}-cifar.pt.feature.val.npy".format(epoch))
            y_eval = np.load(path / "epoch_{}-cifar.pt.label.val.npy".format(epoch))

            if extractor not in features[dataset]:
                features[dataset][extractor] = {}

            if seed not in features[dataset][extractor]:
                features[dataset][extractor][seed] = {}

            if sub_sampled:
                features[dataset][extractor][seed][num_mini_batches] = (
                    X_train[:2000],
                    y_train[:2000],
                    X_eval[:2000],
                    y_eval[:2000],
                )
            else:
                features[dataset][extractor][seed][num_mini_batches] = (
                    X_train,
                    y_train,
                    X_eval,
                    y_eval,
                )

    return features


def filter_ag() -> set:
    "obtain self-supervised weights by seed, augmentation type, and negative samples."

    rows = []
    dataset = "ag_news"
    ids = set()

    for results_path_per_classifier in pathlib.Path(f"../results/{dataset}/eval/").iterdir():
        result_path_generator = results_path_per_classifier.glob("**/results.json")
        classifier = results_path_per_classifier.name


        for p in result_path_generator:

            # load config
            with open(p.parent / ".hydra" / "config.yaml") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

                seed = config["experiment"]["seed"]

                if classifier == "linear":
                    continue

            weights = config["experiment"]["target_weight_file"]

            if "self_supervised" not in str(weights):
                continue

            self_sup_path = pathlib.Path(get_weight_path_in_current_system(config["experiment"]["target_weight_file"])).parent
            with open(self_sup_path / ".hydra" / "config.yaml") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            num_mini_batches = config["experiment"]["batches"]

            # load classification results
            with open(p) as f:
                classification_results = json.load(f)

                val_accuracies = []
                for key, v in classification_results.items():
                    if "val_acc" in v:
                        k = "val_acc"
                    else:
                        k = "highest_val_acc"

                    val_accuracies.append(v[k] * 100.)
                val_acc = max(val_accuracies)
            a = config["dataset"]["augmentation_type"]
            rows.append([seed, num_mini_batches, a, val_acc, weights])


    columns = ("seed", "neg", "aug type", "val acc", "weights")
    acc_df = pd.DataFrame(rows, columns=columns).sort_values(by=["seed", "neg", "aug type"])
    idx = acc_df.groupby(['seed', "neg", "aug type"])["val acc"].transform(max) == acc_df['val acc']
    ag_targets = set(acc_df[idx]["weights"].to_list())

    return ag_targets
