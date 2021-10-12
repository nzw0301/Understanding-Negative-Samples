from collections import Counter
from pathlib import Path
from typing import Callable, Tuple, Optional

import numpy as np
import requests
import torch.utils.data
import torchtext
import torchtext.vocab
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


class AGNEWS(torch.utils.data.Dataset):

    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> Tuple[str, int]:
        return self.data[index], self.targets[index]


class IntAGNEWS(torch.utils.data.Dataset):

    def __init__(self, dataset: AGNEWS, fn: Callable, vocab: torchtext.vocab.Vocab) -> None:
        self.data = [np.array(fn(x)) for x in dataset.data]
        self.targets = dataset.targets
        self.vocab = vocab

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def _revert_dataset(dataset: torchtext.datasets.AG_NEWS) -> AGNEWS:
    vocab = dataset.get_vocab()
    new_data = []
    new_targets = []
    for y, words in dataset:
        normalized_line = " ".join([vocab.itos[x] for x in words])

        new_data.append(normalized_line)
        new_targets.append(y)

    return AGNEWS(new_data, new_targets)


def get_train_val_datasets(
        root="~/pytorch_datasets",
        min_freq: int = 5,
) -> Tuple[IntAGNEWS, IntAGNEWS]:
    """
    :param root: Path to save data.
    :param min_freq: The minimal frequency of words used in the training.

    :return: Tuple of (train dataset, val dataset).
    """

    # since 0.9.0, torchtext.dataset's implementation largely improves,
    # where each sample is tuple of int and raw text.
    # For previous versions, torchtext's dataset is tuple of int (label) and tensor of int (sequence of words).
    # therefore, we convert int tensor to raw text (with normalize) for consistent interface.
    if int(torchtext.__version__.split(".")[1]) >= 9:
        ValueError("No test")

    root = Path(root)
    try:
        train, val = torchtext.datasets.AG_NEWS(root=root)

        train_set = _revert_dataset(train)
        val_set = _revert_dataset(val)

    except requests.exceptions.ConnectionError:
        from torchtext.datasets.text_classification import _csv_iterator

        def create_dataset_from_csv_iter(path):
            new_data, new_targets = [], []
            for label, generator in _csv_iterator(path, ngrams=1, yield_cls=True):
                new_data.append(" ".join(generator))
                new_targets.append(label)
            return AGNEWS(new_data, new_targets)

        dataset_name = "ag_news_csv"

        train_set = create_dataset_from_csv_iter(root / dataset_name / "train.csv")
        val_set = create_dataset_from_csv_iter(root / dataset_name / "test.csv")

    tokenizer = get_tokenizer("basic_english")

    counter = Counter()
    for line, _ in train_set:
        counter.update(tokenizer(line))

    vocab = Vocab(counter, specials=["<unk>"], min_freq=min_freq, specials_first=False)

    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

    train_set = IntAGNEWS(train_set, text_pipeline, vocab)
    val_set = IntAGNEWS(val_set, text_pipeline, vocab)

    return train_set, val_set


def collate_eval_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mainly from
    https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#generate-data-batch-and-iterator
    """

    label_list, text_list, offsets = [], [], [0]
    for (int_words, label) in batch:
        label_list.append(label)
        text_list.append(int_words)
        offsets.append(len(int_words))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.tensor(np.concatenate(text_list))
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    return text_list, label_list, offsets


class BaseCollate(object):
    def __init__(self, mask_ratio: float, replace_data: Optional[np.ndarray], rnd: np.random.RandomState,
                 augmentation_type: str = "replace") -> None:

        assert 0. <= mask_ratio < 1

        if augmentation_type == "replace":
            self.data_augmentation = self._replace_data_augmentation

            assert replace_data is not None

            self.replace_data = replace_data
            self.num_candidates = replace_data.shape[1]

        elif augmentation_type == "erase":
            self.data_augmentation = self._erase_data_augmentation
        else:
            raise ValueError("Invalid augmentation type: {}. Should be either `replace` or `erase`".format(augmentation_type))

        self.mask_ratio = mask_ratio
        self.rnd = rnd

    def _erase_data_augmentation(self, int_words: np.ndarray, num_views=2) -> list:
        masks = self.rnd.rand(num_views, len(int_words)) > self.mask_ratio
        augmented_samples = []
        for mask in masks:
            augmented_samples.append(int_words[mask])

        return augmented_samples

    def _replace_data_augmentation(self, int_words: np.ndarray, num_views=2) -> list:
        masks = self.rnd.rand(num_views, len(int_words)) > self.mask_ratio
        augmented_samples = []
        for mask in masks:
            # non replaced words
            processed_text = int_words[mask]

            # replacement words
            mask = ~mask
            target_word_ids = int_words[mask]
            random_indices = self.rnd.randint(low=0, high=self.num_candidates, size=(len(target_word_ids), 1))

            processed_text = np.concatenate([
                processed_text,
                np.take_along_axis(self.replace_data[target_word_ids], random_indices, 1).reshape(-1)
            ])
            augmented_samples.append(processed_text)

        return augmented_samples


class CollateSupervised(BaseCollate):
    def __init__(self, mask_ratio: float, replace_data: Optional[np.ndarray], rnd: np.random.RandomState,
                 augmentation_type: str):
        super(CollateSupervised, self).__init__(mask_ratio, replace_data, rnd, augmentation_type)

    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        mainly from
        https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#generate-data-batch-and-iterator
        """

        label_list, text_list, offsets = [], [], [0]
        for (int_words, label) in batch:
            label_list.append(label)

            processed_text = self.data_augmentation(int_words, 1)[0]

            text_list.append(processed_text)
            offsets.append(len(processed_text))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.tensor(np.concatenate(text_list))
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

        return text_list, label_list, offsets


class CollateSelfSupervised(BaseCollate):
    def __init__(self, mask_ratio: float, replace_data: Optional[np.ndarray], rnd: np.random.RandomState,
                 augmentation_type: str):
        super(CollateSelfSupervised, self).__init__(mask_ratio, replace_data, rnd, augmentation_type)

    def __call__(self, batch) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        anchor_text_list, anchor_offsets = [], [0]
        positive_text_list, positive_offsets = [], [0]

        for int_words, _ in batch:
            anchor_processed_text, positive_processed_text = self.data_augmentation(int_words, 2)

            anchor_text_list.append(anchor_processed_text)
            anchor_offsets.append(len(anchor_processed_text))

            positive_text_list.append(positive_processed_text)
            positive_offsets.append(len(positive_processed_text))

        anchor_text_list = torch.tensor(np.concatenate(anchor_text_list))
        anchor_offsets = torch.tensor(anchor_offsets[:-1]).cumsum(dim=0)

        positive_text_list = torch.tensor(np.concatenate(positive_text_list))
        positive_offsets = torch.tensor(positive_offsets[:-1]).cumsum(dim=0)

        return (anchor_text_list, anchor_offsets), (positive_text_list, positive_offsets)


class CollateBound(BaseCollate):
    def __init__(self, mask_ratio: float, replace_data: Optional[np.ndarray], rnd: np.random.RandomState,
                 augmentation_type: str):
        super(CollateBound, self).__init__(mask_ratio, replace_data, rnd, augmentation_type)

    def __call__(self, batch) -> Tuple[
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
        torch.Tensor
    ]:
        """
        mainly from
        https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#generate-data-batch-and-iterator
        """

        anchor_text_list, anchor_offsets = [], [0]
        positive_text_list, positive_offsets = [], [0]
        fax_list = []
        label_list = []

        for int_words, fax, target in batch:
            anchor_processed_text, positive_processed_text = self.data_augmentation(int_words, 2)

            anchor_text_list.append(anchor_processed_text)
            anchor_offsets.append(len(anchor_processed_text))

            positive_text_list.append(positive_processed_text)
            positive_offsets.append(len(positive_processed_text))

            fax_list.append(fax)
            label_list.append(target)

        anchor_text_list = torch.tensor(np.concatenate(anchor_text_list))
        anchor_offsets = torch.tensor(anchor_offsets[:-1]).cumsum(dim=0)

        positive_text_list = torch.tensor(np.concatenate(positive_text_list))
        positive_offsets = torch.tensor(positive_offsets[:-1]).cumsum(dim=0)
        fax_list = torch.tensor(np.stack(fax_list), dtype=torch.float32)
        label_list = torch.tensor(label_list, dtype=torch.int64)

        return ((anchor_text_list, anchor_offsets), (positive_text_list, positive_offsets)), fax_list, label_list
