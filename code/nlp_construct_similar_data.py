import io
from pathlib import Path

import numba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.data.ag_news import get_train_val_datasets


def load_vectors(fname, words_set: set):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

    words = []
    data = []
    for line in fin:
        word, vec = line.rstrip().split(' ', 1)
        if word in words_set:
            words.append(word)
            data.append(list(map(float, vec.split())))
    return words, np.array(data)


def main():
    min_freq = 5
    top_k = 5

    # initialize data loaders
    training_dataset, _ = get_train_val_datasets(
        root=Path.home() / "pytorch_datasets",
        min_freq=min_freq,
    )

    fname = "crawl-300d-2M.vec"
    fasstext_vocab, vectors = load_vectors(
        fname,
        set(training_dataset.vocab.itos)
    )

    sim = cosine_similarity(vectors)

    @numba.jit(nopython=True, parallel=True)
    def get_top_k_per_word_in_fasttext(sim, top_k):
        N = len(sim)
        sim = sim - (2 * np.eye(N))  # do not select its own
        returned_values = np.zeros((N, top_k))
        for i in range(N):
            returned_values[i] = np.argsort(sim[i])[-top_k:]
        return returned_values

    top_k_list = get_top_k_per_word_in_fasttext(sim, top_k)
    top_k_list = top_k_list.astype(int)

    fasstext_stoi = {v: i for i, v in enumerate(fasstext_vocab)}
    data = []
    for i, word in enumerate(training_dataset.vocab.itos):
        if word in fasstext_stoi:
            words = [training_dataset.vocab.stoi[fasstext_vocab[ft_i]] for ft_i in top_k_list[fasstext_stoi[word]]]
        else:
            words = [i] * top_k
        data.append(words)
    np.save("ag_replace_ids", np.array(data))


if __name__ == "__main__":
    """
    To run this code,
    `python nlp_construct_simiar_data.py`
    """
    main()
