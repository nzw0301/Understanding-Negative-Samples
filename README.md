# Codes of Understanding Negative Samples in Instance Discriminative Self-supervised Representation Learning

## Create experimental env

- Linux machine with four GPUs
- `Conda`
- Python's dependencies: [this file](./spec-file.txt):

```bash
conda create --name research --file spec-file.txt
conda activate research
```

### External dependency

#### Apex install

```bash
git clone git@github.com:NVIDIA/apex.git
cd apex
git checkout 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a  # to fix the exact library version
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Detailed version of PyTorch

```bash
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

## Preparation

In NLP experiments, we need similar words constructed from fasttext's pre-trained word embeddings.

```bash
cd code
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
python nlp_construct_simiar_data.py
```

Please fill the value of `replace_data` in [`conf/dataset/ag_news.yaml`](./conf/dataset/ag_news.yaml) with the generated file's path. For example,

```yaml
replace_data: /home/nzw/code/ag_replace_ids.npy
```

## Training

Please run __content of all scripts__ in `./scripts/**/train/` under [`code`](./code).

## Evaluation

After training, please run [`./gather_weights.py`](./gather_weights.py) to generate text files for evaluation.

Please run __content__ of all scripts in `./scripts/**/eval/` under [`code`](./code) as well.

For the AG news dataset, please run [`code/notebooks/filter_ag_news.ipynb`](code/notebooks/filter_ag_news.ipynb) __after__ evaluation of mean classifier __before__ the other evaluation scripts such as linear classifier and bound computation.

To obtain all figures and tables in the paper, you run notebooks in [`code/notebooks/`](./code/notebooks). The codes save generated figures and tables into [`./doc/figs`](./doc/figs) and [`./doc/tabs`](./doc/tabs), respectively.

- [`code/notebooks/bound.ipynb`](code/notebooks/bound.ipynb) generates Figure 1, the content of Table 2, and Tables 4 and 5.
- [`code/notebooks/coupon.ipynb`](code/notebooks/coupon.ipynb) generates Table 3.
- [`code/notebooks/collision.ipynb`](code/notebooks/collision.ipynb) computes upper bound of the collision term used in `code/notebooks/bound.ipynb`.
- [`code/notebooks/collosion_analysis.ipynb`](code/notebooks/collosion_analysis.ipynb) generates Figures 3, 4, and 5.

## Related resources

- [Openreview](https://openreview.net/forum?id=pZ5X_svdPQ)
- [arXiv](https://arxiv.org/abs/2102.06866)
- [Slides](https://speakerdeck.com/nzw0301/understanding-negative-samples-in-instance-discriminative-self-supervised-representation-learning)
- [NeurIPS Poster](https://drive.google.com/file/d/1uGDY2YrneNF2bFgjh1yMlDeUpVk1GQRL/view)
- [Video](https://nips.cc/virtual/2021/poster/27316)
- [Video in Japanese](https://www.youtube.com/watch?v=nK46OGjoosQ)

## Reference
```
@inproceedings{NS2021,
    title     = {Understanding Negative Samples in Instance Discriminative Self-supervised Representation Learning},
    author    = {Kento Nozawa, Issei Sato},
    year      = {2021},
    booktitle = {NeurIPS},
    pages     = {5784--5797},
}
```
