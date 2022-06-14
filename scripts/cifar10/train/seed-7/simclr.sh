#!/bin/sh

python launch.py --nproc_per_node=4 self_supervised.py dataset=cifar10 experiment.batches=32 experiment.seed=7
python launch.py --nproc_per_node=4 self_supervised.py dataset=cifar10 experiment.batches=64 experiment.seed=7
python launch.py --nproc_per_node=4 self_supervised.py dataset=cifar10 experiment.batches=128 experiment.seed=7
python launch.py --nproc_per_node=4 self_supervised.py dataset=cifar10 experiment.batches=256 experiment.seed=7
python launch.py --nproc_per_node=4 self_supervised.py dataset=cifar10 experiment.batches=512 experiment.seed=7
