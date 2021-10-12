#!/bin/sh

python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=128 experiment.seed=11
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=256 experiment.seed=11
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=384 experiment.seed=11
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=512 experiment.seed=11
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=640 experiment.seed=11
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=768 experiment.seed=11
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=896 experiment.seed=11
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=1024 experiment.seed=11
