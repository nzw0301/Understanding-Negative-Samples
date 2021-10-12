#!/bin/sh

python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=128
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=256
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=384
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=512
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=640
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=768
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=896
python launch.py --nproc_per_node=4 self_supervised.py experiment.batches=1024
