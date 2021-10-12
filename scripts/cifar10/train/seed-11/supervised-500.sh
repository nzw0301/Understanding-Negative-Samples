#!/bin/sh

python launch.py --nproc_per_node=4 supervised.py dataset=cifar10 experiment.seed=11
