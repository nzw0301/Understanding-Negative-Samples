#!/bin/sh

target_weights_list=../scripts/cifar10/eval/seed-13/all_weights.txt

cat $target_weights_list | while read f
do
  python launch.py --nproc_per_node=4 linear_eval.py dataset=cifar10 experiment.seed=13 experiment.use_projection_head=true experiment.normalize=true experiment.target_weight_file=${f}
done
