#!/bin/sh

target_weights_list=../scripts/cifar100/eval/seed-13/all_weights.txt

cat $target_weights_list | while read f
do
  python launch.py --nproc_per_node=4 linear_eval.py dataset=cifar100 experiment.seed=13 experiment.use_projection_head=false experiment.normalize=false experiment.target_weight_file=${f}
done
