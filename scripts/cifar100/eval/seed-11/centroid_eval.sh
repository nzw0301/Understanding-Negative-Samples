#!/bin/sh

parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python -m centroid_eval experiment.use_projection_head=true experiment.target_weight_file={1} experiment.gpu_id={#} dataset=cifar100 experiment.seed=11" :::: ../scripts/cifar100/eval/seed-11/all_weights.txt
