#!/bin/sh

parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python -m bound experiment.target_weight_file={1} experiment.gpu_id={#} dataset=cifar10 experiment.seed=7" :::: ../scripts/cifar10/eval/seed-7/all_weights.txt
