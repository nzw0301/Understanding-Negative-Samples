#!/bin/sh

parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python -m save_features experiment.target_weight_file={1} experiment.gpu_id={#} experiment.use_projection_head=true dataset=cifar10 experiment.seed=11" :::: ../scripts/cifar10/eval/seed-11/all_weights.txt

parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python -m save_features experiment.target_weight_file={1} experiment.gpu_id={#} experiment.use_projection_head=false dataset=cifar10 experiment.seed=11" :::: ../scripts/cifar10/eval/seed-11/all_weights.txt
