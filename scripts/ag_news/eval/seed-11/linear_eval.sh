#!/bin/sh

parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_linear_eval.py experiment.target_weight_file={1} experiment.use_projection_head={2} experiment.normalize={2} experiment.gpu_id={#} experiment.seed=11" :::: ../scripts/ag_news/eval/seed-11/best_weights.txt ::: true false
