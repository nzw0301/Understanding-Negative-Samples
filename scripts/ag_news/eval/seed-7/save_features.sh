#!/bin/sh

parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python -m nlp_save_features experiment.target_weight_file={1} experiment.gpu_id={#} experiment.use_projection_head={2} experiment.seed=7" :::: ../scripts/ag_news/eval/seed-7/best_weights.txt ::: true false
