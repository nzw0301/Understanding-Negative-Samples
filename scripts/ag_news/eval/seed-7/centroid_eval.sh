#!/bin/sh

parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python -m nlp_centroid_eval experiment.target_weight_file={1} experiment.gpu_id={#} experiment.seed=7" :::: ../scripts/ag_news/eval/seed-7/all_weights.txt
parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python -m nlp_centroid_eval experiment.use_projection_head=false experiment.target_weight_file={1} experiment.gpu_id={#} experiment.seed=7" :::: ../scripts/ag_news/eval/seed-7/all_supervised_weights.txt
