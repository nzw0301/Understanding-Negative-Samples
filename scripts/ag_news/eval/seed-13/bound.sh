#!/bin/sh

parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python -m nlp_bound experiment.target_weight_file={1} experiment.gpu_id={#} experiment.seed=13" :::: ../scripts/ag_news/eval/seed-13/best_weights.txt

parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python -m nlp_bound experiment.target_weight_file={1} experiment.gpu_id={#} experiment.num_classes_per_class=2 experiment.seed=13" :::: ../scripts/ag_news/eval/seed-13/best_weights.txt
