#!/bin/sh



parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_self_supervised.py experiment.seed=13 experiment.batches={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type=erase experiment.gpu_id={#}" ::: 16 ::: 1.0 0.1 ::: true false
parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_self_supervised.py experiment.seed=13 experiment.batches={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type=erase experiment.gpu_id={#}" ::: 32 ::: 1.0 0.1 ::: true false
parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_self_supervised.py experiment.seed=13 experiment.batches={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type=erase experiment.gpu_id={#}" ::: 64 ::: 1.0 0.1 ::: true false
parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_self_supervised.py experiment.seed=13 experiment.batches={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type=erase experiment.gpu_id={#}" ::: 128 ::: 1.0 0.1 ::: true false
parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_self_supervised.py experiment.seed=13 experiment.batches={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type=erase experiment.gpu_id={#}" ::: 256 ::: 1.0 0.1 ::: true false
parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_self_supervised.py experiment.seed=13 experiment.batches={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type=erase experiment.gpu_id={#}" ::: 512 ::: 1.0 0.1 ::: true false
