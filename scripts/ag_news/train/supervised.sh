#!/bin/sh



#parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_supervised.py experiment.seed={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type={4} experiment.gpu_id={#}" ::: 7 ::: 1.0 0.1 ::: true false ::: replace
#parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_supervised.py experiment.seed={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type={4} experiment.gpu_id={#}" ::: 7 ::: 1.0 0.1 ::: true false ::: erase
parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_supervised.py experiment.seed={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type={4} experiment.gpu_id={#}" ::: 11 ::: 1.0 0.1 ::: true false ::: replace
parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_supervised.py experiment.seed={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type={4} experiment.gpu_id={#}" ::: 11 ::: 1.0 0.1 ::: true false ::: erase
parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_supervised.py experiment.seed={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type={4} experiment.gpu_id={#}" ::: 13 ::: 1.0 0.1 ::: true false ::: replace
parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_supervised.py experiment.seed={1} optimizer.lr={2} optimizer.linear_schedule={3} dataset.augmentation_type={4} experiment.gpu_id={#}" ::: 13 ::: 1.0 0.1 ::: true false ::: erase
