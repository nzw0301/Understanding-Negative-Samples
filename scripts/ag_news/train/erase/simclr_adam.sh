#!/bin/sh



parallel -j 4 "export CUDA_VISIBLE_DEVICES=0,1,2,3; python nlp_self_supervised.py optimizer=nlp/simclr_adam experiment.batches={1} optimizer.lr={2} optimizer.linear_schedule={3} experiment.gpu_id={#}" ::: 512 256 32 16 64 128 ::: 1.0 0.1 ::: true false
