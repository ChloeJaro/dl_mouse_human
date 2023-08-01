#!/bin/bash -l

sbatch -p gpu_short --gres gpu:1 \
#singularity_python_run.sh \
pipeline.sh
exps/e1_ae/exp/tune_model.py \
++tag=tune/run_$i \ 
