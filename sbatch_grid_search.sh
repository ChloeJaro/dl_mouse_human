#!/bin/bash -l

iterations=10

for i in $(seq 1 "$iterations")
    do
    sbatch -p gpu_short --gres gpu:1 \
    #singularity_python_run.sh \
    pipeline.sh \
    exps/e1_ae/exp/tune_model.py \
    ++tag=tune/run_$i \ 
    ++seed=$i \
    
    done