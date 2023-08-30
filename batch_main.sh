#!/bin/bash -l


iterations=4

for i in $(seq 1 "$iterations")
do
sbatch -p gpu_short --gres gpu:1 \
singularity_python_run.sh \
exps/e1_ae/exp/main.py \
++tag=test_l105_layer1000/run_$i \
++seed=$i \
++model.encoder_layers=[1000,1000] \
++model.decoder_layers=[1000,1000] \
++l1_weight=0.5 \
++reconst_weight=0.8 \
++dropout=0
done

