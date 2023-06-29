#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=deeplearn
#SBATCH --partition=imgvoltaq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --qos=img

srun ./singularity_python_run.sh $@