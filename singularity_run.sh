#!/bin/bash

module load singularity

singularity run --nv --bind ./:/usr/src deeplearn.sif "$@" \
    ++node_id=$SLURM_NODEID ++task_id=$SLURM_LOCALID \
    ++nnodes=$SLURM_JOB_NUM_NODES ++ntasks_per_node=$SLURM_NTASKS_PER_NODE