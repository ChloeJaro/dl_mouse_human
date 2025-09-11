#!/bin/bash

set -e


export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export MPLCONFIGDIR=/tmp/$SLURM_JOB_ID/matplotlib
mkdir -p $MPLCONFIGDIR

export HYDRA_FULL_ERROR=1

python3 "$@"