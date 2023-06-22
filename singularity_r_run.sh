#!/bin/bash

module load singularity

singularity run --nv --bind ./:/usr/src deeplearn.sif r_run.sh "$@"