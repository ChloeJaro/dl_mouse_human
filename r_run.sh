#!/bin/bash

set -e

source /root/miniconda3/bin/activate r_env

Rscript "$@"