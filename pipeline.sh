#!bin/bash

# -----------------------------------------------------------

# dl_mouse_human pipeline
# Based on code from Antoine Beauchamp and Mohammed Amer
# Author: Chloe Jaroszynski

# On cluster (BMRC/Jalapeno)

module purge

# Setup python virtual environment to run
source setup_venv.sh

python3 "$@"
