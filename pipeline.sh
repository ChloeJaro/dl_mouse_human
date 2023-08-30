#!bin/bash

# -----------------------------------------------------------

# dl_mouse_human pipeline
# Based on code from Antoine Beauchamp and Mohammed Amer
# Author: Chloe Jaroszynski

# On cluster (BMRC/Jalapeno)

module purge

# Setup python virtual environment
#sh setup_venv.sh
source /well/mars/users/uvy786/python/.venv/bin/activate
python3 "$@"
