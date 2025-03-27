# On BMRC or Jalapeno before starting, clear all modules
#module purge
module load Anaconda3
eval "$(conda shell.bash hook)"

conda activate abagen
# lightning requires python > 3.8
#module load Python/3.9.5-GCCcore-10.3.0

# Create venv if it doesn't exist already
if [ ! -d "/well/mars/users/uvy786/python/.venv" ]; then
    echo "Initializing python virtual environment"

    # Create venv
    python3 -m venv /well/mars/users/uvy786/python/.venv

    echo "installing python packages..."

    # Activate the venv
    source /well/mars/users/uvy786/python/.venv/bin/activate

    # Upgrade pip and install requirements
    echo "Upgrading pip"
    pip install pip --upgrade

    pip3 install -r requirements.txt

    #deactivate the venv
    deactivate

fi

# Load necessary modules

#Activate the python venv
#source /path/to/projectA-${MODULE_CPU_TYPE}/bin/activate #TODO setup for Ivybridge nodes

#source /well/mars/users/uvy786/python/.venv/bin/activate

