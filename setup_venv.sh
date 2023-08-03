# On BMRC or Jalapeno before starting, clear all modules
module purge

# lightning requires python > 3.8
module load Python/3.9.6-GCCcore-11.2.0

# Create venv if it doesn't exist already
if [ ! -d ".venv" ]; then
    echo "Initializing python virtual environment"

    # Create venv
    python3 -m venv .venv

    echo "installing python packages..."

    # Activate the venv
    source .venv/bin/activate

    # Upgrade pip and install requirements
    echo "Upgrading pip"
    pip install pip --upgrade

    pip3 install -r requirements.txt

    #deactivate the venv
    deactivate

fi

# Load necessary modules

module load TensorFlow

