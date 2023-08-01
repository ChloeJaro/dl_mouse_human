# On BMRC or Jalapeno before starting, clear all modules
module purge

module load Python/Python-3.7.4-GCCcore-8.3.0

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

module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

