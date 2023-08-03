#On Jalapeno / BMRC
module purge
module load Python/3.7.4-GCCcore-8.3.0

#If venv does not exist, create it
if [ ! -d ".venv" ]; then
	echo "Initializing python virtual environment..."

	#Create the venv
	python3 -m venv .venv

	echo "Installing python packages..."

	#Activate the venv
	source .venv/bin/activate

	#Upgrade pip
	echo "Upgrading pip..."
	pip install pip --upgrade

	#Install necessary python packages
	pip3 install -r requirements.txt

	#Deactivate the venv
	deactivate
fi

#Load necessary modules 
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

#Activate the python venv
source .venv/bin/activate