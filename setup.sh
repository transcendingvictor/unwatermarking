#!/bin/bash

# Define the name of the virtual environment
VENV_NAME="venv"

# Check for existing virtual environment
if [ -d "$VENV_NAME" ]; then
  echo "Virtual environment $VENV_NAME already exists."
else
  # Create a virtual environment
  echo "Creating virtual environment..."
  python3 -m venv $VENV_NAME
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source $VENV_NAME/bin/activate

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages from requirements.txt
echo "Installing required packages from requirements.txt..."
pip install -r requirements.txt

echo "Setup is complete. Virtual environment '$VENV_NAME' is ready to use."