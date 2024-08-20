#!/bin/bash

# Define variables
CONDA_DIR="/users/jiali/miniconda"
CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
ENV_YAML="charon_env.yaml"

# Function to display a message
echo_message() {
    echo "======================================"
    echo "$1"
    echo "======================================"
}

# Step 1: Download and install Miniconda
if [ ! -d "$CONDA_DIR" ]; then
    echo_message "Downloading and installing Miniconda..."
    wget $CONDA_URL -O $CONDA_INSTALLER
    bash $CONDA_INSTALLER -b -p $CONDA_DIR
    rm $CONDA_INSTALLER
else
    echo_message "Miniconda already installed."
fi

# Step 2: Initialize conda
echo_message "Initializing conda..."
source "$CONDA_DIR/bin/activate"

# Step 3: Update conda
echo_message "Updating conda..."
conda update -y conda

# Step 4: Check if the environment YAML file exists
if [ -f "$ENV_YAML" ]; then
    echo_message "Creating the 'charon' environment from $ENV_YAML..."
    conda env create -f $ENV_YAML
else
    echo "Error: $ENV_YAML file not found!"
    exit 1
fi

# Step 5: Activate the charon environment
echo_message "Activating the 'charon' environment..."
conda activate charon

# Step 6: Verify the environment creation
if conda env list | grep -q "charon"; then
    echo_message "The 'charon' environment has been successfully created and activated."
else
    echo "Error: Failed to create the 'charon' environment."
    exit 1
fi

echo_message "Setup complete. You can now use the 'charon' environment."
