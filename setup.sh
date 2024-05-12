#!/bin/bash

# Function to install wget on Ubuntu
install_wget_ubuntu() {
    sudo apt-get update
    sudo apt-get install -y wget
}

# Function to install wget on macOS
install_wget_mac() {
    brew install wget
}

# Check if wget is installed
if ! command -v wget &> /dev/null; then
    echo "wget is not installed. Installing..."
    # Check the operating system
    os=$(uname)
    if [ "$os" = "Linux" ]; then
        echo "Detected Linux (assuming Ubuntu)"
        install_wget_ubuntu
    elif [ "$os" = "Darwin" ]; then
        echo "Detected macOS"
        install_wget_mac
    else
        echo "Unsupported operating system: $os"
        exit 1
    fi
fi

# Create models directory if it doesn't exist
mkdir -p models

# Check if the model file already exists
if [ ! -f models/gpt4all-13b-snoozy-q4_0.gguf ]; then
    echo "Downloading model file..."
    wget https://gpt4all.io/models/gguf/gpt4all-13b-snoozy-q4_0.gguf -P models/
else
    echo "Model file already exists. Skipping download."
fi

# Install Python dependencies
pip install -r requirements.txt

# Upgrade or install gpt4all package
pip install --upgrade --quiet gpt4all
