#! /usr/bin/env bash

# Create a `venv` virtual environment, activate and install all required packages for development.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}/..

# Install Python 3.7
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get install python3.7 python3.7-dev python3.7-venv -y

# Install GANce apt dependencies
sudo apt-get install build-essential cmake libgtk-3-dev libboost-all-dev ffmpeg -y

# Create GANce virtual environment
python3.7 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r ./requirements/prod.txt -r ./requirements/test.txt