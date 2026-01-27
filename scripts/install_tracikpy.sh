#!/usr/bin/env bash
set -euo pipefail

echo "Initializing tracikpy submodule..."
# git submodule update --init --recursive third_party/tracikpy
# sudo apt-get install libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev libnlopt-cxx-dev

# get articubot conda environment from conda info
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate articubot

# Clear the problematic variables
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH

# Set compiler to system default (to match system headers)
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Only add Eigen3, which is often in a non-standard subfolder (/usr/include/eigen3)
# Do NOT add /usr/include here.
export C_INCLUDE_PATH=/usr/include/eigen3
export CPLUS_INCLUDE_PATH=/usr/include/eigen3

echo "Building and installing tracikpy..."
cd third_party
pip install ./tracikpy
cd ..

exit 0