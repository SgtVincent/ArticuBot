#!/usr/bin/env bash
set -euo pipefail

echo "Initializing tracikpy submodule..."
git submodule update --init --recursive third_party/tracikpy

echo "Ensure the 'articubot' conda environment is active (conda activate articubot)"
echo "Installing tracikpy into the active environment..."
python -m pip install -e third_party/tracikpy

cat <<'NOTE'
Done. If the build fails, make sure the following system packages are installed on Ubuntu:
  sudo apt-get install libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev libnlopt-cxx-dev
NOTE

exit 0
