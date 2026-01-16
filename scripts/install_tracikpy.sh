#!/usr/bin/env bash
set -euo pipefail

echo "Initializing tracikpy submodule..."
git submodule update --init --recursive third_party/tracikpy

# Prefer installing into the 'articubot' environment to avoid touching base.
if [ "${CONDA_DEFAULT_ENV:-}" = "articubot" ]; then
  echo "Active conda env is 'articubot' — installing tracikpy into it."
  python -m pip install -e third_party/tracikpy
else
  echo "Installing tracikpy into 'articubot' env using 'conda run' to avoid polluting base."
  # Try with --no-capture-output for better logging, fall back without if unsupported
  conda run -n articubot --no-capture-output python -m pip install -e third_party/tracikpy || \
    conda run -n articubot python -m pip install -e third_party/tracikpy
fi

cat <<'NOTE'
Done. If the build fails, make sure the following system packages are installed on Ubuntu:
  sudo apt-get install libeigen3-dev liborocos-kdl-dev libkdl-parser-dev liburdfdom-dev libnlopt-dev libnlopt-cxx-dev
NOTE

exit 0
