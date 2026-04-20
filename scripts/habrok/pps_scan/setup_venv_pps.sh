#!/bin/bash
# One-time setup of the Python virtual environment on Habrok for pps_qj.
# Run this ONCE on a login node before submitting any array jobs.
set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venvs/pps_qj}"
REPO_DIR="${REPO_DIR:-$HOME/pps_qj}"

echo "Loading Habrok modules..."
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2023.02-gfbf-2022b

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR"
    python -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# SciPy-bundle on Habrok already provides numpy, scipy, matplotlib. We only
# install anything NOT already present in the bundle. Edit as needed.
python -m pip install --no-deps --upgrade \
    ipykernel jupyterlab

echo ""
echo "Venv ready: $VENV_DIR"
echo "Python:     $(which python)"
echo "numpy:      $(python -c 'import numpy, sys; print(numpy.__version__)')"
echo "scipy:      $(python -c 'import scipy, sys; print(scipy.__version__)')"
echo ""
echo "Smoke-check the package import:"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
python -m pps_qj.parallel.grid_pps doob | head -5
