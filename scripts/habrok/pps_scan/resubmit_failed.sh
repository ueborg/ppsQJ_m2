#!/bin/bash
# Resubmit only the missing task ids for a Doob or cloning scan.
# Usage:
#   bash resubmit_failed.sh doob  <output_dir> submit_doob_small.sh
#   bash resubmit_failed.sh clone <output_dir> submit_clone_pps.sh
set -euo pipefail

SCAN="${1:?scan type (doob|clone)}"
OUTPUT_DIR="${2:?output directory}"
SCRIPT="${3:?sbatch script to resubmit}"

REPO_DIR="${REPO_DIR:-$HOME/pps_qj}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/pps_qj}"

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2023.02-gfbf-2022b
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

MISSING=$(python -m pps_qj.parallel.aggregate_pps check_missing "$SCAN" "$OUTPUT_DIR")

if [ -z "$MISSING" ]; then
    echo "All $SCAN tasks complete."
    exit 0
fi

echo "Resubmitting missing $SCAN tasks: $MISSING"
sbatch --array="$MISSING" "$SCRIPT"
