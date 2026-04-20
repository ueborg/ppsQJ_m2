#!/bin/bash
#SBATCH --job-name=pps_analysis
#SBATCH --output=logs/pps_analysis_%j.out
#SBATCH --error=logs/pps_analysis_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
# Submit AFTER the array jobs with a dependency. Example:
#   SMALL=$(sbatch --parsable submit_doob_small.sh)
#   MEDIUM=$(sbatch --parsable submit_doob_medium.sh)
#   LARGE=$(sbatch --parsable submit_doob_large.sh)
#   CLONE=$(sbatch --parsable submit_clone_pps.sh)
#   sbatch --dependency=afterany:${SMALL}:${MEDIUM}:${LARGE}:${CLONE} \
#          submit_analysis.sh

REPO_DIR="${REPO_DIR:-$HOME/pps_qj}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/pps_qj}"
DOOB_DIR="${DOOB_DIR:-/scratch/$USER/pps_qj/pps_doob_scan}"
CLONE_DIR="${CLONE_DIR:-/scratch/$USER/pps_qj/pps_clone_scan}"
FIG_DIR="${FIG_DIR:-/scratch/$USER/pps_qj/figures}"

mkdir -p logs "$FIG_DIR"
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2023.02-gfbf-2022b
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

# Aggregate (produces .pkl files if complete).
python -m pps_qj.parallel.aggregate_pps doob "$DOOB_DIR"
python -m pps_qj.parallel.aggregate_pps clone "$CLONE_DIR"

# Partial-data-tolerant monitor + figures (runs whether 10% or 100% complete).
python analysis/monitor_and_plot.py "$DOOB_DIR" --output-figures "$FIG_DIR"

# Full analysis figures (only if complete, else skipped gracefully).
DOOB_PKL="$DOOB_DIR/doob_aggregate.pkl"
CLONE_PKL="$CLONE_DIR/clone_aggregate.pkl"
if [ -f "$DOOB_PKL" ]; then
    python analysis/pps_phase_diagram.py "$DOOB_PKL" \
        "${CLONE_PKL:-none}" "$FIG_DIR"
fi

echo "Analysis complete. Figures: $FIG_DIR"
echo "Copy results to \$HOME (scratch is not backed up):"
echo "  rsync -av /scratch/\$USER/pps_qj/ \$HOME/pps_qj_results/"
