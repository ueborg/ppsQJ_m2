#!/bin/bash
#SBATCH --job-name=pps_doob_medium
#SBATCH --output=logs/pps_doob_medium_%A_%a.out
#SBATCH --error=logs/pps_doob_medium_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB
# L=64 680..849 | L=96 850..1019
#SBATCH --array=680-1019%20

REPO_DIR="${REPO_DIR:-$HOME/pps_qj}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/pps_qj}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/pps_qj/pps_doob_scan}"

mkdir -p logs "$OUTPUT_DIR/backward_passes"
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2023.02-gfbf-2022b
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"
python -m pps_qj.parallel.worker_doob_pps \
    "$SLURM_ARRAY_TASK_ID" "$OUTPUT_DIR" "$SLURM_CPUS_PER_TASK"
