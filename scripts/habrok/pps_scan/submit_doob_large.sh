#!/bin/bash
#SBATCH --job-name=pps_doob_large
#SBATCH --output=logs/pps_doob_large_%A_%a.out
#SBATCH --error=logs/pps_doob_large_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64GB
# L=128 1020..1189 | L=192 1190..1359 | L=256 1360..1529
# If L=256 tasks time out, resubmit only the 1360..1529 range with reduced
# n_traj by editing n_traj_for_L in pps_qj/parallel/grid_pps.py.
#SBATCH --array=1020-1529%10

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
