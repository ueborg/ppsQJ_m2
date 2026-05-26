#!/bin/bash
# Slope-test cloning run — Part A: ζ ∈ {0.80, 0.90} at L ≤ 128
#
# Physics goal: fill the two ζ-gaps in the v2 grid needed for the slope
# test that discriminates Möbius (d λ_c/d ζ|_{ζ=1} = 1/8) from naive
# NLSM (= 1/4).  Part A covers all L ≤ 128; Part B (submit_clone_slope_large.sh)
# covers L = 192, 256 where finite-size effects are controlled.
#
# Grid: task_ids 0..383  (384 tasks total, see grid_pps.py slope section)
#   Part A layout: 8 L-values × 24 λ-points × 2 ζ-values = 384 tasks
#   L=  8: tasks   0.. 47   N_c=2000
#   L= 16: tasks  48.. 95   N_c=1000
#   L= 24: tasks  96..143   N_c=800
#   L= 32: tasks 144..191   N_c=500
#   L= 48: tasks 192..239   N_c=300
#   L= 64: tasks 240..287   N_c=200
#   L= 96: tasks 288..335   N_c=150
#   L=128: tasks 336..383   N_c=100
#
# Wall time: L=128 worst case ~20 min; 4h is conservative.
# Output dir: /scratch/$USER/pps_qj/pps_clone_slope
#
#SBATCH --job-name=pps_slope_small
#SBATCH --output=logs/pps_slope_small_%A_%a.out
#SBATCH --error=logs/pps_slope_small_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --array=0-383%96

REPO_DIR="${REPO_DIR:-$HOME/pps_qj}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/pps_qj}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/pps_qj/pps_clone_slope}"

mkdir -p logs "$OUTPUT_DIR"
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2023.02-gfbf-2022b
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"
python -m pps_qj.parallel.worker_clone_pps \
    "$SLURM_ARRAY_TASK_ID" "$OUTPUT_DIR" --grid slope
