#!/bin/bash
# Slope-test cloning run — Part B: L = 192, 256 at ζ ∈ {0.70, 0.80, 0.90}
#
# Physics goal: precision λ_c at the three large-ζ values needed to extract
# the slope d λ_c/d ζ|_{ζ=1}.  L = 192, 256 are required because finite-size
# bias at L ≤ 128 (~0.05 in λ_c) exceeds the discriminating signal (~0.04
# between Möbius and naive NLSM predictions at ζ = 0.7).
#
# Grid: task_ids 384..527  (144 tasks total, see grid_pps.py slope section)
#   Part B layout: 2 L-values × 24 λ-points × 3 ζ-values = 144 tasks
#   L=192: tasks 384..455  N_c=80   ζ ∈ {0.70, 0.80, 0.90}
#   L=256: tasks 456..527  N_c=40   ζ ∈ {0.70, 0.80, 0.90}
#
# Wall time: empirical from earlier FST runs — L=256, α=0.35 ≈ 20h.
#   36h covers all λ with generous margin.
# cpus-per-task=5: intra-task parallelism over N_REAL=5 realisations.
#   Set PPS_N_WORKERS=5 via the env block below.
# Output dir: same /scratch/$USER/pps_qj/pps_clone_slope as Part A.
#
#SBATCH --job-name=pps_slope_large
#SBATCH --output=logs/pps_slope_large_%A_%a.out
#SBATCH --error=logs/pps_slope_large_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --array=384-527%10

REPO_DIR="${REPO_DIR:-$HOME/pps_qj}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/pps_qj}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/pps_qj/pps_clone_slope}"

mkdir -p logs "$OUTPUT_DIR"
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2023.02-gfbf-2022b
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export PPS_N_WORKERS=5   # one worker per realisation, 5 realisations per task

cd "$REPO_DIR"
python -m pps_qj.parallel.worker_clone_pps \
    "$SLURM_ARRAY_TASK_ID" "$OUTPUT_DIR" --grid slope
