#!/bin/bash
# Case A (self-dual) guided cloning production grid.
# 5 L {32,48,64,96,128} x 15 zeta {0.05..0.85} x 13 lambda on [0.42,0.58]
# = 975 tasks (ids 0..974, under Habrok MaxArraySize 1001). lambda_c=1/2 is
# pinned by self-duality for every zeta, so the window is centered on 0.5.
# zeta=1 is NOT here: the existing Case A grid (--grid v1) carries the
# bias-free zeta=1 anchor.
#
# Guided cloning (PPS_GUIDED=1, proposal_c=zeta) + window lengthening
# PPS_DTAU_MULT=12. Case A dt0 ~ 1/(2L) gives thousands of windows at x1;
# x12 keeps ESS ~0.94 while removing most genealogical-coalescence bias in
# the absolute entropy (S plateaus at x12 in the calibration sweep).
#
# SUBMISSION (size-binned; ids from guided_caseA_tier_ranges):
#   sbatch --array=0-194%48   --time=00:45:00 --job-name=cA_32_64  scripts/habrok/pps_scan/submit_caseA_guided.sh  # L=32 ids 0-194
#   sbatch --array=195-389%48 --time=01:00:00 --job-name=cA_48     scripts/habrok/pps_scan/submit_caseA_guided.sh  # L=48
#   sbatch --array=390-584%48 --time=01:30:00 --job-name=cA_64     scripts/habrok/pps_scan/submit_caseA_guided.sh  # L=64
#   sbatch --array=585-779%48 --time=03:00:00 --job-name=cA_96     scripts/habrok/pps_scan/submit_caseA_guided.sh  # L=96
#   sbatch --array=780-974%32 --time=05:00:00 --job-name=cA_128    scripts/habrok/pps_scan/submit_caseA_guided.sh  # L=128
# Measured cost (PPS_DTAU_MULT=12): ~4,400 core-h total; longest L=128 ~3 h.
#
#SBATCH --job-name=cA_guided
#SBATCH --output=logs/cA_guided_%A_%a.out
#SBATCH --error=logs/cA_guided_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --array=0-194%48

REPO_DIR="${REPO_DIR:-$HOME/pps_qj}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/pps_qj}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/pps_qj/pps_caseA_guided}"

mkdir -p logs "$OUTPUT_DIR"
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2023.02-gfbf-2022b
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=5

export PPS_GUIDED=1
export PPS_DTAU_MULT=12
export PPS_RECORD_RENYI=1

cd "$REPO_DIR"
python -m pps_qj.parallel.worker_caseA \
    "$SLURM_ARRAY_TASK_ID" "$OUTPUT_DIR" --grid guided
