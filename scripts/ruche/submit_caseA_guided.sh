#!/bin/bash
# Case A (self-dual) guided cloning grid, RUCHE version.
# Grid (pps_qj.parallel.grid_caseA, --grid guided):
#   L in {32,48,64,96,128}  x  15 zeta {0.05..0.85}  x  13 lambda on [0.42,0.58]
#   = 975 tasks (ids 0..974), L outer. lambda_c = 1/2 pinned by self-duality,
#   so the window is dense and centered on 0.5. nreal=5 per cell (built in).
# Guided cloning: PPS_GUIDED=1 (proposal_c=zeta), PPS_DTAU_MULT=12 window
# lengthening (keeps ESS ~0.95-0.98). zeta=1 is NOT in this grid.
#
# Task-id ranges by L (15 zeta x 13 lambda = 195 tasks each):
#   L=32 0-194   L=48 195-389   L=64 390-584   L=96 585-779   L=128 780-974
#
# SUBMISSION (size-binned):
#   sbatch -p cpu_med  --time=01:30:00 --array=0-194%40    scripts/ruche/submit_caseA_guided.sh  # L=32
#   sbatch -p cpu_med  --time=02:30:00 --array=195-389%40  scripts/ruche/submit_caseA_guided.sh  # L=48
#   sbatch -p cpu_med  --time=03:30:00 --array=390-584%40  scripts/ruche/submit_caseA_guided.sh  # L=64
#   sbatch -p cpu_long --time=12:00:00 --array=585-779%25  scripts/ruche/submit_caseA_guided.sh  # L=96
#   sbatch -p cpu_long --time=24:00:00 --array=780-974%20  scripts/ruche/submit_caseA_guided.sh  # L=128
#
#SBATCH --job-name=cA_guided
#SBATCH --output=logs/cA_%A_%a.out
#SBATCH --error=logs/cA_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16GB
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/ppsQJ_m2}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/pps/caseA_guided}"
mkdir -p logs "$OUTPUT_DIR"

module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORKDIR/envs/pps_qj"
export PYTHONPATH="$REPO_DIR"
python -c "import numpy; print('numpy', numpy.__version__)"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PPS_N_WORKERS="${PPS_N_WORKERS:-5}"
export PPS_GUIDED=1
export PPS_DTAU_MULT=12
export PPS_RECORD_RENYI=1

cd "$REPO_DIR"
srun python -m pps_qj.parallel.worker_caseA "$SLURM_ARRAY_TASK_ID" "$OUTPUT_DIR" --grid guided
