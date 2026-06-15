#!/bin/bash
#
# Guided-cloning PILOT
#
# Validate the thinned-proposal (c = zeta) cloning upgrade before committing
# to a full low-zeta campaign.
#
# Guided proposal:
#   proposal intensity = zeta * physical intensity
#   exact residual weight per window:
#
#       G_k = exp[-(1-zeta) * Delta_Lambda_k]
#
# Default guided configuration:
#   PPS_GUIDED=1
#   PPS_DTAU_MULT=8
#
# Pilot grid:
#
#   L=32,  N_c=500, T=64:
#       grid task IDs 510, 525, 540
#
#   L=64,  N_c=200, T=128:
#       grid task IDs 765, 780, 795
#
#   L=128, N_c=200, T=128:
#       grid task IDs 1020, 1035, 1050
#
# Run guided:
#
#   sbatch scripts/habrok/pps_scan/submit_clone_guided_pilot.sh
#
# Run standard baseline:
#
#   PPS_GUIDED=0 \
#   PPS_DTAU_MULT=1 \
#   OUTPUT_DIR=/scratch/$USER/pps_qj/pps_clone_std_pilot \
#   sbatch scripts/habrok/pps_scan/submit_clone_guided_pilot.sh
#
#SBATCH --job-name=pps_guided_pilot
#SBATCH --output=pps_guided_pilot_%A_%a.out
#SBATCH --error=pps_guided_pilot_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --array=0-8%9

set -euo pipefail

# ---------------------------------------------------------------------------
# Map the compact Slurm array index 0,...,8 to the actual grid task ID.
#
# Using the original grid task IDs directly as Slurm array indices can fail
# when an ID exceeds the cluster's configured MaxArraySize.
# ---------------------------------------------------------------------------

GRID_TASK_IDS=(
    510
    525
    540
    765
    780
    795
    1020
    1035
    1050
)

ARRAY_INDEX="${SLURM_ARRAY_TASK_ID}"

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#GRID_TASK_IDS[@]} )); then
    echo "ERROR: Invalid array index: ${ARRAY_INDEX}" >&2
    exit 1
fi

GRID_TASK_ID="${GRID_TASK_IDS[$ARRAY_INDEX]}"

# ---------------------------------------------------------------------------
# Paths and run controls
# ---------------------------------------------------------------------------

REPO_DIR="${REPO_DIR:-$HOME/pps_qj}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/pps_qj}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/pps_qj/pps_clone_guided_pilot}"

PPS_GUIDED="${PPS_GUIDED:-1}"
PPS_DTAU_MULT="${PPS_DTAU_MULT:-8}"
PPS_N_WORKERS="${PPS_N_WORKERS:-5}"

mkdir -p "$OUTPUT_DIR"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "ERROR: Repository directory does not exist: $REPO_DIR" >&2
    exit 1
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: Virtual environment not found: $VENV_DIR" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Software environment
# ---------------------------------------------------------------------------

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2023.02-gfbf-2022b

source "$VENV_DIR/bin/activate"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Parallelism is across the five independent realisations.
# Prevent NumPy/SciPy from creating additional nested BLAS threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PPS_N_WORKERS
export PPS_GUIDED
export PPS_DTAU_MULT

# ---------------------------------------------------------------------------
# Diagnostic information
# ---------------------------------------------------------------------------

echo "============================================================"
echo "Guided-cloning pilot"
echo "============================================================"
echo "SLURM job ID          : ${SLURM_JOB_ID}"
echo "SLURM array job ID    : ${SLURM_ARRAY_JOB_ID}"
echo "SLURM array index     : ${SLURM_ARRAY_TASK_ID}"
echo "Mapped grid task ID   : ${GRID_TASK_ID}"
echo "Hostname              : $(hostname)"
echo "Start time            : $(date --iso-8601=seconds)"
echo "Repository            : ${REPO_DIR}"
echo "Output directory      : ${OUTPUT_DIR}"
echo "PPS_GUIDED            : ${PPS_GUIDED}"
echo "PPS_DTAU_MULT         : ${PPS_DTAU_MULT}"
echo "PPS_N_WORKERS         : ${PPS_N_WORKERS}"
echo "Allocated CPUs        : ${SLURM_CPUS_PER_TASK:-unknown}"
echo "============================================================"

cd "$REPO_DIR"

python -m pps_qj.parallel.worker_clone_pps \
    "$GRID_TASK_ID" \
    "$OUTPUT_DIR"

exit_status=$?

echo "============================================================"
echo "End time              : $(date --iso-8601=seconds)"
echo "Exit status           : ${exit_status}"
echo "============================================================"

exit "$exit_status"
```
