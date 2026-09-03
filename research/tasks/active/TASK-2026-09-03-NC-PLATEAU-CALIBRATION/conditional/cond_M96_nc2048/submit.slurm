#!/bin/bash
#SBATCH --job-name=BLOCKED-xm96b
#SBATCH --partition=cpu_long
#SBATCH --array=0-107%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=18:00:00
#SBATCH --output=logs/xm96b_%A_%a.out
#SBATCH --error=logs/xm96b_%A_%a.err
#
# ============================================================================
# cond_M96_nc2048 -- TASK-2026-09-03-NC-PLATEAU-CALIBRATION
#
#            *** CONDITIONAL -- DO NOT SUBMIT ***
#            *** CAMPAIGN C ADJUDICATION -- AND ONLY ONE OF THE TWO M96 ARMS
#
# NOT SUBMITTED BY ANY AGENT, EVER. research/RESOURCE_POLICY.md section 4.
# And not by the researcher either, until the gate above is adjudicated.
#
# PURPOSE
#   CONDITIONAL, STAGE 1. L = 96 mock-production scan over the frozen 9-point grid at N_c = 2048, R = 12.
#
# CELLS
#   L = 96, T = 96, zeta = 0.35, systematic resampling, R = 12
#     N_c=2048  lambda=0.2032  dtau_mult=6    K=618   rows=12
#     N_c=2048  lambda=0.2182  dtau_mult=6    K=664   rows=12
#     N_c=2048  lambda=0.2232  dtau_mult=6    K=679   rows=12
#     N_c=2048  lambda=0.2282  dtau_mult=6    K=694   rows=12
#     N_c=2048  lambda=0.2332  dtau_mult=6    K=709   rows=12
#     N_c=2048  lambda=0.2382  dtau_mult=6    K=725   rows=12
#     N_c=2048  lambda=0.2432  dtau_mult=6    K=740   rows=12
#     N_c=2048  lambda=0.2482  dtau_mult=6    K=755   rows=12
#     N_c=2048  lambda=0.2632  dtau_mult=6    K=801   rows=12
#
# COST -- measured Ruche rates [16.106] ms per clone-window
#   108 tasks, ~702 core-hours (983 pessimistic).
#   slowest single task ~7.34 h (10.27 h pessimistic).
#   elapsed ~14.68 h at the cap below, queue wait EXCLUDED.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# HARD INTERLOCK. THIS ARM IS BLOCKED.
#
#   CAMPAIGN C ADJUDICATION -- AND ONLY ONE OF THE TWO M96 ARMS
#
# Release ONLY if campaign C identifies N_c = 2048 as the smallest N_c meeting the frozen production adequacy criterion at L = 96, or if it identifies none and the researcher accepts a scan at the largest calibrated rung. Never together with cond_M96_nc1024.
#
# Every array task exits 3 immediately, before importing the sampler, unless
# the researcher has created the release file below by hand. Submitting this
# arm before the adjudication therefore costs a few seconds of no-ops rather
# than 702 core-hours and a result nobody is entitled to interpret.
#
#   touch ../GATE_RELEASED_cond_M96_nc2048
#   # and write into it which adjudication released it, and when
# ---------------------------------------------------------------------------
RELEASE="$SLURM_SUBMIT_DIR/../GATE_RELEASED_cond_M96_nc2048"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED: cond_M96_nc2048 has not been released." >&2
    echo "  Gate: CAMPAIGN C ADJUDICATION -- AND ONLY ONE OF THE TWO M96 ARMS" >&2
    echo "  Create $RELEASE only after that adjudication." >&2
    echo "  See ../CONDITIONAL_SUBMISSION.md." >&2
    exit 3
fi
echo "[xm96b] released by: $(head -c 400 "$RELEASE")"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

PPSQJ_PYTHON="${PPSQJ_PYTHON:-/gpfs/workdir/ercetinut/envs/pps_qj/bin/python}"
if [ ! -x "$PPSQJ_PYTHON" ]; then
    echo "PPSQJ_PYTHON is not executable: $PPSQJ_PYTHON" >&2
    exit 2
fi
export PATH="$(dirname "$PPSQJ_PYTHON"):$PATH"

echo "[xm96b] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[xm96b] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
