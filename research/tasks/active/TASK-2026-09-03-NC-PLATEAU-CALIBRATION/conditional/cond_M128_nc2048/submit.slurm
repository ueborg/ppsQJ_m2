#!/bin/bash
#SBATCH --job-name=BLOCKED-xm128a
#SBATCH --partition=cpu_long
#SBATCH --array=0-71%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=9G
#SBATCH --time=72:00:00
#SBATCH --output=logs/xm128a_%A_%a.out
#SBATCH --error=logs/xm128a_%A_%a.err
#
# ============================================================================
# cond_M128_nc2048 -- TASK-2026-09-03-NC-PLATEAU-CALIBRATION
#
#            *** CONDITIONAL -- DO NOT SUBMIT ***
#            *** CAMPAIGN D ADJUDICATION -- STRONGLY GATED
#
# NOT SUBMITTED BY ANY AGENT, EVER. research/RESOURCE_POLICY.md section 4.
# And not by the researcher either, until the gate above is adjudicated.
#
# PURPOSE
#   CONDITIONAL, STAGE 1. L = 128 mock-production scan over the frozen 9-point grid at N_c = 2048, R = 8.
#
# CELLS
#   L = 128, T = 128, zeta = 0.35, systematic resampling, R = 8
#     N_c=2048  lambda=0.2032  dtau_mult=6    K=1102  rows=8
#     N_c=2048  lambda=0.2182  dtau_mult=6    K=1183  rows=8
#     N_c=2048  lambda=0.2232  dtau_mult=6    K=1210  rows=8
#     N_c=2048  lambda=0.2282  dtau_mult=6    K=1237  rows=8
#     N_c=2048  lambda=0.2332  dtau_mult=6    K=1264  rows=8
#     N_c=2048  lambda=0.2382  dtau_mult=6    K=1291  rows=8
#     N_c=2048  lambda=0.2432  dtau_mult=6    K=1318  rows=8
#     N_c=2048  lambda=0.2482  dtau_mult=6    K=1345  rows=8
#     N_c=2048  lambda=0.2632  dtau_mult=6    K=1427  rows=8
#
# COST -- measured Ruche rates [33.592] ms per clone-window
#   72 tasks, ~1739 core-hours (2435 pessimistic).
#   slowest single task ~27.27 h (38.18 h pessimistic).
#   elapsed ~54.54 h at the cap below, queue wait EXCLUDED.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# HARD INTERLOCK. THIS ARM IS BLOCKED.
#
#   CAMPAIGN D ADJUDICATION -- STRONGLY GATED
#
# Release ONLY if campaign D's N_c = 2048 rung PASSES the frozen adequacy screen at L = 128. If it fails, the conditional N_c = 4096 central rung comes first and this arm stays blocked. An adequate N_c must be identified BEFORE a 9-point scan at this L is run at all.
#
# Every array task exits 3 immediately, before importing the sampler, unless
# the researcher has created the release file below by hand. Submitting this
# arm before the adjudication therefore costs a few seconds of no-ops rather
# than 1739 core-hours and a result nobody is entitled to interpret.
#
#   touch ../GATE_RELEASED_cond_M128_nc2048
#   # and write into it which adjudication released it, and when
# ---------------------------------------------------------------------------
RELEASE="$SLURM_SUBMIT_DIR/../GATE_RELEASED_cond_M128_nc2048"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED: cond_M128_nc2048 has not been released." >&2
    echo "  Gate: CAMPAIGN D ADJUDICATION -- STRONGLY GATED" >&2
    echo "  Create $RELEASE only after that adjudication." >&2
    echo "  See ../CONDITIONAL_SUBMISSION.md." >&2
    exit 3
fi
echo "[xm128a] released by: $(head -c 400 "$RELEASE")"

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

echo "[xm128a] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[xm128a] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
