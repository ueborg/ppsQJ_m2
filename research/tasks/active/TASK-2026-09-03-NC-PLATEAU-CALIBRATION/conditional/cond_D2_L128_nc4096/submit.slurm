#!/bin/bash
#SBATCH --job-name=BLOCKED-xd4096
#SBATCH --partition=cpu_long
#SBATCH --array=0-7%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=26G
#SBATCH --time=168:00:00
#SBATCH --output=logs/xd4096_%A_%a.out
#SBATCH --error=logs/xd4096_%A_%a.err
#
# ============================================================================
# cond_D2_L128_nc4096 -- TASK-2026-09-03-NC-PLATEAU-CALIBRATION
#
#            *** CONDITIONAL -- DO NOT SUBMIT ***
#            *** CAMPAIGN D ADJUDICATION
#
# NOT SUBMITTED BY ANY AGENT, EVER. research/RESOURCE_POLICY.md section 4.
# And not by the researcher either, until the gate above is adjudicated.
#
# PURPOSE
#   CONDITIONAL. L = 128, T = 128, lambda = 0.3032, N_c = 4096, R = 8. One further rung on the hardest ladder in the programme. Read the wall-time line before releasing this: a single population is a multi-day job.
#
# CELLS
#   L = 128, T = 128, zeta = 0.35, systematic resampling, R = 8
#     N_c=4096  lambda=0.3032  dtau_mult=6    K=1643  rows=8
#
# COST -- measured Ruche rates [38.243] ms per clone-window
#   8 tasks, ~572 core-hours (801 pessimistic).
#   slowest single task ~71.49 h (100.09 h pessimistic).
#   elapsed ~71.49 h at the cap below, queue wait EXCLUDED.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# HARD INTERLOCK. THIS ARM IS BLOCKED.
#
#   CAMPAIGN D ADJUDICATION
#
# Recommend this arm if, on the L = 128 ladder completed by campaign D, EITHER |Delta_1024| = |I_2048 - I_1024| is resolved OUTSIDE the frozen material tolerance tau_I = 0.006 (i.e. the 95 % interval excludes [-tau_I, +tau_I]), OR no plateau criterion P1-P5 of ../SUCCESS_CRITERIA.yaml is satisfied at the top of that ladder. Do NOT recommend it because the observed Delta_1024 'looks large'.
#
# Every array task exits 3 immediately, before importing the sampler, unless
# the researcher has created the release file below by hand. Submitting this
# arm before the adjudication therefore costs a few seconds of no-ops rather
# than 572 core-hours and a result nobody is entitled to interpret.
#
#   touch ../GATE_RELEASED_cond_D2_L128_nc4096
#   # and write into it which adjudication released it, and when
# ---------------------------------------------------------------------------
RELEASE="$SLURM_SUBMIT_DIR/../GATE_RELEASED_cond_D2_L128_nc4096"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED: cond_D2_L128_nc4096 has not been released." >&2
    echo "  Gate: CAMPAIGN D ADJUDICATION" >&2
    echo "  Create $RELEASE only after that adjudication." >&2
    echo "  See ../CONDITIONAL_SUBMISSION.md." >&2
    exit 3
fi
echo "[xd4096] released by: $(head -c 400 "$RELEASE")"

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

echo "[xd4096] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[xd4096] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
