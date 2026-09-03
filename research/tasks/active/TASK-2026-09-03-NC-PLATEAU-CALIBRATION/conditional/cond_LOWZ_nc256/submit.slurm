#!/bin/bash
#SBATCH --job-name=BLOCKED-xz256
#SBATCH --partition=cpu_med
#SBATCH --array=0-47%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=logs/xz256_%A_%a.out
#SBATCH --error=logs/xz256_%A_%a.err
#
# ============================================================================
# cond_LOWZ_nc256 -- TASK-2026-09-03-NC-PLATEAU-CALIBRATION
#
#            *** CONDITIONAL -- DO NOT SUBMIT ***
#            *** OPTIONAL -- NOT PART OF THE zeta = 0.35 CALIBRATION
#
# NOT SUBMITTED BY ANY AGENT, EVER. research/RESOURCE_POLICY.md section 4.
# And not by the researcher either, until the gate above is adjudicated.
#
# PURPOSE
#   OPTIONAL. L = 64, T = 64, zeta = 0.10, lambda = 0.3032, N_c = 256, R = 48.
#
# CELLS
#   L = 64, T = 64, zeta = 0.1, systematic resampling, R = 48
#     N_c=256   lambda=0.3032  dtau_mult=6    K=408   rows=48
#
# COST -- measured Ruche rates [6.634] ms per clone-window
#   48 tasks, ~9 core-hours (13 pessimistic).
#   slowest single task ~0.19 h (0.27 h pessimistic).
#   elapsed ~0.19 h at the cap below, queue wait EXCLUDED.
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# HARD INTERLOCK. THIS ARM IS BLOCKED.
#
#   OPTIONAL -- NOT PART OF THE zeta = 0.35 CALIBRATION
#
# As cond_LOWZ_nc64. The pre-registered kill criterion needs BOTH population sizes: drift at zeta = 0.10 greater than or equal to drift at zeta = 0.35 kills the guided-residual mechanism and revives Born-rarity reasoning. Release both or neither.
#
# Every array task exits 3 immediately, before importing the sampler, unless
# the researcher has created the release file below by hand. Submitting this
# arm before the adjudication therefore costs a few seconds of no-ops rather
# than 9 core-hours and a result nobody is entitled to interpret.
#
#   touch ../GATE_RELEASED_cond_LOWZ_nc256
#   # and write into it which adjudication released it, and when
# ---------------------------------------------------------------------------
RELEASE="$SLURM_SUBMIT_DIR/../GATE_RELEASED_cond_LOWZ_nc256"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED: cond_LOWZ_nc256 has not been released." >&2
    echo "  Gate: OPTIONAL -- NOT PART OF THE zeta = 0.35 CALIBRATION" >&2
    echo "  Create $RELEASE only after that adjudication." >&2
    echo "  See ../CONDITIONAL_SUBMISSION.md." >&2
    exit 3
fi
echo "[xz256] released by: $(head -c 400 "$RELEASE")"

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

echo "[xz256] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[xz256] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
