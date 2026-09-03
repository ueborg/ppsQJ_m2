#!/bin/bash
#SBATCH --job-name=ncplat-a8192
#SBATCH --partition=cpu_long
#SBATCH --array=0-47%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=18:00:00
#SBATCH --output=logs/a8192_%A_%a.out
#SBATCH --error=logs/a8192_%A_%a.err
#
# ============================================================================
# A_L64_nc8192 -- TASK-2026-09-03-NC-PLATEAU-CALIBRATION  (campaign A)
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage, gate and approval level. The researcher types
# the submission command by hand. Neither this file nor the preflight contains
# one.
#
# PURPOSE
#   CAMPAIGN A. L = 64, T = 64, lambda = 0.3032, N_c = 8192, R = 48. New top rung of the deep central ladder: the rung that decides whether a high-N_c plateau is OBSERVED rather than inferred by eye.
#
# CELLS (frozen; do not hand-edit -- regenerate with tools/build_arms.py)
#   L = 64, T = 64, zeta = 0.35, systematic resampling
#     N_c=8192  lambda=0.3032  dtau_mult=6    K=408   fresh=48  reuse=0
#   Populations listed as `reuse` ALREADY EXIST as exact-compatible completed
#   runs and are absent from this manifest on purpose. Recomputing them would
#   buy nothing. See ../REUSE_LEDGER.csv.
#
# COST -- from per-clone-window rates MEASURED on Ruche on this identical code
# path, never from a requested --time and never from a laptop probe.
#   48 array tasks, ~333.2 core-hours
#   (466.5 pessimistic).
#   slowest single task ~6.94 h (9.72 h pessimistic).
#   elapsed ~6.94 h at the cap below
#   (9.72 h pessimistic), EXCLUDING queue wait.
#   adopted rate(s) [7.477] ms per clone-window.
#   Provenance and the model's own failure modes: ../COST_MODEL.md.
#
# --time=18:00:00 is >= 1.6x the PESSIMISTIC slowest task. It was chosen from
#   the cost model and the partition was then chosen to fit it, never the
#   other way round.
#
# --mem=7G is 1.35x a MEASURED peak-RSS model (4647 MB).
#   The formula every predecessor package used (128 + 2*N_c*per_clone) is
#   UNDER-conservative by about a factor two -- direct ru_maxrss measurements
#   are in ../COST_MODEL.md section "Memory". At the N_c this campaign reaches
#   that stops being a rounding error.
#
# PARTITION -- cpu_long. cpu_med when --time fits its 4 h MaxTime,
#   cpu_long otherwise. cpu_short is never used at any --time: it is
#   effectively serialised for this account by QOSMaxJobsPerUserLimit.
#   preflight.py recomputes this rule and EXITS NONZERO on a mismatch.
#
# --array=0-47  : ONE task per manifest row, exactly. Do not change the range;
#              preflight.py fails if it stops matching the manifest.
#   %64      : a concurrency cap only, not a grant. If the allocation gives
#              64 slots in TOTAL rather than per array, lower it -- the %N cap
#              is the ONLY number in this file that is safe to hand-edit.
# ============================================================================

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# A batch job does not reliably inherit an interactive PATH, so the interpreter
# is resolved EXPLICITLY here rather than trusting `python3`. There is no conda
# on Ruche; this is a plain venv/prefix on the work filesystem.
PPSQJ_PYTHON="${PPSQJ_PYTHON:-/gpfs/workdir/ercetinut/envs/pps_qj/bin/python}"
if [ ! -x "$PPSQJ_PYTHON" ]; then
    echo "PPSQJ_PYTHON is not executable: $PPSQJ_PYTHON" >&2
    echo "Export PPSQJ_PYTHON to the validated interpreter, e.g." >&2
    echo "  export PPSQJ_PYTHON=\$WORKDIR/envs/pps_qj/bin/python" >&2
    exit 2
fi
export PATH="$(dirname "$PPSQJ_PYTHON"):$PATH"

echo "[a8192] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
echo "[a8192] partition=${SLURM_JOB_PARTITION:-?}  python=$PPSQJ_PYTHON"
"$PPSQJ_PYTHON" -c 'import sys,numpy;print("[a8192] resolved",sys.executable,"numpy",numpy.__version__)'

"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[a8192] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
