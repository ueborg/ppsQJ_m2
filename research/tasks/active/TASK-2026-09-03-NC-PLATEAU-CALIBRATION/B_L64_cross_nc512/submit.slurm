#!/bin/bash
#SBATCH --job-name=ncplat-b512
#SBATCH --partition=cpu_med
#SBATCH --array=0-335%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=logs/b512_%A_%a.out
#SBATCH --error=logs/b512_%A_%a.err
#
# ============================================================================
# B_L64_cross_nc512 -- TASK-2026-09-03-NC-PLATEAU-CALIBRATION  (campaign B)
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage, gate and approval level. The researcher types
# the submission command by hand. Neither this file nor the preflight contains
# one.
#
# PURPOSE
#   CAMPAIGN B. L = 64, T = 64, the frozen 7-point transition-region grid 0.2182-0.2482 at N_c = 512, matched R = 48. Tests whether finite-N_c distorts the SHAPE of CMI(lambda) where the low-L locator sits, not just its level.
#
# CELLS (frozen; do not hand-edit -- regenerate with tools/build_arms.py)
#   L = 64, T = 64, zeta = 0.35, systematic resampling
#     N_c=512   lambda=0.2182  dtau_mult=6    K=294   fresh=48  reuse=0
#     N_c=512   lambda=0.2232  dtau_mult=6    K=300   fresh=48  reuse=0
#     N_c=512   lambda=0.2282  dtau_mult=6    K=307   fresh=48  reuse=0
#     N_c=512   lambda=0.2332  dtau_mult=6    K=314   fresh=48  reuse=0
#     N_c=512   lambda=0.2382  dtau_mult=6    K=321   fresh=48  reuse=0
#     N_c=512   lambda=0.2432  dtau_mult=6    K=327   fresh=48  reuse=0
#     N_c=512   lambda=0.2482  dtau_mult=6    K=334   fresh=48  reuse=0
#   Populations listed as `reuse` ALREADY EXIST as exact-compatible completed
#   runs and are absent from this manifest on purpose. Recomputing them would
#   buy nothing. See ../REUSE_LEDGER.csv.
#
# COST -- from per-clone-window rates MEASURED on Ruche on this identical code
# path, never from a requested --time and never from a laptop probe.
#   336 array tasks, ~90.9 core-hours
#   (127.2 pessimistic).
#   slowest single task ~0.29 h (0.40 h pessimistic).
#   elapsed ~1.73 h at the cap below
#   (2.42 h pessimistic), EXCLUDING queue wait.
#   adopted rate(s) [6.057] ms per clone-window.
#   Provenance and the model's own failure modes: ../COST_MODEL.md.
#
# --time=01:00:00 is >= 1.6x the PESSIMISTIC slowest task. It was chosen from
#   the cost model and the partition was then chosen to fit it, never the
#   other way round.
#
# --mem=1G is 1.35x a MEASURED peak-RSS model (569 MB).
#   The formula every predecessor package used (128 + 2*N_c*per_clone) is
#   UNDER-conservative by about a factor two -- direct ru_maxrss measurements
#   are in ../COST_MODEL.md section "Memory". At the N_c this campaign reaches
#   that stops being a rounding error.
#
# PARTITION -- cpu_med. cpu_med when --time fits its 4 h MaxTime,
#   cpu_long otherwise. cpu_short is never used at any --time: it is
#   effectively serialised for this account by QOSMaxJobsPerUserLimit.
#   preflight.py recomputes this rule and EXITS NONZERO on a mismatch.
#
# --array=0-335  : ONE task per manifest row, exactly. Do not change the range;
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

echo "[b512] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
echo "[b512] partition=${SLURM_JOB_PARTITION:-?}  python=$PPSQJ_PYTHON"
"$PPSQJ_PYTHON" -c 'import sys,numpy;print("[b512] resolved",sys.executable,"numpy",numpy.__version__)'

"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[b512] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
