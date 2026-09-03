#!/bin/bash
#SBATCH --job-name=lowlam-x32
#SBATCH --partition=cpu_med
#SBATCH --array=0-95%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --output=logs/x32_%A_%a.out
#SBATCH --error=logs/x32_%A_%a.err
#
# ============================================================================
# lowlamL32 -- TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage and gate. The researcher types the submission
# command by hand. Neither this file nor the preflight contains one.
#
# PURPOSE
#   LOWLAM-L32: the four new low-lambda points at L = 32, T = 32, N_c = 1024, R = 24. Completes the L = 32 curve to 17 points.
#
#   Only the FOUR NEW lambdas are here. The thirteen lambdas
#   0.2332 ... 0.3532 are ABSENT from this manifest on purpose:
#   TASK-2026-09-02-MOCK-PRODUCTION already measured them at this exact
#   (L, T, zeta, N_c, dtau_mult, resample_scheme) and they are carried in
#   ../frozen_inputs/predecessor_nc1024_populations.csv. Recomputing them
#   would have cost ~195 core-hours and bought nothing. See
#   ../REUSE_AND_DEDUP_AUDIT.md.
#
# CELL (frozen; do not hand-edit -- regenerate with tools/build_arms.py)
#   L = 32, T = 32, zeta = 0.35, N_c = 1024,
#   4 NEW lambdas, grid indices 0-3 of the frozen 17-point grid:
#     0.1932, 0.2032, 0.2132, 0.2232
#   n_steps: 64, 68, 71, 74
#   dtau_mult = 6, systematic resampling,
#   R = 24 independent populations per lambda.
#
# COST -- FITTED TO MEASURED RUCHE wall_s, never to a requested --time.
#   96 array tasks, ~3.57 core-hours total, slowest single task
#   ~2.4 min predicted (3.3 min pessimistic),
#   elapsed ~4.8 min at the cap below (6.7 min
#   pessimistic), EXCLUDING queue wait, peak ~262 MB per task.
#
#   Model: wall_s = 0.815551 * n_steps + 68.43, least squares over the
#   predecessor's completed N_c=1024 runs at this L (n_steps span
#   78-117), taken against the alternative
#   1.888 ms/clone-window worst observed rate, larger adopted.
#   Provenance: ../COST_MODEL.md.
#
# PARTITION -- cpu_med, by design, for ALL THREE arms.
#   This is NOT the predecessor's "smallest partition that fits --time" rule.
#   The preceding campaign showed cpu_short is effectively serialised for this
#   account by QOSMaxJobsPerUserLimit while cpu_med delivered real parallelism.
#   --time=00:20:00 would fit cpu_short's 1 h MaxTime on two of the three
#   arms; it is deliberately not sent there. See ../SCHEDULER_DECISION.md.
#   preflight.py requires cpu_med and EXITS NONZERO on anything else.
#
# --array=0-95  : ONE task per manifest row, exactly. Do not change the
#                range; preflight.py fails if it stops matching the manifest.
#   %64        : a concurrency cap only. 96 tasks at %64 is exactly two
#                waves, which is why the elapsed figure above is floored at
#                two slowest-tasks rather than at throughput.
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

echo "[x32] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
echo "[x32] partition=${SLURM_JOB_PARTITION:-?}  python=$PPSQJ_PYTHON"
"$PPSQJ_PYTHON" -c 'import sys,numpy;print("[x32] resolved",sys.executable,"numpy",numpy.__version__)'

"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[x32] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
