#!/bin/bash
#SBATCH --job-name=smccert-arm1
#SBATCH --partition=cpu_med
#SBATCH --array=0-111%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/arm1_%A_%a.out
#SBATCH --error=logs/arm1_%A_%a.err
#
# ============================================================================
# ARM 1 — TASK-2026-08-31-SMCCERT, prepared by TASK-2026-09-01-SMCRUCHE-READY,
#         deployment layer repaired by TASK-2026-09-01-SMCRUCHE-PACKFIX.
#
# NOT SUBMITTED BY ANY AGENT. The researcher submits this by hand.
# Read RUCHE_RUNBOOK.md and this arm's README.md first.
# ============================================================================
#
# QUESTION
#   Does Var(Ihat) ~ N_c^-1 hold at measured VIF >= 40 AT PRODUCTION GEOMETRY
#   (T = L)? Three local attempts returned INCONCLUSIVE; the sharpest gave
#   gamma = +0.630 CI [+0.389, +0.856] and missed the pre-registered KILL by
#   0.003 at one CI endpoint.
#
# CELL (frozen, do not edit)
#   L = 96, T = 96.0, zeta = 0.35, lambda = 0.3032, dtau_mult = 6.0,
#   systematic resampling. N_c in {128, 256, 512}, R = 32 / 32 / 48 new runs.
#   Seeds CONTINUE the SMCSTAT A-P96 stream, so no completed run is repeated.
#
# PARTITION
#   --time=04:00:00 is the frozen wall request. On Ruche:
#       cpu_short  MaxTime 01:00:00   <- TOO SHORT; the first attempt landed
#                                        here by default and was killed.
#       cpu_med    MaxTime 04:00:00   <- exactly accommodates 04:00:00.
#       cpu_long   MaxTime 7-00:00:00
#   cpu_med is the smallest partition that fits. The wall request was NOT
#   changed to suit the partition; the partition was chosen to suit it.
#
# RESOURCES (from run_preflight.sh; re-run it, do not trust this comment)
#   112 array tasks, ~62 core-hours total, slowest single task ~0.86 core-h,
#   peak ~732 MB per task.
#
# --array=0-111  : ONE task per manifest row. Do not change the range; it is
#                  the manifest length and run_cell.py indexes into it.
#   %64          : concurrency cap only. Raise or lower freely — the work is
#                  embarrassingly parallel and every task checkpoints itself by
#                  writing its own output file, so a partial run loses nothing
#                  and a resubmission skips what is already done.
# --mem=2G       : ~2.8x the measured 732 MB peak.
# --cpus-per-task=1 and the thread pins below: the sampler is single-threaded by
#                  design. Extra threads only oversubscribe.

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# --- Python environment ------------------------------------------------------
# A batch job does not reliably inherit an interactive PATH, so the interpreter
# is resolved EXPLICITLY here rather than trusting `python3` to be the right one.
# There is no conda on Ruche; this is a plain venv/prefix on the work filesystem.
# Override with PPSQJ_PYTHON if the environment moves.
PPSQJ_PYTHON="${PPSQJ_PYTHON:-/gpfs/workdir/ercetinut/envs/pps_qj/bin/python}"
if [ ! -x "$PPSQJ_PYTHON" ]; then
    echo "PPSQJ_PYTHON is not executable: $PPSQJ_PYTHON" >&2
    echo "Export PPSQJ_PYTHON to the validated interpreter, e.g." >&2
    echo "  export PPSQJ_PYTHON=\$WORKDIR/envs/pps_qj/bin/python" >&2
    exit 2
fi
export PATH="$(dirname "$PPSQJ_PYTHON"):$PATH"

echo "[arm1] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
echo "[arm1] partition=${SLURM_JOB_PARTITION:-?}  python=$PPSQJ_PYTHON"
"$PPSQJ_PYTHON" -c 'import sys,numpy;print("[arm1] resolved",sys.executable,"numpy",numpy.__version__)'

"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[arm1] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
