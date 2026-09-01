#!/bin/bash
#SBATCH --job-name=smccert-arm2
#SBATCH --partition=cpu_long
#SBATCH --array=0-191%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --output=logs/arm2_%A_%a.out
#SBATCH --error=logs/arm2_%A_%a.err
#
# ============================================================================
# ARM 2 — TASK-2026-08-31-SMCCERT, prepared by TASK-2026-09-01-SMCRUCHE-READY,
#         deployment layer repaired by TASK-2026-09-01-SMCRUCHE-PACKFIX.
#
# NOT SUBMITTED BY ANY AGENT. The researcher submits this by hand.
#
# RUN ARM 1 FIRST. Its stopping criterion applies here: if ARM 1's gamma comes
# back KILLED, N_eff saturates, more particles cannot rescue L >= 128, and an
# L = 128 bias calibration is answering a question the programme no longer has.
# ============================================================================
#
# QUESTION
#   What is B at L = 128 at production geometry? Every L = 128 cell is currently
#   CALIBRATION_REQUIRED and no L = 128 run exists anywhere in the programme:
#   A-P128 was cancelled by the SMCSTAT frozen stopping rule 3.
#
# CELL (frozen, do not edit)
#   L = 128, T = 128.0, zeta = 0.35, lambda = 0.3032, dtau_mult = 6.0,
#   systematic resampling. N_c in {64, 128, 256}, R = 64 each.
#
# PARTITION
#   --time=08:00:00 is the FROZEN wall request and was NOT changed. On Ruche:
#       cpu_short  MaxTime 01:00:00   too short
#       cpu_med    MaxTime 04:00:00   TOO SHORT for an 8 h request
#       cpu_long   MaxTime 7-00:00:00 <- the smallest that accommodates it
#   cpu_long is therefore the smallest currently-valid CPU partition for this
#   arm. The wall request was chosen from the science (the slowest task is
#   estimated at ~1.73 h and that estimate is DERIVED, not measured), so the
#   partition follows the request rather than the other way round.
#
# RESOURCES (from run_preflight.sh; re-run it, do not trust this comment)
#   192 array tasks, ~194 core-hours total, slowest single task ~1.73 core-h,
#   peak ~665 MB per task.
#
#   THE COST IS THE LEAST RELIABLE NUMBER IN THIS PACKAGE. The L = 128 rate is
#   DERIVED from the SMCSTAT timing probe's 96 -> 128 ratio of 2.250, not
#   measured: no L = 128 run exists in this programme. Treat it as +/-50% and
#   run the single-task check in RUCHE_RUNBOOK.md before queueing 192.
#
# --array=0-191  : ONE task per manifest row. Do not change the range.
#   %64          : concurrency cap only; see arm1/submit.slurm.
# --mem=2G       : ~3.1x the measured 665 MB peak.

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

echo "[arm2] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
echo "[arm2] partition=${SLURM_JOB_PARTITION:-?}  python=$PPSQJ_PYTHON"
"$PPSQJ_PYTHON" -c 'import sys,numpy;print("[arm2] resolved",sys.executable,"numpy",numpy.__version__)'

"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[arm2] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
