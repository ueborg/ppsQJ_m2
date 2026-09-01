#!/bin/bash
#SBATCH --job-name=smccert-arm2
#SBATCH --array=0-191%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --output=logs/arm2_%A_%a.out
#SBATCH --error=logs/arm2_%A_%a.err
#
# ============================================================================
# ARM 2 — TASK-2026-08-31-SMCCERT, prepared by TASK-2026-09-01-SMCRUCHE-READY
#
# NOT SUBMITTED BY ANY AGENT. The researcher submits this by hand.
# Read RUCHE_RUNBOOK.md and this arm's README.md first.
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
# RESOURCES (from run_preflight.sh; re-run it, do not trust this comment)
#   192 array tasks, ~194 core-hours total, slowest single task ~1.73 core-h,
#   peak ~665 MB per task.
#
#   THE COST IS THE LEAST RELIABLE NUMBER IN THIS PACKAGE. The L = 128 rate is
#   DERIVED from the SMCSTAT timing probe's 96 -> 128 ratio of 2.250, not
#   measured: no L = 128 run exists in this programme. Treat it as +/-50% and
#   run the single-task check in RUCHE_RUNBOOK.md step 9 before queueing 192.
#
# --array=0-191  : ONE task per manifest row. Do not change the range.
#   %64          : concurrency cap only; see arm1/submit.slurm.
# --mem=2G       : ~3.1x the measured 665 MB peak.
# --time=08:00:00: ~4.6x the estimated slowest task, and the estimate itself is
#                  soft — hence the larger walltime than ARM 1.

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ -z "${PPSQJ_REPO:-}" ]; then
    echo "PPSQJ_REPO is not set. Export it to the ppsQJ_m2 repository root" >&2
    exit 1
fi
export PPSQJ_REPO

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

echo "[arm2] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
python3 run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[arm2] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
