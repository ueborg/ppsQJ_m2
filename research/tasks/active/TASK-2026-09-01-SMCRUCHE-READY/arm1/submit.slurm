#!/bin/bash
#SBATCH --job-name=smccert-arm1
#SBATCH --array=0-111%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/arm1_%A_%a.out
#SBATCH --error=logs/arm1_%A_%a.err
#
# ============================================================================
# ARM 1 — TASK-2026-08-31-SMCCERT, prepared by TASK-2026-09-01-SMCRUCHE-READY
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
# --time=04:00:00: ~4.6x the slowest task. Generous for a shared node.
# --cpus-per-task=1 and the thread pins below: the sampler is single-threaded by
#                  design. Extra threads only oversubscribe.

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

echo "[arm1] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
python3 run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[arm1] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
