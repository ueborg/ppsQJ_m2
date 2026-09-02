#!/bin/bash
#SBATCH --job-name=hrl-b
#SBATCH --partition=cpu_med
#SBATCH --array=0-287%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --output=logs/b_%A_%a.out
#SBATCH --error=logs/b_%A_%a.err
#
# ============================================================================
# armB -- TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage and gate. The researcher types the submission
# command by hand. Neither this file nor the preflight contains one.
#
# RECOMMENDED for the overnight campaign.
#
# PURPOSE
#   ARM B: cheap-L, high-population three-point lambda stencil. Does a large enough population make CMI(lambda) a statistically coherent local curve?
#
# CELL (frozen; do not hand-edit -- regenerate with tools/build_arms.py)
#   L = 64, T = 64, zeta = 0.35, N_c = 1024,
#   lambda in {0.2932, 0.3032, 0.3132}, dtau_mult = 6, systematic resampling,
#   R = 96 independent populations per lambda.
#
# COST (from tools/cost_model.py; re-run run_preflight.sh, do not trust this)
#   288 array tasks, ~167 core-hours total, slowest single task
#   ~0.60 h predicted (0.84 h pessimistic),
#   peak ~665 MB per task.
#
#   Unlike the predecessor package, the L = 64 rate is MEASURED on Ruche
#   from completed ARM1/ARM2 runs of this identical code path. It is not the
#   Mac-probe extrapolation the predecessor had to flag at +/-50 %.
#
# PARTITION -- the SMALLEST valid partition for this arm's request, chosen from
# the request, not inherited from the predecessor.
#   Ruche MaxTime: cpu_short 1 h, cpu_med 4 h, cpu_long 7 d.
#   requested 03:00:00  ->  cpu_med
#   preflight.py re-derives this and EXITS NONZERO on any mismatch.
#
# --array=0-287  : ONE task per manifest row, exactly. Do not change the
#                 range; preflight.py fails if it stops matching the manifest.
#   %64         : a concurrency cap only. LOWER IT if you queue several arms at
#                 once and the allocation gives you 64 slots in total rather
#                 than 64 per array -- see RUCHE_RUNBOOK.md.
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

echo "[b] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
echo "[b] partition=${SLURM_JOB_PARTITION:-?}  python=$PPSQJ_PYTHON"
"$PPSQJ_PYTHON" -c 'import sys,numpy;print("[b] resolved",sys.executable,"numpy",numpy.__version__)'

"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[b] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
