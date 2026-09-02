#!/bin/bash
#SBATCH --job-name=mock-m48
#SBATCH --partition=cpu_med
#SBATCH --array=0-311%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --output=logs/m48_%A_%a.out
#SBATCH --error=logs/m48_%A_%a.err
#
# ============================================================================
# mockL48 -- TASK-2026-09-02-MOCK-PRODUCTION
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage and gate. The researcher types the submission
# command by hand. Neither this file nor the preflight contains one.
#
# PURPOSE
#   MOCK-L48: the full 13-point CMI(lambda) scan at L = 48, T = 48, N_c = 1024.
#
# CELL (frozen; do not hand-edit -- regenerate with tools/build_arms.py)
#   L = 48, T = 48, zeta = 0.35, N_c = 1024,
#   13 lambdas on the frozen 13-point grid:
#     0.2332, 0.2432, 0.2532, 0.2632, 0.2732, 0.2832, 0.2932, 0.3032, 0.3132, 0.3232, 0.3332, 0.3432, 0.3532
#   dtau_mult = 6, systematic resampling,
#   R = 24 independent populations per lambda.
#
# COST (from tools/cost_model.py; re-run run_preflight.sh, do not trust this)
#   312 array tasks, ~59 core-hours total, slowest single task
#   ~0.23 h predicted (0.32 h pessimistic),
#   elapsed ~1.06 h at the cap below (1.48 h pessimistic),
#   peak ~430 MB per task.
#
#   The rate is 3.000 ms per clone-window, anchored on
#   wall_s ACTUALLY RECORDED by completed Ruche jobs of this identical code
#   path -- never on a requested --time. Provenance: ../COST_MODEL.md.
#
# PARTITION -- the SMALLEST valid partition for this arm's request, chosen from
# the request, not inherited from any predecessor.
#   Ruche MaxTime: cpu_short 1 h, cpu_med 4 h, cpu_long 7 d.
#   requested 02:00:00  ->  cpu_med
#   preflight.py re-derives this and EXITS NONZERO on any mismatch.
#
# --array=0-311  : ONE task per manifest row, exactly. Do not change the
#                 range; preflight.py fails if it stops matching the manifest.
#   %64         : a concurrency cap only. On 2026-09-02 the predecessor's ARM B
#                 demonstrated that this allocation grants 64 concurrent slots
#                 PER ARRAY with three arrays live at once. LOWER IT if the
#                 accounting check in ../RUCHE_RUNBOOK.md section 2 says your
#                 slots are shared in total rather than per array.
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

echo "[m48] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
echo "[m48] partition=${SLURM_JOB_PARTITION:-?}  python=$PPSQJ_PYTHON"
"$PPSQJ_PYTHON" -c 'import sys,numpy;print("[m48] resolved",sys.executable,"numpy",numpy.__version__)'

"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[m48] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
