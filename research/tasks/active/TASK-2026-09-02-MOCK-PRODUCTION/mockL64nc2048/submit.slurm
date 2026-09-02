#!/bin/bash
#SBATCH --job-name=mock-m64h
#SBATCH --partition=cpu_med
#SBATCH --array=0-71%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --output=logs/m64h_%A_%a.out
#SBATCH --error=logs/m64h_%A_%a.err
#
# ============================================================================
# mockL64nc2048 -- TASK-2026-09-02-MOCK-PRODUCTION
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage and gate. The researcher types the submission
# command by hand. Neither this file nor the preflight contains one.
#
# PURPOSE
#   MOCK-L64-NC2048: the shape check. Three lambdas at N_c = 2048 against ARM B's N_c = 1024 at exactly the same three lambdas, to measure whether the finite-population correction Delta_N(lambda) = I_2048 - I_1024 is a common shift, lambda-dependent, or unresolved. No 1/N_c law is fitted.
#
# CELL (frozen; do not hand-edit -- regenerate with tools/build_arms.py)
#   L = 64, T = 64, zeta = 0.35, N_c = 2048,
#   3 lambdas on the frozen 13-point grid:
#     0.2932, 0.3032, 0.3132
#   dtau_mult = 6, systematic resampling,
#   R = 24 independent populations per lambda.
#
# COST (from tools/cost_model.py; re-run run_preflight.sh, do not trust this)
#   72 array tasks, ~97 core-hours total, slowest single task
#   ~1.39 h predicted (1.95 h pessimistic),
#   elapsed ~1.75 h at the cap below (2.45 h pessimistic),
#   peak ~1202 MB per task.
#
#   The rate is 5.820 ms per clone-window, anchored on
#   wall_s ACTUALLY RECORDED by completed Ruche jobs of this identical code
#   path -- never on a requested --time. Provenance: ../COST_MODEL.md.
#
# PARTITION -- the SMALLEST valid partition for this arm's request, chosen from
# the request, not inherited from any predecessor.
#   Ruche MaxTime: cpu_short 1 h, cpu_med 4 h, cpu_long 7 d.
#   requested 04:00:00  ->  cpu_med
#   preflight.py re-derives this and EXITS NONZERO on any mismatch.
#
# --array=0-71  : ONE task per manifest row, exactly. Do not change the
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

echo "[m64h] task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date -u +%FT%TZ)"
echo "[m64h] partition=${SLURM_JOB_PARTITION:-?}  python=$PPSQJ_PYTHON"
"$PPSQJ_PYTHON" -c 'import sys,numpy;print("[m64h] resolved",sys.executable,"numpy",numpy.__version__)'

"$PPSQJ_PYTHON" run_cell.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_SUBMIT_DIR}/results"
echo "[m64h] task ${SLURM_ARRAY_TASK_ID} done at $(date -u +%FT%TZ)"
