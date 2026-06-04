#!/bin/bash
# =============================================================================
# submit_clone_rescue_L128.sh  --  lean L=128 rescue run (PRIORITY)
#
# Tasks 0..129 of the rescue grid: L=128, 10 zeta x 13 narrow lambda.
#   N_c=250, T=100, lambda windows centered on measured dense crossings.
#   Predicted ~13h/task; 130 tasks / 24 concurrent ~= 72h wall.
#
# Pairs against existing dense L=64 (and L=96) data for (64,128) crossings.
# Output dir SEPARATE from the dense campaign so nothing collides.
#
# Usage:
#   bash slurm/submit_clone_rescue_L128.sh [TASK_LO] [TASK_HI] [OUTDIR] [WALL]
# Defaults: 0 129  /scratch/$USER/pps_qj/pps_clone_rescue  96:00:00
# =============================================================================
set -euo pipefail

TASK_LO=${1:-0}
TASK_HI=${2:-129}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_rescue}
WALL_TIME=${4:-96:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="clrsc_L128_${TASK_LO}_${TASK_HI}"
PARTITION="regular"
LOG_DIR="/scratch/${USER}/pps_qj/logs"
CPUS_PER_TASK=5
CONCURRENT_TASKS=24

sbatch <<SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=${CONCURRENT_TASKS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${WALL_TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rug.nl
set -euo pipefail

echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}  [rescue L=128, tasks ${TASK_LO}..${TASK_HI}]"
echo "Started: \$(date)   Output: ${OUTPUT_DIR}"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=${CPUS_PER_TASK}
export PPS_RECORD_RENYI=1
export PPS_FORCE_RERUN=1
export PPS_DTAU_MULT=\${PPS_DTAU_MULT:-1.0}
# Pin the trajectory backend explicitly.  Do NOT rely on the code default:
# the Mac checkout defaults to 'scalar' but an un-pulled Habrok checkout may
# still default to 'batched', so production behaviour would otherwise depend
# on which revision the node holds.  'scalar' is bit-exact w.r.t. the dense
# campaign data these crossings are compared against.  Flip to 'batched' only
# after the N_c crossover + statistical-equivalence diagnostics clear it
# (PPS_BACKEND=batched bash slurm/submit_clone_rescue_L128.sh ...).
export PPS_BACKEND=\${PPS_BACKEND:-scalar}

mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

_progress() {
  local t0=\$(date +%s)
  while true; do
    local n=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
    echo "[\$(date +%H:%M:%S)] \${n} npz files (elapsed \$(( (\$(date +%s)-t0)/3600 ))h)"
    sleep 600
  done
}
_progress & PROGRESS_PID=\$!

seq ${TASK_LO} ${TASK_HI} | xargs -P ${CONCURRENT_TASKS} -I{} \
    python -m pps_qj.parallel.worker_clone_rescue_pps {} ${OUTPUT_DIR}

kill \${PROGRESS_PID} 2>/dev/null || true
echo "Finished: \$(date)  npz files: \$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)"
SLURM_SCRIPT

echo "Submitted ${JOB_NAME}: tasks ${TASK_LO}..${TASK_HI} (${N_TASKS}), wall ${WALL_TIME}, out ${OUTPUT_DIR}"
