#!/bin/bash
# =============================================================================
# submit_clone_rescue_L160.sh  --  optional L=160 rescue run
#
# Tasks 130..259 of the rescue grid: L=160, 10 zeta x 13 narrow lambda.
#   N_c=120, T=100.  Predicted ~16h/task; 130 tasks / 24 conc ~= 87h wall.
# Only worth running if you have node-time after L=128. Gives a 4th FSS
# point (64,96,128,160) -> can fit the correction-to-scaling exponent.
#
# Usage:
#   bash slurm/submit_clone_rescue_L160.sh [TASK_LO] [TASK_HI] [OUTDIR] [WALL]
# Defaults: 130 259  /scratch/$USER/pps_qj/pps_clone_rescue  120:00:00
# =============================================================================
set -euo pipefail

TASK_LO=${1:-130}
TASK_HI=${2:-259}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_rescue}
WALL_TIME=${4:-120:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="clrsc_L160_${TASK_LO}_${TASK_HI}"
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

echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}  [rescue L=160, tasks ${TASK_LO}..${TASK_HI}]"
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
# Pin the trajectory backend explicitly (see submit_clone_rescue_L128.sh).
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
