#!/bin/bash
# =============================================================================
# submit_clone_dense_medium_L.sh  —  Dense fine-grid scan, Node 2
#
# Covers L = 48, 64  (tasks 2056..3083, 1028 tasks total)
# Dense grid: 2 sizes × 514 tasks/L = 1028 tasks
#   N_c = 600 (L=48), 400 (L=64)
#   Intra-task MP: CPUS_PER_TASK=5 (one worker per realisation)
#   Concurrency:   24 tasks × 5 cores = 120 cores on 1 node
#
# Worst-case timing estimate (with Renyi enabled, n_workers=5):
#   L=48 worst case: ~1.7 h/task wallclock
#   L=64 worst case: ~2.5 h/task wallclock
#   Mean ~1.5 h/task → 1028/24 ≈ 43 rounds × 1.5h = ~64h
#   → 48h walltime is tight; bump to 72h if jobs come close to the wall.
#
# Output: /scratch/${USER}/pps_qj/pps_clone_dense  (shared with small/large scripts)
#
# Usage:
#   bash slurm/submit_clone_dense_medium_L.sh [TASK_LO] [TASK_HI] [OUTPUT_DIR] [WALL_TIME]
#
# L-task ranges (dense grid, 514 tasks/L):
#   L=48: tasks 2056..2569
#   L=64: tasks 2570..3083
#
# Examples:
#   All medium-L: bash slurm/submit_clone_dense_medium_L.sh 2056 3083
#   L=64 only:    bash slurm/submit_clone_dense_medium_L.sh 2570 3083
# =============================================================================
set -euo pipefail

TASK_LO=${1:-2056}
TASK_HI=${2:-3083}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_dense}
WALL_TIME=${4:-48:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="cldn_med_${TASK_LO}_${TASK_HI}"
PARTITION="regular"
LOG_DIR="/scratch/${USER}/pps_qj/logs"

CPUS_PER_TASK=5
CONCURRENT_TASKS=24
TOTAL_CORES=$(( CPUS_PER_TASK * CONCURRENT_TASKS ))   # = 120

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

echo "======================================================================"
echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}  [clone-dense medium-L: L=48,64]"
echo "Node: \$(hostname)   Cores: ${TOTAL_CORES} (${CONCURRENT_TASKS} tasks × ${CPUS_PER_TASK} cores)"
echo "Started: \$(date)"
echo "Tasks: ${TASK_LO}..${TASK_HI}  (${N_TASKS} total)"
echo "Output: ${OUTPUT_DIR}"
echo "PPS_RECORD_RENYI=1   PPS_FORCE_RERUN=1"
echo "======================================================================"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PPS_N_WORKERS=${CPUS_PER_TASK}
export PPS_RECORD_RENYI=1
export PPS_FORCE_RERUN=1
export PPS_DTAU_MULT=\${PPS_DTAU_MULT:-1.0}

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

_progress() {
    local t0=\$(date +%s)
    while true; do
        local n=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
        local elapsed=\$(( \$(date +%s) - t0 ))
        echo "[\$(date +%H:%M:%S)] \${n} npz files  (elapsed: \$(( elapsed/3600 ))h\$(( (elapsed%3600)/60 ))m)"
        sleep 300
    done
}
_progress &
PROGRESS_PID=\$!

seq ${TASK_LO} ${TASK_HI} | xargs -P ${CONCURRENT_TASKS} -I{} \
    python -m pps_qj.parallel.worker_clone_dense_pps {} ${OUTPUT_DIR}

kill \${PROGRESS_PID} 2>/dev/null || true

N_DONE=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
N_FAIL=\$(grep -rl '"status": "failed"' ${OUTPUT_DIR}/summary_clone_*.json 2>/dev/null | wc -l)
echo "======================================================================"
echo "Finished: \$(date)"
echo "Total npz files in output dir: \${N_DONE}"
echo "Failed this job: \${N_FAIL}"
echo "======================================================================"
SLURM_SCRIPT

echo "Submitted: ${JOB_NAME}"
echo "  Tasks:        ${TASK_LO}..${TASK_HI} (${N_TASKS} tasks)"
echo "  Concurrency:  ${CONCURRENT_TASKS} tasks × ${CPUS_PER_TASK} cores = ${TOTAL_CORES} cores"
echo "  Wall time:    ${WALL_TIME}   Partition: ${PARTITION}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Monitor:      squeue -u \$USER"
echo "  Log:          tail -f ${LOG_DIR}/${JOB_NAME}_<JOBID>.out"
