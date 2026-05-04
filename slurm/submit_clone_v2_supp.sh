#!/bin/bash
# =============================================================================
# submit_clone_v2_supp.sh  —  v2 supplement scan (low-λ + large-ζ tasks)
#
# Covers supplement tasks 0..299 (300 tasks total):
#   Block A (0..59):   low-λ small-ζ — L=[32,48,64,96,128], λ=[0.005..0.075]
#   Block B (60..299): large-ζ      — L=[32,48,64,96,128], ζ=[0.90,0.95]
#
# Worker: worker_clone_v2_supp_pps.py (routes to task_params_clone_v2_supp)
# Concurrency: 24 tasks × 5 cores = 120 cores, same as the v2 large script
#
# Wall time: 24h covers worst case (L=128 task near transition)
#
# Usage:
#   bash slurm/submit_clone_v2_supp.sh [TASK_LO] [TASK_HI] [OUTPUT_DIR] [WALL_TIME]
#
# Block ranges:
#   Block A (low-λ):  tasks 0..59
#   Block B (large-ζ): tasks 60..299
#
# Examples:
#   Full supplement:  bash slurm/submit_clone_v2_supp.sh 0 299
#   Block A only:     bash slurm/submit_clone_v2_supp.sh 0 59
#   Block B only:     bash slurm/submit_clone_v2_supp.sh 60 299
# =============================================================================
set -euo pipefail

TASK_LO=${1:-0}
TASK_HI=${2:-299}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_v2_supp}
WALL_TIME=${4:-24:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="clv2_supp_${TASK_LO}_${TASK_HI}"
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
echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}  [clone-v2 supplement]"
echo "Node: \$(hostname)   Cores: ${TOTAL_CORES}   Started: \$(date)"
echo "Tasks: ${TASK_LO}..${TASK_HI}  (${N_TASKS} total, completed tasks skipped)"
echo "Output: ${OUTPUT_DIR}"
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
    python -m pps_qj.parallel.worker_clone_v2_supp_pps {} ${OUTPUT_DIR}

kill \${PROGRESS_PID} 2>/dev/null || true

N_DONE=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
echo "======================================================================"
echo "Finished: \$(date)   Total npz: \${N_DONE}"
echo "======================================================================"
SLURM_SCRIPT

echo "Submitted: ${JOB_NAME}"
echo "  Tasks:        ${TASK_LO}..${TASK_HI} (${N_TASKS} tasks)"
echo "  Concurrency:  ${CONCURRENT_TASKS} tasks × ${CPUS_PER_TASK} cores = ${TOTAL_CORES} cores"
echo "  Wall time:    ${WALL_TIME}   Partition: ${PARTITION}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Monitor:      squeue -u \$USER"
