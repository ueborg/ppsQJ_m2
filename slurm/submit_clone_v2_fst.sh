#!/bin/bash
# ============================================================================================ 
# submit_clone_v2_fst.sh  -  Finite-size test: L=192, L=256
#
# Tests Scenario A vs Scenario B for the empirical separatrix at zeta~0.143
# in the L<=128 aggregate.  See pps_qj/parallel/grid_pps.py for context.
#
# Grid:
#   L:    [192, 256]                                      (2 sizes)
#   lam:  [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]  (27points)
#   zeta: [0.05, 0.10, 0.14, 0.18, 0.50, 1.00]        (6 points)
# Total: 84 tasks, ids 0..83.
#
#   L=192: tasks  0..41   (N_c=80, T_cap=80)
#   L=256: tasks 42..83   (N_c=40, T_cap=50)
#
# Worker: worker_clone_v2_fst_pps.py (routes to task_params_clone_fst)
# Concurrency: 24 tasks x 5 cores = 120 cores
#
# Wall-time budget (L^4.8 cache-thrashing estimate):
#   L=192 worst case (alpha=0.05, T=80): ~22 h/task
#   L=256 worst case (alpha=0.05, T=50): ~31 h/task
# At 24 concurrent tasks per node:
#   L=192 block (42 tasks): 2 rounds x 22h = 44 h
#   L=256 block (42 tasks): 2 rounds x 31h = 62 h
#
# Recommended:
#   bash slurm/submit_clone_v2_fst.sh  0 41 /scratch/${USER}/pps_qj/pps_clone_v2_fst 48:00:00
#   bash slurm/submit_clone_v2_fst.sh 42 83 /scratch/${USER}/pps_qj/pps_clone_v2_fst 72:00:00
# ============================================================================================

set -euo pipefail

TASK_LO=${1:-0}
TASK_HI=${2:-83}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_v2_fst}
WALL_TIME=${4:-48:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="clv2_fst_${TASK_LO}_${TASK_HI}"
PARTITION="regular"
LOG_DIR="/scratch/${USER}/pps_qj/logs"

CPUS_PER_TASK=5
CONCURRENT_TASKS=24
TOTAL_CORES=$(( CPUS_PER_TASK * CONCURRENT_TASKS ))

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

echo "================================================================================"
echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}"
echo "Node: \$(hostname)   Cores: ${TOTAL_CORES}"
echo "Started: \$(date)"
echo "Tasks: ${TASK_LO}..${TASK_HI}  (${N_TASKS} total)"
echo "Output: ${OUTPUT_DIR}"
echo "================================================================================="

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PPS_N_WORKERS=${CPUS_PER_TASK}
export PPS_DTAU_MULT=\${PPS_DTAU_MULT:-1.0}

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

_progress() {
    local t0=\$(date +%s)
    while true; do
        local n=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
        local elapsed=\$(( \$(date +%s) - t0 ))
        echo "[\$(date +%H:%M:%S)] \${n} npz files (elapsed: \$(( elapsed/3600 ))h\$(( (elapsed%3600)/60 ))m)"
        sleep 600
    done
}
_progress &
PROGRESS_PID=\$!

seq ${TASK_LO} ${TASK_HI} | xargs -P ${CONCURRENT_TASKS} -I{} \\
    python -m pps_qj.parallel.worker_clone_v2_fst_pps {} ${OUTPUT_DIR}

kill \${PROGRESS_PID} 2>/dev/null || true

N_DONE=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
echo "================================================================================="
echo "Finished: \$(date)   Total npz: \${N_DONE}"
echo "================================================================================"
SLURM_SCRIPT

echo "Submitted: ${JOB_NAME}"
echo "  Tasks:        ${TASK_LO}..${TASK_HI} (${N_TASKS} tasks)"
echo "  Concurrency:  ${CONCURRENT_TASKS} tasks x ${CPUS_PER_TASK} cores = ${TOTAL_CORES} cores"
echo "  Wall time:    ${WALL_TIME}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Monitor:      squeue -u \$USER"
