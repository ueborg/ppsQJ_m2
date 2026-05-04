#!/bin/bash
# =============================================================================
# submit_clone_v2_small.sh  —  v2 production scan, Node 1
#
# Covers L = 8, 16, 24  (tasks 0..719, 720 tasks total)
# Grid: 24 λ × 10 ζ × 3 L = 720 tasks
#       N_c = 2000 (L=8), 1000 (L=16), 800 (L=24)
#       Serial: 1 core per task, 120 concurrent tasks on 1 node
#
# Worst-case wall time: L=24, α=0.02, T=200, N_c=800, N_REAL=5
#   → δτ = 1/(2×0.02×23) ≈ 1.09, K ≈ 200/1.09 ≈ 184 steps
#   → per step ≈ 800 × ~0.7ms ≈ 0.56s
#   → per task ≈ 184 × 0.56s × 5 real = 515s ≈ 9 min
#   → 720 tasks / 120 cores ≈ 6 rounds × 9 min ≈ 54 min
#   → Safe wall time: 06:00:00
#
# Usage:
#   bash slurm/submit_clone_v2_small.sh [TASK_LO] [TASK_HI] [OUTPUT_DIR] [WALL_TIME]
#
# L-task ranges (v2 grid, 240 tasks/L):
#   L=8 :  tasks 0..239
#   L=16:  tasks 240..479
#   L=24:  tasks 480..719
#
# Examples:
#   All small:   bash slurm/submit_clone_v2_small.sh 0 719
#   L=24 only:   bash slurm/submit_clone_v2_small.sh 480 719
# =============================================================================
set -euo pipefail

TASK_LO=${1:-0}
TASK_HI=${2:-719}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_v2}
WALL_TIME=${4:-06:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="clv2_small_${TASK_LO}_${TASK_HI}"
PARTITION="regular"
N_CORES=120
LOG_DIR="/scratch/${USER}/pps_qj/logs"

sbatch <<SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=${N_CORES}
#SBATCH --cpus-per-task=1
#SBATCH --time=${WALL_TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rug.nl

set -euo pipefail

echo "======================================================================"
echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}  [clone-v2 small: L=8,16,24]"
echo "Node: \$(hostname)   Cores: \${SLURM_NTASKS}   Started: \$(date)"
echo "Tasks: ${TASK_LO}..${TASK_HI}  (${N_TASKS} total, completed tasks skipped)"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================================"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

# Pin BLAS to 1 thread; with 120 serial tasks any thread pool oversubscribes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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

seq ${TASK_LO} ${TASK_HI} | xargs -P \${SLURM_NTASKS} -I{} \
    python -m pps_qj.parallel.worker_clone_v2_pps {} ${OUTPUT_DIR}

kill \${PROGRESS_PID} 2>/dev/null || true

N_DONE=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
echo "======================================================================"
echo "Finished: \$(date)"
echo "Total npz files in output dir: \${N_DONE}"
echo "======================================================================"
SLURM_SCRIPT

echo "Submitted: ${JOB_NAME}"
echo "  Tasks:     ${TASK_LO}..${TASK_HI} (${N_TASKS} tasks, ${N_CORES} cores)"
echo "  Wall time: ${WALL_TIME}   Partition: ${PARTITION}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Monitor:   squeue -u \$USER"
echo "  Log:       tail -f ${LOG_DIR}/${JOB_NAME}_<JOBID>.out"
