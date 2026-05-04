#!/bin/bash
# =============================================================================
# submit_clone_v2_large.sh  —  v2 production scan, Node 3
#
# Covers L = 64, 96, 128  (tasks 1200..1919, 720 tasks total)
# Grid: 24 λ × 10 ζ × 3 L = 720 tasks
#       N_c = 200 (L=64), 150 (L=96), 100 (L=128)
#       Intra-task MP: CPUS_PER_TASK=5 (one worker per realisation)
#       Concurrency:   24 tasks × 5 cores = 120 cores on 1 node
#
# Worst-case timing: L=128, α=0.02 (λ=0.02, hardest task)
#   T=100 (capped), δτ = 1/(2×0.02×127) ≈ 0.197, K ≈ 508 steps
#   per step ≈ 100 × ~400ms ≈ 40s
#   per task ≈ 508 × 40s / 5 workers ≈ 4064s ≈ 68 min
#   720 tasks / 24 concurrent = 30 rounds × 68 min ≈ 34h
#   → Wall time 48:00:00 covers this with comfortable margin.
#
# L=64 worst case: ~9min/task; L=96 worst case: ~24min/task — both fast
# relative to L=128.  The L=128 tasks are the bottleneck.
#
# Note on PPS_N_WORKERS
# ----------------------
# N_REAL=5 realisations per task; 5 workers → perfect load balance (each
# worker handles exactly one realisation).  More workers idle, fewer
# serialise realisations.  CPUS_PER_TASK and PPS_N_WORKERS must both be 5.
#
# Usage:
#   bash slurm/submit_clone_v2_large.sh [TASK_LO] [TASK_HI] [OUTPUT_DIR] [WALL_TIME]
#
# L-task ranges (v2 grid, 240 tasks/L):
#   L=64 : tasks 1200..1439
#   L=96 : tasks 1440..1679
#   L=128: tasks 1680..1919
#
# Examples:
#   All large:     bash slurm/submit_clone_v2_large.sh 1200 1919
#   L=64+96 only:  bash slurm/submit_clone_v2_large.sh 1200 1679 <outdir> 24:00:00
#   L=128 only:    bash slurm/submit_clone_v2_large.sh 1680 1919 <outdir> 48:00:00
# =============================================================================
set -euo pipefail

TASK_LO=${1:-1200}
TASK_HI=${2:-1919}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_v2}
WALL_TIME=${4:-48:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="clv2_large_${TASK_LO}_${TASK_HI}"
PARTITION="regular"
LOG_DIR="/scratch/${USER}/pps_qj/logs"

# Intra-task parallelism: 5 cores per task, 24 concurrent tasks = 120 cores.
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
echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}  [clone-v2 large: L=64,96,128]"
echo "Node: \$(hostname)   Cores: ${TOTAL_CORES} (${CONCURRENT_TASKS} tasks × ${CPUS_PER_TASK} cores)"
echo "Started: \$(date)"
echo "Tasks: ${TASK_LO}..${TASK_HI}  (${N_TASKS} total, completed tasks skipped)"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================================"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

# Critical: pin every BLAS/OpenMP backend to 1 thread per worker process.
# Each task spawns 5 single-threaded workers; without this pin every worker
# would try to use all 5 cores allocated to the task, oversubscribing by 5x.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# run_cloning reads PPS_N_WORKERS; set to match CPUS_PER_TASK for N_REAL=5
# (one worker per realisation → 100% parallel efficiency).
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

# xargs -P limits concurrent *task* processes; each task manages its own
# pool of CPUS_PER_TASK workers internally via PPS_N_WORKERS.
seq ${TASK_LO} ${TASK_HI} | xargs -P ${CONCURRENT_TASKS} -I{} \
    python -m pps_qj.parallel.worker_clone_v2_pps {} ${OUTPUT_DIR}

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
