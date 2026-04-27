#!/bin/bash
# =============================================================================
# submit_clone_scan_L64.sh  —  L=64 specific submission with intra-task MP
#
# Why a separate script
# ---------------------
# At L=64 the per-clone trajectory cost jumped from 1.51 ms (L=32) to 43.8 ms
# (~30x), driven by L2-cache thrashing on the 256 KB model matrices in the
# serial clone loop.  That made N_c=200, N_REAL=5 cost 78.5 h/task at α=0.4
# and 169 h/task at α=0.9 — both infeasible on a single core.
#
# The fix is intra-task multiprocessing: each worker process holds a cached
# copy of the model in a global, so the L2 working set is per-core and stays
# warm.  With 8 cores per task we keep N_c=200, N_REAL=5 across the full λ
# range and the worst-case task drops to ~12 h.
#
# Layout: 1 node × 120 cores → 15 concurrent tasks × 8 cores/task.
#
# Critical: PPS_N_WORKERS=8 + OMP_NUM_THREADS=1 + MKL_NUM_THREADS=1.
# Without the BLAS pin, each of the 8 worker processes will try to launch
# its own thread pool and the cores oversubscribe by 8x.
#
# Usage:
#   bash slurm/submit_clone_scan_L64.sh [TASK_LO] [TASK_HI] [OUTPUT_DIR] [WALL_TIME]
#
# L=64 task range: 765..1019 (255 tasks).  Default wall time: 24:00:00.
# Completed tasks are skipped automatically (idempotency guard in worker).
# =============================================================================
set -euo pipefail

TASK_LO=${1:-765}
TASK_HI=${2:-1019}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_scan_v2}
WALL_TIME=${4:-24:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="pps_clone_L64_${TASK_LO}_${TASK_HI}"
PARTITION="regular"
LOG_DIR="/scratch/${USER}/pps_qj/logs"

# Intra-task parallelism
CPUS_PER_TASK=8
CONCURRENT_TASKS=15        # 8 * 15 = 120 cores total
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

echo "======================================================================"
echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}"
echo "Node: \$(hostname)   Cores: ${TOTAL_CORES} (${CONCURRENT_TASKS} tasks × ${CPUS_PER_TASK} cores)"
echo "Started: \$(date)"
echo "Tasks: ${TASK_LO}..${TASK_HI}  (${N_TASKS} total, completed tasks skipped)"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================================"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

# Pin all BLAS / OpenMP backends to a single thread per worker process.
# With CPUS_PER_TASK=8 and PPS_N_WORKERS=8 we want 8 single-threaded workers,
# not 8 workers × N_threads each.  Without this pin every worker would try
# to use the full node, oversubscribing the cores by ~8x.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# run_cloning reads PPS_N_WORKERS and dispatches the per-step clone evolution
# to a Pool of that size (see _n_workers_from_env in worker_clone_pps.py).
export PPS_N_WORKERS=${CPUS_PER_TASK}

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Progress reporter every 5 min
_progress() {
    local t0=\$(date +%s)
    while true; do
        local n=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
        local elapsed=\$(( \$(date +%s) - t0 ))
        echo "[\$(date +%H:%M:%S)] \${n} npz files in output dir  (elapsed: \$(( elapsed/3600 ))h\$(( (elapsed%3600)/60 ))m)"
        sleep 300
    done
}
_progress &
PROGRESS_PID=\$!

# xargs -P limits concurrent task processes; each task forks its own pool of
# CPUS_PER_TASK workers internally.
seq ${TASK_LO} ${TASK_HI} | xargs -P ${CONCURRENT_TASKS} -I{} \
    python -m pps_qj.parallel.worker_clone_pps {} ${OUTPUT_DIR}

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
echo "  Monitor:      squeue -u \$USER"
echo "  Log:          tail -f ${LOG_DIR}/${JOB_NAME}_<JOBID>.out"
