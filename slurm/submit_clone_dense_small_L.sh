#!/bin/bash
# =============================================================================
# submit_clone_dense_small_L.sh  —  Dense fine-grid scan, Node 1
#
# Covers L = 8, 16, 24, 32  (tasks 0..2055, 2056 tasks total)
# Dense grid: 4 sizes × 514 tasks/L = 2056 tasks
#   N_c = 4000 (L=8), 2000 (L=16), 1600 (L=24), 1000 (L=32)
#   Intra-task MP: CPUS_PER_TASK=5 (one worker per realisation)
#   Concurrency:   24 tasks × 5 cores = 120 cores on 1 node
#
# Worst-case timing estimate (with Renyi enabled, n_workers=5):
#   L=32, ζ=1.0 worst case: ~1.8 h/task wallclock
#   2056 tasks / 24 concurrent ≈ 86 rounds × geometric-mean ~25min = ~36h
#   Most tasks much faster (L=8,16,24 finish in minutes)
#   → 24h walltime should suffice with margin given mixed L scheduling.
#
# New observables stored in every .npz (vs. v2):
#   CMI_mean, CMI_err, S_AB_mean, S_AB_err, S_BC_mean, S_B_mean, S_ABC_mean
#   S_renyi_2_mean, S_renyi_3_mean  (only because PPS_RECORD_RENYI=1)
#
# Output: /scratch/${USER}/pps_qj/pps_clone_dense  (shared with medium/large scripts)
# Seeds disjoint from v2/FST/slope by construction (_DENSE_SEED_OFFSET=7e9).
#
# Usage:
#   bash slurm/submit_clone_dense_small_L.sh [TASK_LO] [TASK_HI] [OUTPUT_DIR] [WALL_TIME]
#
# L-task ranges (dense grid, 514 tasks/L):
#   L=8 : tasks    0.. 513
#   L=16: tasks  514..1027
#   L=24: tasks 1028..1541
#   L=32: tasks 1542..2055
#
# Examples:
#   All small-L: bash slurm/submit_clone_dense_small_L.sh 0 2055
#   L=32 only:   bash slurm/submit_clone_dense_small_L.sh 1542 2055
# =============================================================================
set -euo pipefail

TASK_LO=${1:-0}
TASK_HI=${2:-2055}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_dense}
WALL_TIME=${4:-24:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="cldn_small_${TASK_LO}_${TASK_HI}"
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
echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}  [clone-dense small-L: L=8,16,24,32]"
echo "Node: \$(hostname)   Cores: ${TOTAL_CORES} (${CONCURRENT_TASKS} tasks × ${CPUS_PER_TASK} cores)"
echo "Started: \$(date)"
echo "Tasks: ${TASK_LO}..${TASK_HI}  (${N_TASKS} total)"
echo "Output: ${OUTPUT_DIR}"
echo "PPS_RECORD_RENYI=1   PPS_FORCE_RERUN=1"
echo "======================================================================"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

# Critical: pin every BLAS/OpenMP backend to 1 thread per worker process.
# Each task spawns 5 single-threaded workers; without this pin every worker
# would try to use all 5 cores allocated to the task, oversubscribing 5x.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Realisation-level parallelism: one worker per N_REAL=5.
export PPS_N_WORKERS=${CPUS_PER_TASK}

# CRITICAL: enable Renyi-2 / Renyi-3 recording.  This is OFF by default;
# the v2 campaign did NOT set this, which is why Renyi fields are NaN in
# the existing aggregates.  Required for the cleaner-observables analysis.
export PPS_RECORD_RENYI=1

# Force rerun even if .npz exists at that task ID, to guarantee the new
# dense campaign produces entirely fresh data (independent of v2).
# Seeds are also offset by +7e9 vs v2, so collisions are doubly prevented.
export PPS_FORCE_RERUN=1

# Optional δτ multiplier — keep default 1.0 for highest accuracy.
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

# xargs -P limits concurrent task processes; each task manages its own
# pool of CPUS_PER_TASK workers internally via PPS_N_WORKERS.
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
