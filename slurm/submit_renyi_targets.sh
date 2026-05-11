#!/bin/bash
# =============================================================================
# submit_renyi_targets.sh - Targeted reruns with PPS_RECORD_RENYI=1
#
# Reruns 30 strategically chosen tasks from the v2 main grid to test the
# free-Dirac CFT prediction c_n = (c/6) * (1 + 1/n) by fitting Renyi
# entropies S_2, S_3 against ln L alongside the existing S_1.
#
# Strategic points (each at L = 32, 48, 64, 96, 128 = 5 tasks):
#   (lam=0.30,  zeta=1.00)  — just below c=1 line at zeta=1
#   (lam=0.35,  zeta=1.00)  — right at c=1 line at zeta=1
#   (lam=0.45,  zeta=1.00)  — at Var(S) peak (inside crossover, zeta=1)
#   (lam=0.325, zeta=0.50)  — at c=1 line at zeta=0.50
#   (lam=0.50,  zeta=0.50)  — inside crossover band at zeta=0.50
#   (lam=0.10,  zeta=0.20)  — at c=1 line at zeta=0.20
#
# Output directory: /scratch/$USER/pps_qj/pps_clone_renyi/
# Output is in NEW directory, does NOT overwrite production data.
#
# Total runtime estimate (with CPUS_PER_TASK=5 and 24 concurrent tasks):
#   Sum of task-hours: ~100 task-h
#   Wall time:         ~6-8h with comfortable margin
#
# Usage:
#   bash slurm/submit_renyi_targets.sh                  # default settings
#   bash slurm/submit_renyi_targets.sh /scratch/myoutput 12:00:00
#
# After completion: scripts/analyze_renyi_targets.py reads the renyi output
# directory and produces the c_n(L) plots and CFT consistency tests.
# =============================================================================
set -euo pipefail

OUTPUT_DIR=${1:-/scratch/${USER}/pps_qj/pps_clone_renyi}
WALL_TIME=${2:-12:00:00}

JOB_NAME="renyi_targets"
PARTITION="regular"
LOG_DIR="/scratch/${USER}/pps_qj/logs"

CPUS_PER_TASK=5
CONCURRENT_TASKS=24
TOTAL_CORES=$(( CPUS_PER_TASK * CONCURRENT_TASKS ))

# Strategic task IDs from make_clone_v2_grid(). Computed offline; the mapping
# from (L, lam, zeta) -> task ID is deterministic in the grid order.
# 6 strategic (lam, zeta) points x 5 L values = 30 tasks.
TASK_IDS=(744 789 796 809 849 866 984 1029 1036 1049 1089 1106 1224 1269 1276 1289 1329 1346 1464 1509 1516 1529 1569 1586 1704 1749 1756 1769 1809 1826)
N_TASKS=${#TASK_IDS[@]}

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

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
echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}  [Renyi + correlation reruns]"
echo "Node: \$(hostname)   Cores: ${TOTAL_CORES} (${CONCURRENT_TASKS} tasks x ${CPUS_PER_TASK} cores)"
echo "Started: \$(date)"
echo "Task IDs: ${TASK_IDS[@]}"
echo "Total tasks: ${N_TASKS}"
echo "Output:   ${OUTPUT_DIR}"
echo "======================================================================"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

# Pin BLAS/OpenMP to 1 thread per worker
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Match worker pool to allocated CPUs
export PPS_N_WORKERS=${CPUS_PER_TASK}

# Default optimisation flags (matching production)
export PPS_DTAU_MULT=\${PPS_DTAU_MULT:-2.0}

# === THE NEW BIT: enable Renyi + correlation function recording ===
export PPS_RECORD_RENYI=1

mkdir -p ${OUTPUT_DIR}

_progress() {
    local t0=\$(date +%s)
    while true; do
        local n=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
        local elapsed=\$(( \$(date +%s) - t0 ))
        echo "[\$(date +%H:%M:%S)] \${n}/${N_TASKS} npz  (elapsed: \$(( elapsed/3600 ))h\$(( (elapsed%3600)/60 ))m)"
        sleep 300
    done
}
_progress &
PROGRESS_PID=\$!

printf '%s\n' ${TASK_IDS[@]} | xargs -P ${CONCURRENT_TASKS} -I{} \
    python -m pps_qj.parallel.worker_clone_v2_pps {} ${OUTPUT_DIR}

kill \${PROGRESS_PID} 2>/dev/null || true

N_DONE=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
echo "======================================================================"
echo "Finished: \$(date)"
echo "Total npz files: \${N_DONE} / ${N_TASKS}"
echo "======================================================================"
SLURM_SCRIPT

echo "Submitted. To monitor:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/${JOB_NAME}_*.out"
