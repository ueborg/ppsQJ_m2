#!/bin/bash
# =============================================================================
# submit_clone_scan.sh  (updated: 120-core version, tighter wall times)
#
# Usage:
#   bash slurm/submit_clone_scan.sh [TASK_LO] [TASK_HI] [OUTPUT_DIR] [WALL_TIME]
#
# Preset invocations:
#   L=16 finish:  bash slurm/submit_clone_scan.sh 255 509 <outdir> 06:00:00
#   L=32 run:     bash slurm/submit_clone_scan.sh 510 764 <outdir> 24:00:00
#   L=64 run:     bash slurm/submit_clone_scan.sh 765 1019 <outdir> 48:00:00
#
# L=64 wall-time note
# -------------------
# Profiling shows ~43.8 ms/clone at L=64 (L^4.8 effective scaling due to
# L2-cache thrashing on 256 KB model matrices in the serial clone loop).
# N_c=100 and N_REAL=2 (set automatically by NREAL_FOR_L in grid_pps.py)
# gives ~15.7 h/task at alpha=0.4 and ~35.3 h/task at alpha=0.9.  48 h
# covers all tasks; check your partition's MaxWallDuration before submitting.
#
# Completed tasks are skipped automatically (idempotency guard in worker).
# =============================================================================
set -euo pipefail

TASK_LO=${1:-255}
TASK_HI=${2:-509}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_scan_v2}
WALL_TIME=${4:-06:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="pps_clone_${TASK_LO}_${TASK_HI}"
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
echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}"
echo "Node: \$(hostname)   Cores: \${SLURM_NTASKS}   Started: \$(date)"
echo "Tasks: ${TASK_LO}..${TASK_HI}  (${N_TASKS} total, completed tasks skipped)"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================================"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

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

seq ${TASK_LO} ${TASK_HI} | xargs -P \${SLURM_NTASKS} -I{} \
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
echo "  Tasks:     ${TASK_LO}..${TASK_HI} (${N_TASKS} tasks, ${N_CORES} cores)"
echo "  Wall time: ${WALL_TIME}   Partition: ${PARTITION}"
echo "  Monitor:   squeue -u \$USER"
echo "  Log:       tail -f ${LOG_DIR}/${JOB_NAME}_<JOBID>.out"
