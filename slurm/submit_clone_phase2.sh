#!/bin/bash
# =============================================================================
# submit_clone_phase2.sh  —  Phase-2 high-L supplement (L=160/192/256)
#
# Decisive small-zeta window (zeta in {0.10,0.15,0.20,0.25,0.30}) x 13-point
# lambda mesh centered on the MEASURED dense lambda_c^(128)(zeta).
# Runs worker_clone_phase2_pps (thin shim over the clone worker; saves B_L +
# full CMI tripartition + Renyi-2/3, same field set as the dense worker).
#
# PREREQUISITE: _PHASE2_CENTERS_VERIFIED must be True in grid_pps.py (set
# 2026-06-11 from the ladder/dense 64,128 crossings). The grid builder warns
# loudly otherwise.
#
# Task layout (L outer, zeta middle, lambda inner): 65 tasks/L, 195 total.
#   L=160: tasks   0.. 64    N_c=400
#   L=192: tasks  65..129    N_c=300
#   L=256: tasks 130..194    N_c=250
#
# Usage:
#   bash slurm/submit_clone_phase2.sh [TASK_LO] [TASK_HI] [OUTPUT_DIR] [WALL_TIME]
# Examples:
#   L=160 only (recommended first wave):
#     bash slurm/submit_clone_phase2.sh 0 64 /scratch/$USER/pps_qj/pps_clone_phase2 120:00:00
#   L=192:  bash slurm/submit_clone_phase2.sh 65 129  <outdir> 120:00:00
#   L=256:  bash slurm/submit_clone_phase2.sh 130 194 <outdir> 168:00:00
# =============================================================================
set -euo pipefail

TASK_LO=${1:-0}
TASK_HI=${2:-64}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_phase2}
WALL_TIME=${4:-120:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="clph2_${TASK_LO}_${TASK_HI}"
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

echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}  tasks ${TASK_LO}..${TASK_HI} (${N_TASKS})"
echo "Started: \$(date)"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=${CPUS_PER_TASK}
export PPS_RECORD_RENYI=1
export PPS_FORCE_RERUN=\${PPS_FORCE_RERUN:-0}
export PPS_DTAU_MULT=\${PPS_DTAU_MULT:-1.0}

mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

seq ${TASK_LO} ${TASK_HI} | xargs -P ${CONCURRENT_TASKS} -I{} \
    python -m pps_qj.parallel.worker_clone_phase2_pps {} ${OUTPUT_DIR}

echo "Finished: \$(date)  npz: \$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)"
SLURM_SCRIPT

echo "Submitted: ${JOB_NAME}  tasks ${TASK_LO}..${TASK_HI} (${N_TASKS})  wall ${WALL_TIME}  partition ${PARTITION}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Monitor: squeue -u \$USER"
