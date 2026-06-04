#!/bin/bash
# =============================================================================
# submit_clone_dense_L64_backfill.sh  --  finish the 228 missing L=64 tasks
#
# The medium_L dense job ran out of 48h walltime with 228/1028 tasks
# unfinished: all 228 are L=64 (task IDs 2856..3083), the most expensive
# rows in the medium block.  This backfills them into the SAME dense output
# dir so aggregate sees them as part of the dense campaign.
#
# These use the dense grid's N_c=400 at L=64, T=150.  Predicted ~1.8h/task
# (worst ~3.2h); 228 tasks / 24 conc ~= 17h wall -> fits a 24h job easily.
#
# Reads the task IDs from missing_medium_tasks.txt (or a contiguous range).
#
# Usage:
#   bash slurm/submit_clone_dense_L64_backfill.sh [TASK_LO] [TASK_HI] [OUTDIR] [WALL]
# Defaults: 2856 3083  /scratch/$USER/pps_qj/pps_clone_dense  24:00:00
# =============================================================================
set -euo pipefail

TASK_LO=${1:-2856}
TASK_HI=${2:-3083}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_dense}
WALL_TIME=${4:-24:00:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="cldn_L64bf_${TASK_LO}_${TASK_HI}"
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

echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}  [dense L=64 backfill, tasks ${TASK_LO}..${TASK_HI}]"
echo "Started: \$(date)   Output: ${OUTPUT_DIR}"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=${CPUS_PER_TASK}
export PPS_RECORD_RENYI=1
export PPS_FORCE_RERUN=1
export PPS_DTAU_MULT=\${PPS_DTAU_MULT:-1.0}
# Pin the trajectory backend explicitly (see submit_clone_rescue_L128.sh).
# This backfills into the scalar-generated dense campaign, so 'scalar' keeps
# the L=64 row bit-consistent with the rest of pps_clone_dense.
export PPS_BACKEND=\${PPS_BACKEND:-scalar}

mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

# Use the existing DENSE worker (same grid the medium_L job used).
seq ${TASK_LO} ${TASK_HI} | xargs -P ${CONCURRENT_TASKS} -I{} \
    python -m pps_qj.parallel.worker_clone_dense_pps {} ${OUTPUT_DIR}

echo "Finished: \$(date)"
SLURM_SCRIPT

echo "Submitted ${JOB_NAME}: ${N_TASKS} L=64 backfill tasks, wall ${WALL_TIME}"
