#!/bin/bash
# =============================================================================
# submit_zeta0_scan.sh  —  ζ=0 no-click (postselected) benchmark
#
# Covers all 72 ζ=0 tasks: L=[8,16,24,32,48,64,96,128] × 9 λ values
#
# Worker: worker_zeta0_pps.py (deterministic, no random jumps, N_REAL=1)
# Cost:   O(L^3) per task; L=128 task ~0.3s wall clock.
# All 72 tasks complete in a few minutes even on a single core.
# 120 serial tasks gives ~1 simultaneous round: wall time 00:30:00 is safe.
#
# Output format: zeta0_XXXX.npz (note: different prefix from clone tasks)
#
# Usage:
#   bash slurm/submit_zeta0_scan.sh [TASK_LO] [TASK_HI] [OUTPUT_DIR] [WALL_TIME]
#
# Task ranges per L (9 tasks/L):
#   L=  8:  0.. 8     L= 16:  9..17    L= 24: 18..26    L= 32: 27..35
#   L= 48: 36..44     L= 64: 45..53    L= 96: 54..62    L=128: 63..71
#
# Examples:
#   All:          bash slurm/submit_zeta0_scan.sh 0 71
#   Large L only: bash slurm/submit_zeta0_scan.sh 45 71 <outdir> 00:30:00
# =============================================================================
set -euo pipefail

TASK_LO=${1:-0}
TASK_HI=${2:-71}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_zeta0}
WALL_TIME=${4:-00:30:00}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))
JOB_NAME="zeta0_${TASK_LO}_${TASK_HI}"
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
echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}  [ζ=0 no-click benchmark]"
echo "Node: \$(hostname)   Cores: \${SLURM_NTASKS}   Started: \$(date)"
echo "Tasks: ${TASK_LO}..${TASK_HI}  (${N_TASKS} deterministic, completed skipped)"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================================"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

# Single-threaded per task: scipy expm uses LAPACK internally but is fast
# enough at these sizes without thread pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

seq ${TASK_LO} ${TASK_HI} | xargs -P \${SLURM_NTASKS} -I{} \
    python -m pps_qj.parallel.worker_zeta0_pps {} ${OUTPUT_DIR}

N_DONE=\$(ls ${OUTPUT_DIR}/zeta0_*.npz 2>/dev/null | wc -l)
N_NOCONV=\$(grep -l '"converged": false' ${OUTPUT_DIR}/summary_zeta0_*.json 2>/dev/null | wc -l)
echo "======================================================================"
echo "Finished: \$(date)"
echo "Total npz files: \${N_DONE}  (not-converged: \${N_NOCONV})"
echo "======================================================================"
SLURM_SCRIPT

echo "Submitted: ${JOB_NAME}"
echo "  Tasks:     ${TASK_LO}..${TASK_HI} (${N_TASKS} tasks, ${N_CORES} cores)"
echo "  Wall time: ${WALL_TIME}   Partition: ${PARTITION}"
echo "  Output:    ${OUTPUT_DIR}  (files: zeta0_XXXX.npz)"
echo "  Monitor:   squeue -u \$USER"
