#!/bin/bash
# =============================================================================
# submit_dense_zeta_large.sh — Dense small-zeta coverage, L=192,256
#
# Companion to submit_dense_zeta_small.sh. Same zeta values and lambda
# brackets, extended to L=192 and L=256.
#
# 24 tasks, up to 31h each. Job array, each element 5 cores.
# Wall time: 48h (regularlong required).
#
# Usage:
#   bash slurm/submit_dense_zeta_large.sh
#   bash slurm/submit_dense_zeta_large.sh /scratch/myoutput 48:00:00
# =============================================================================
set -euo pipefail

OUTPUT_DIR=${1:-/scratch/${USER}/pps_qj/pps_dense_zeta}
WALL_TIME=${2:-48:00:00}
JOB_NAME="dz_large"
LOG_DIR="/scratch/${USER}/pps_qj/logs"
CPUS_PER_TASK=5
MAX_CONCURRENT=24

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

TASK_FILE="${LOG_DIR}/${JOB_NAME}_tasks.txt"
cat > "${TASK_FILE}" << 'TASKS'
192 0.0390 0.0200
256 0.0390 0.0200
192 0.0700 0.0200
256 0.0700 0.0200
192 0.1050 0.0200
256 0.1050 0.0200
192 0.0470 0.0300
256 0.0470 0.0300
192 0.0860 0.0300
256 0.0860 0.0300
192 0.1290 0.0300
256 0.1290 0.0300
192 0.0550 0.0400
256 0.0550 0.0400
192 0.0990 0.0400
256 0.0990 0.0400
192 0.1490 0.0400
256 0.1490 0.0400
192 0.0770 0.0800
256 0.0770 0.0800
192 0.1410 0.0800
256 0.1410 0.0800
192 0.2110 0.0800
256 0.2110 0.0800
TASKS

N_TASKS=$(wc -l < "${TASK_FILE}" | tr -d ' ')

sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --array=0-$((N_TASKS - 1))%${MAX_CONCURRENT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${WALL_TIME}
#SBATCH --partition=regular
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%A_%a.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rug.nl

set -euo pipefail
read -r L LAM ZETA <<< \$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${TASK_FILE}")
echo "Array \${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}: L=\${L} lam=\${LAM} zeta=\${ZETA}  node=\$(hostname)  \$(date)"
module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=${CPUS_PER_TASK} PPS_DTAU_MULT=\${PPS_DTAU_MULT:-2.0}
mkdir -p ${OUTPUT_DIR}
python -m pps_qj.parallel.worker_fss_direct \${L} \${LAM} \${ZETA} ${OUTPUT_DIR}
echo "Done: \$(date)"
SLURM

echo "Submitted ${N_TASKS}-element array (max ${MAX_CONCURRENT} concurrent, ${CPUS_PER_TASK} cores each)."
echo "Output: ${OUTPUT_DIR}"
