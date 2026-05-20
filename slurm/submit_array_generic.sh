#!/bin/bash
# =============================================================================
# submit_array_generic.sh — Generic job-array launcher for worker_fss_direct
#
# Not called directly. Sourced by run_A/B/C scripts, or call with:
#   bash submit_array_generic.sh TASK_FILE OUTPUT_DIR WALL_TIME RENYI JOB_NAME
#
# Arguments:
#   $1  TASK_FILE   — path to file with one "L lam zeta" per line
#   $2  OUTPUT_DIR  — where .npz files are written
#   $3  WALL_TIME   — e.g. 08:00:00 or 48:00:00
#   $4  RENYI       — 0 or 1 (sets PPS_RECORD_RENYI)
#   $5  JOB_NAME    — SLURM job name
#   $6  PARTITION   — SLURM partition(s), e.g. "regular" or "regular,parallel"
# =============================================================================
set -euo pipefail

TASK_FILE="${1:?need TASK_FILE}"
OUTPUT_DIR="${2:?need OUTPUT_DIR}"
WALL_TIME="${3:-08:00:00}"
RENYI="${4:-0}"
JOB_NAME="${5:-fss_array}"
PARTITION="${6:-regular,parallel}"

CPUS_PER_TASK=5
LOG_DIR="/scratch/${USER}/pps_qj/logs"
N_TASKS=$(grep -c . "${TASK_FILE}")
ARRAY_SPEC="0-$((N_TASKS - 1))"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "  Submitting: ${JOB_NAME}"
echo "  Tasks: ${N_TASKS}  Wall: ${WALL_TIME}  Renyi: ${RENYI}"
echo "  Partition: ${PARTITION}  Output: ${OUTPUT_DIR}"

sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --array=${ARRAY_SPEC}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${WALL_TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%A_%a.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${USER}@rug.nl

set -euo pipefail
read -r L LAM ZETA <<< \$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${TASK_FILE}")
echo "\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}: L=\${L} lam=\${LAM} zeta=\${ZETA}  node=\$(hostname)  \$(date)"
module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=${CPUS_PER_TASK}
export PPS_DTAU_MULT=\${PPS_DTAU_MULT:-2.0}
export PPS_RECORD_RENYI=${RENYI}
mkdir -p ${OUTPUT_DIR}
python -m pps_qj.parallel.worker_fss_direct \${L} \${LAM} \${ZETA} ${OUTPUT_DIR}
echo "Done: \$(date)"
SLURM
