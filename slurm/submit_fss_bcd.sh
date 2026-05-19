#!/bin/bash
# Tests B, C, D — 6 tasks, companion to the already-submitted Test A array.
set -euo pipefail

OUTPUT_DIR=${1:-/scratch/${USER}/pps_qj/pps_fss_test}
WALL_TIME=${2:-48:00:00}

JOB_NAME="fss_bcd"
LOG_DIR="/scratch/${USER}/pps_qj/logs"
CPUS_PER_TASK=5
CONCURRENT_TASKS=6

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

TASK_FILE="${LOG_DIR}/${JOB_NAME}_tasks.txt"
cat > "${TASK_FILE}" << 'TASKS'
192 0.1000 0.20
256 0.1000 0.20
192 0.4500 1.00
256 0.4500 1.00
192 0.1550 0.20
256 0.1550 0.20
TASKS

sbatch << SLURM
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=${CONCURRENT_TASKS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${WALL_TIME}
#SBATCH --partition=regular
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rug.nl

set -euo pipefail
echo "Job \${SLURM_JOB_ID}: $(date)"
module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PPS_N_WORKERS=${CPUS_PER_TASK} PPS_DTAU_MULT=2.0
mkdir -p ${OUTPUT_DIR}
cat "${TASK_FILE}" | xargs -P ${CONCURRENT_TASKS} -I{} bash -c '
    read -r L LAM ZETA <<< "{}"
    python -m pps_qj.parallel.worker_fss_direct \$L \$LAM \$ZETA '"${OUTPUT_DIR}"'
'
echo "Done: $(date)  npz: \$(ls ${OUTPUT_DIR}/fss_*.npz 2>/dev/null | wc -l)"
SLURM

echo "Submitted: 6 tasks x 5 cores = 30 cores"