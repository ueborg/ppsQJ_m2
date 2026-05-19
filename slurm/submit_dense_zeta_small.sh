#!/bin/bash
# =============================================================================
# submit_dense_zeta_small.sh — Dense small-zeta coverage, L=32..128
#
# PURPOSE:
#   Add zeta in {0.02, 0.03, 0.04, 0.08} at L in {32,48,64,96,128}
#   to make the power-law fit log(lam_c) = phi*log(zeta) + const statistically
#   meaningful (6 zeta points instead of 2 below zeta=0.10).
#
#   Lambda brackets are lam_c*{0.55, 1.00, 1.50} where lam_c ~ 0.497*sqrt(zeta).
#
# 60 tasks, all fast (<4h each). Single node, xargs -P 60 = 300 cores.
# Wall time: 8h (regularshort eligible if available).
#
# Usage:
#   bash slurm/submit_dense_zeta_small.sh
#   bash slurm/submit_dense_zeta_small.sh /scratch/myoutput 08:00:00
# =============================================================================
set -euo pipefail

OUTPUT_DIR=${1:-/scratch/${USER}/pps_qj/pps_dense_zeta}
WALL_TIME=${2:-08:00:00}
JOB_NAME="dz_small"
LOG_DIR="/scratch/${USER}/pps_qj/logs"
CPUS_PER_TASK=5
CONCURRENT_TASKS=60

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

TASK_FILE="${LOG_DIR}/${JOB_NAME}_tasks.txt"
cat > "${TASK_FILE}" << 'TASKS'
32 0.0390 0.0200
32 0.0700 0.0200
32 0.1050 0.0200
48 0.0390 0.0200
48 0.0700 0.0200
48 0.1050 0.0200
64 0.0390 0.0200
64 0.0700 0.0200
64 0.1050 0.0200
96 0.0390 0.0200
96 0.0700 0.0200
96 0.1050 0.0200
128 0.0390 0.0200
128 0.0700 0.0200
128 0.1050 0.0200
32 0.0470 0.0300
32 0.0860 0.0300
32 0.1290 0.0300
48 0.0470 0.0300
48 0.0860 0.0300
48 0.1290 0.0300
64 0.0470 0.0300
64 0.0860 0.0300
64 0.1290 0.0300
96 0.0470 0.0300
96 0.0860 0.0300
96 0.1290 0.0300
128 0.0470 0.0300
128 0.0860 0.0300
128 0.1290 0.0300
32 0.0550 0.0400
32 0.0990 0.0400
32 0.1490 0.0400
48 0.0550 0.0400
48 0.0990 0.0400
48 0.1490 0.0400
64 0.0550 0.0400
64 0.0990 0.0400
64 0.1490 0.0400
96 0.0550 0.0400
96 0.0990 0.0400
96 0.1490 0.0400
128 0.0550 0.0400
128 0.0990 0.0400
128 0.1490 0.0400
32 0.0770 0.0800
32 0.1410 0.0800
32 0.2110 0.0800
48 0.0770 0.0800
48 0.1410 0.0800
48 0.2110 0.0800
64 0.0770 0.0800
64 0.1410 0.0800
64 0.2110 0.0800
96 0.0770 0.0800
96 0.1410 0.0800
96 0.2110 0.0800
128 0.0770 0.0800
128 0.1410 0.0800
128 0.2110 0.0800
TASKS

N_TASKS=$(wc -l < "${TASK_FILE}" | tr -d ' ')

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
echo "Job \${SLURM_JOB_ID} started: \$(date)  node: \$(hostname)"
module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=${CPUS_PER_TASK} PPS_DTAU_MULT=\${PPS_DTAU_MULT:-2.0}
mkdir -p ${OUTPUT_DIR}

cat "${TASK_FILE}" | xargs -P ${CONCURRENT_TASKS} -I{} bash -c '
    read -r L LAM ZETA <<< "{}"
    python -m pps_qj.parallel.worker_fss_direct \$L \$LAM \$ZETA '"${OUTPUT_DIR}"'
'
echo "Done: \$(date)  npz: \$(ls ${OUTPUT_DIR}/fss_*.npz 2>/dev/null | wc -l)"
SLURM

echo "Submitted ${N_TASKS} tasks x ${CPUS_PER_TASK} cores = $((N_TASKS*CPUS_PER_TASK)) cores."
echo "Output: ${OUTPUT_DIR}"
