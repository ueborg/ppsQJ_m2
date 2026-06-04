#!/bin/bash
# =============================================================================
# submit_bench_production.sh  --  production-mimic backend benchmark
#
# Runs the REAL rescue worker (worker_clone_rescue_pps) at the true production
# shape (L=128, N_c=250, T=100 from the rescue grid) on a few real task IDs,
# once with PPS_BACKEND=scalar and once with PPS_BACKEND=batched, into a
# THROWAWAY scratch dir.  This exercises the exact production code path
# (realisation pool, npz I/O, aggregation fields) -- not a synthetic call.
#
# To keep each task affordable, PPS_DTAU_MULT shrinks n_steps by that factor
# WITHOUT touching L or N_c (so the scalar/batched ratio and per-step cost
# structure are preserved).  Production per-task wall ~= measured_wall * MULT.
#
#   default MULT=5  ->  each task ~1/5 of production (~2.5-3h instead of ~13h)
#   2 task IDs x 2 backends x ~3h  ~=  ~12h job.  Adjust NTASKS/MULT/WALL.
#
# Usage:
#   bash slurm/submit_bench_production.sh [TASK_LO] [NTASKS] [MULT] [WALL]
# Defaults: 0 2 5 16:00:00   (task IDs 0,1 of the rescue grid)
# =============================================================================
set -euo pipefail

TASK_LO=${1:-0}
NTASKS=${2:-2}
MULT=${3:-5}
WALL_TIME=${4:-16:00:00}

TASK_HI=$(( TASK_LO + NTASKS - 1 ))
OUTPUT_DIR=/scratch/${USER}/pps_qj/pps_bench
JOB_NAME="clbench_${TASK_LO}_${TASK_HI}_m${MULT}"
PARTITION="regular"
LOG_DIR="/scratch/${USER}/pps_qj/logs"
CPUS_PER_TASK=5

sbatch <<SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${WALL_TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rug.nl
set -euo pipefail

echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}"
echo "Production-mimic: L=128/N_c=250/T=100 rescue tasks ${TASK_LO}..${TASK_HI}, MULT=${MULT}"
echo "Started: \$(date)   Output(throwaway): ${OUTPUT_DIR}"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=${CPUS_PER_TASK}
export PPS_RECORD_RENYI=1
export PPS_FORCE_RERUN=1
export PPS_DTAU_MULT=${MULT}

mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

echo ""
echo "=== per-task wall times (seconds); production ~= wall * ${MULT} ==="
for BACKEND in scalar batched; do
  export PPS_BACKEND=\${BACKEND}
  OUT=${OUTPUT_DIR}/\${BACKEND}
  mkdir -p \${OUT}
  for TID in \$(seq ${TASK_LO} ${TASK_HI}); do
    T0=\$SECONDS
    python -m pps_qj.parallel.worker_clone_rescue_pps \${TID} \${OUT} >/dev/null 2>&1 || echo "  (task \${TID} \${BACKEND} returned nonzero)"
    DT=\$(( SECONDS - T0 ))
    echo "  backend=\${BACKEND} task=\${TID}: \${DT}s  (prod est \$(( DT * ${MULT} / 3600 ))h)"
  done
done

echo ""
echo "Finished: \$(date)"
echo "Throwaway outputs in ${OUTPUT_DIR} -- safe to rm -rf after reading walltimes."
SLURM_SCRIPT

echo "Submitted ${JOB_NAME}: tasks ${TASK_LO}..${TASK_HI}, MULT=${MULT}, wall ${WALL_TIME}"
echo "Compare the scalar vs batched per-task seconds in the .out log."
