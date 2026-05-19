#!/bin/bash
# =============================================================================
# submit_fss_test.sh  —  Targeted L=192, 256 runs for the FSS test suite
#
# PURPOSE:
#   Four distinct tests, all run in a single job on one node:
#
#   TEST A — 1/L² lambda_c shift scaling (18 tasks)
#     zeta=0.30, L={192,256}, lam={0.20,0.25,0.30}   bracket lam_c~0.237
#     zeta=0.50, L={192,256}, lam={0.30,0.35,0.40}   bracket lam_c~0.334
#     zeta=1.00, L={192,256}, lam={0.30,0.35,0.40}   bracket lam_c~0.364
#     Prediction (theory/qj_marginal_chiral_correction.md §5):
#       lam_c(zeta=0.30, L=128) shift = -0.127 -> at L=256: -0.032 (4x smaller)
#
#   TEST B — volume-law check at strong PPS (2 tasks)
#     (lam=0.10, zeta=0.20) at L=192, 256
#     S_1/ln L plateau at larger L kills the "anomaly" interpretation once
#     and for all; if S grows as ln L the plateau rises, if volume-law it falls.
#
#   TEST C — Renyi ratio convergence near transition (2 tasks)
#     (lam=0.45, zeta=1.00) at L=192, 256
#     Watch whether c_2/c_1 and c_3/c_1 shift toward 0.75/0.667 at larger L
#     (finite-size correction) or hold at 0.79/0.71 (genuine non-free-Dirac).
#
#   TEST D — universal-class cross-zeta comparison (2 tasks)
#     (lam=0.155, zeta=0.20) at L=192, 256 — lambda/lambda_c ~ 0.70 at zeta=0.20
#     Compare Renyi ratios vs (lam=0.35, zeta=1.00) already in Test A.
#     Same reduced distance from transition -> same ratios IF universality holds.
#
#   NOTE on Binder collapse (recommendation 4):
#     The lambda_c(zeta, L)*sqrt(L) vs zeta*L FSS collapse uses the c_eff=1
#     crossing from the existing S_mean data — no new observables needed.
#     A proper Binder cumulant B_4 requires the 4th moment of S, which the
#     cloning worker does not currently record. That is a separate code change.
#     Tests A-D above provide the input (larger L) for the crossing analysis.
#
# TOTAL: 24 tasks, all run in parallel on one node via xargs -P 24.
#
# WALL TIME: L=192 worst case ~22h, L=256 worst case ~31h -> request 48h.
#
# Usage:
#   bash slurm/submit_fss_test.sh
#   bash slurm/submit_fss_test.sh /scratch/myoutput 48:00:00
# =============================================================================
set -euo pipefail

OUTPUT_DIR=${1:-/scratch/${USER}/pps_qj/pps_fss_test}
WALL_TIME=${2:-48:00:00}

JOB_NAME="fss_test"
PARTITION="regular"
LOG_DIR="/scratch/${USER}/pps_qj/logs"

CPUS_PER_TASK=5
CONCURRENT_TASKS=24
TOTAL_CORES=$(( CPUS_PER_TASK * CONCURRENT_TASKS ))

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

TASK_FILE="${LOG_DIR}/${JOB_NAME}_tasks.txt"
cat > "${TASK_FILE}" << 'TASKS'
192 0.2000 0.30
256 0.2000 0.30
192 0.2500 0.30
256 0.2500 0.30
192 0.3000 0.30
256 0.3000 0.30
192 0.3000 0.50
256 0.3000 0.50
192 0.3500 0.50
256 0.3500 0.50
192 0.4000 0.50
256 0.4000 0.50
192 0.3000 1.00
256 0.3000 1.00
192 0.3500 1.00
256 0.3500 1.00
192 0.4000 1.00
256 0.4000 1.00
192 0.1000 0.20
256 0.1000 0.20
192 0.4500 1.00
256 0.4500 1.00
192 0.1550 0.20
256 0.1550 0.20
TASKS

N_TASKS=$(wc -l < "${TASK_FILE}" | tr -d ' ')

echo "Tasks (${N_TASKS} total):"
echo "  TEST A (1/L^2 shift): 18 tasks across zeta=0.30,0.50,1.00"
echo "  TEST B (volume-law check): lam=0.10 zeta=0.20 at L=192,256"
echo "  TEST C (Renyi ratio convergence): lam=0.45 zeta=1.00 at L=192,256"
echo "  TEST D (universal class): lam=0.155 zeta=0.20 at L=192,256"
echo ""
cat "${TASK_FILE}"
echo ""

sbatch << SLURM_SCRIPT
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
echo "Job \${SLURM_JOB_ID}: ${JOB_NAME}"
echo "Node: \$(hostname)   Cores: ${TOTAL_CORES} (${CONCURRENT_TASKS} x ${CPUS_PER_TASK})"
echo "Started: \$(date)"
echo "Tasks: ${N_TASKS}   Output: ${OUTPUT_DIR}"
echo "======================================================================"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=${CPUS_PER_TASK}
export PPS_DTAU_MULT=\${PPS_DTAU_MULT:-2.0}

mkdir -p ${OUTPUT_DIR}

_progress() {
    local t0=\$(date +%s)
    while true; do
        local n=\$(ls ${OUTPUT_DIR}/fss_*.npz 2>/dev/null | wc -l)
        local elapsed=\$(( \$(date +%s) - t0 ))
        echo "[\$(date +%H:%M:%S)] \${n}/${N_TASKS} npz  (elapsed: \$(( elapsed/3600 ))h\$(( (elapsed%3600)/60 ))m)"
        sleep 600
    done
}
_progress &
PROGRESS_PID=\$!

cat "${TASK_FILE}" | xargs -P ${CONCURRENT_TASKS} -I{} bash -c '
    read -r L LAM ZETA <<< "{}"
    python -m pps_qj.parallel.worker_fss_direct \$L \$LAM \$ZETA '"${OUTPUT_DIR}"'
'

kill \${PROGRESS_PID} 2>/dev/null || true

N_DONE=\$(ls ${OUTPUT_DIR}/fss_*.npz 2>/dev/null | wc -l)
echo "======================================================================"
echo "Finished: \$(date)   npz: \${N_DONE} / ${N_TASKS}"
echo "======================================================================"
SLURM_SCRIPT

echo "Submitted. ${N_TASKS} tasks x ${CPUS_PER_TASK} cores = ${TOTAL_CORES} cores on one node."
echo "Monitor:  squeue -u \$USER"
echo "Progress: tail -f ${LOG_DIR}/${JOB_NAME}_*.out"
