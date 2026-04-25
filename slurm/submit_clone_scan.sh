#!/bin/bash
# =============================================================================
# submit_clone_scan.sh
#
# Submit a full-node cloning scan job to the Habrok SLURM scheduler.
# The job runs entirely in the background — no interactive session needed.
#
# Usage:
#   bash slurm/submit_clone_scan.sh [TASK_LO] [TASK_HI] [OUTPUT_DIR]
#
#   TASK_LO    first task id to run (default: 0)
#   TASK_HI    last task id  to run (default: 1019, full expanded grid)
#   OUTPUT_DIR scratch output path  (default: /scratch/$USER/pps_qj/pps_clone_scan)
#
# Examples:
#   bash slurm/submit_clone_scan.sh           # full grid (all 1020 tasks)
#   bash slurm/submit_clone_scan.sh 0 339     # L=8  and L=16 tasks only
#   bash slurm/submit_clone_scan.sh 340 679   # L=32 tasks only
#   bash slurm/submit_clone_scan.sh 680 1019  # L=64 tasks only
# =============================================================================
set -euo pipefail

TASK_LO=${1:-0}
TASK_HI=${2:-1019}
OUTPUT_DIR=${3:-/scratch/${USER}/pps_qj/pps_clone_scan}

N_TASKS=$(( TASK_HI - TASK_LO + 1 ))

# ---- SLURM settings --------------------------------------------------------
# Rough wall-time estimates per task (single core):
#   L=8   (N_c=2000): ~1–2 min    |  L=8,16  batch (340 tasks): 04:00:00
#   L=16  (N_c=1000): ~3–6 min    |
#   L=32  (N_c=500) : ~15–30 min  |  L=32    batch (340 tasks): 12:00:00
#   L=64  (N_c=200) : ~45–90 min  |  L=64    batch (340 tasks): 24:00:00
# With 64 parallel cores, divide those estimates by 64.

JOB_NAME="pps_clone_${TASK_LO}_${TASK_HI}"
WALL_TIME="24:00:00"   # conservative; reduce for L=8,16 to get shorter queue wait
PARTITION="regular"    # use 'short' (4h max) for small-L batches to queue faster
N_CORES=64             # one full Habrok node; increase to 128 if your allocation allows

LOG_DIR="/scratch/${USER}/pps_qj/logs"

# ---- Job script (heredoc piped to sbatch) ----------------------------------
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
echo "Tasks: ${TASK_LO}..${TASK_HI}  (${N_TASKS} total)"
echo "Output: ${OUTPUT_DIR}"
echo "======================================================================"

# -- Environment -------------------------------------------------------------
module purge
source \${HOME}/venvs/pps_qj/bin/activate

cd \${HOME}/pps_qj

# -- Output and log directories ----------------------------------------------
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# -- Progress helper (logs to job .out file every 5 min) ---------------------
_progress() {
    while true; do
        N_DONE=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
        echo "[\$(date +%H:%M:%S)] \${N_DONE} tasks complete (of ${N_TASKS} submitted)"
        sleep 300
    done
}
_progress &
PROGRESS_PID=\$!

# -- Run all tasks in parallel, one per core ---------------------------------
# xargs -P uses a work-stealing queue: each worker immediately picks up the
# next task as it finishes, keeping all cores busy throughout.
seq ${TASK_LO} ${TASK_HI} | xargs -P \${SLURM_NTASKS} -I{} \
    python -m pps_qj.parallel.worker_clone_pps {} ${OUTPUT_DIR}

# -- Wrap up -----------------------------------------------------------------
kill \${PROGRESS_PID} 2>/dev/null || true

N_DONE=\$(ls ${OUTPUT_DIR}/clone_*.npz 2>/dev/null | wc -l)
N_FAIL=\$(ls ${OUTPUT_DIR}/summary_clone_*.json 2>/dev/null \
    | xargs grep -l '"status": "failed"' 2>/dev/null | wc -l)

echo "======================================================================"
echo "Finished: \$(date)"
echo "Completed: \${N_DONE} npz files written"
echo "Failed:    \${N_FAIL} tasks"
if [ \${N_FAIL} -gt 0 ]; then
    echo "Failed task summaries:"
    ls ${OUTPUT_DIR}/summary_clone_*.json \
        | xargs grep -l '"status": "failed"' \
        | xargs grep -h '"task_id"\|"error"'
fi
echo "======================================================================"
SLURM_SCRIPT

echo ""
echo "Submitted: ${JOB_NAME}"
echo "  Tasks:     ${TASK_LO}..${TASK_HI} (${N_TASKS} tasks on ${N_CORES} cores)"
echo "  Partition: ${PARTITION}  Wall time: ${WALL_TIME}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/${JOB_NAME}_<JOBID>.out"
echo ""
echo "Check results when done:"
echo "  python -c \""
echo "  import json, glob, numpy as np"
echo "  files = sorted(glob.glob('${OUTPUT_DIR}/summary_clone_*.json'))"
echo "  print('L   lam   zeta   S_mean   B_L_mean   B_L_err   theta    n_coll')"
echo "  for f in files:"
echo "      d = json.load(open(f))"
echo "      print(f\\\"{d['L']:3d}  {d['lam']:.2f}  {d['zeta']:.2f}  {d.get('S_mean',float('nan')):.4f}  {d.get('B_L_mean',float('nan')):.4f}  {d.get('B_L_err',float('nan')):.4f}  {d.get('theta_mean',float('nan')):+.4f}  {d.get('n_collapses','?')}\\\")"
echo "  \" | sort -k1,1n -k2,2n -k3,3n"
