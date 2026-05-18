#!/bin/bash
# =============================================================================
# submit_renyi_targets_array.sh - job-array version of submit_renyi_targets.sh
#
# Same set of 30 strategic tasks, same env vars, same output directory layout
# as submit_renyi_targets.sh — but each task is its OWN SLURM array element,
# requesting only 5 cores instead of all 120 at once. This dramatically
# improves scheduling chances when no single 120-core node-slot is free in
# a medium-or-longer partition.
#
# Each array element runs ONE task (one (L, λ, ζ) point) and exits.
# SLURM schedules them independently — they can land on different nodes,
# different partitions, and start as soon as ANY 5-core slot opens up.
#
# Multi-partition spec (regular,parallel) lets SLURM pick whichever has
# space first. With WALL_TIME=10h, this routes to *medium variants;
# with WALL_TIME ≤ 8h, *short variants become eligible (where idle nodes
# currently exist per `sinfo`).
#
# Usage:
#   bash slurm/submit_renyi_targets_array.sh                             # defaults
#   bash slurm/submit_renyi_targets_array.sh /scratch/myoutput 10:00:00  # custom
#
# Throttle concurrent array elements with the SLURM array `%N` modifier
# (e.g., array=0-29%10 = up to 10 running at once). Default unthrottled.
#
# After completion: scripts/analyze_renyi_targets.py reads the output dir.
# =============================================================================
set -euo pipefail

OUTPUT_DIR=${1:-/scratch/${USER}/pps_qj/pps_clone_renyi}
WALL_TIME=${2:-10:00:00}
MAX_CONCURRENT=${3:-30}   # cap on simultaneously running array elements; 30 = all

JOB_NAME="renyi_arr"
LOG_DIR="/scratch/${USER}/pps_qj/logs"
CPUS_PER_TASK=5

# Same strategic task IDs as submit_renyi_targets.sh (6 (λ,ζ) × 5 L).
TASK_IDS=(744 789 796 809 849 866 984 1029 1036 1049 1089 1106 1224 1269 1276 1289 1329 1346 1464 1509 1516 1529 1569 1586 1704 1749 1756 1769 1809 1826)
N_TASKS=${#TASK_IDS[@]}
ARRAY_SPEC="0-$((N_TASKS - 1))%${MAX_CONCURRENT}"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Persist task IDs to a file keyed by array index. Array elements look up
# their own task ID by SLURM_ARRAY_TASK_ID + 1 (sed is 1-indexed).
TASKID_FILE="${LOG_DIR}/${JOB_NAME}_taskids.txt"
printf '%s\n' "${TASK_IDS[@]}" > "${TASKID_FILE}"

sbatch <<SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --array=${ARRAY_SPEC}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${WALL_TIME}
#SBATCH --partition=regular,parallel
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%A_%a.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rug.nl

set -euo pipefail

# Pick the task ID for this array element.
TASK_ID=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKID_FILE}")

echo "======================================================================"
echo "Array job \${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
echo "Task ID: \${TASK_ID}"
echo "Node: \$(hostname)  Partition: \${SLURM_JOB_PARTITION}  Cores: ${CPUS_PER_TASK}"
echo "Started: \$(date)"
echo "Output:  ${OUTPUT_DIR}"
echo "======================================================================"

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

# Pin BLAS / OpenMP to 1 thread per worker. With CPUS_PER_TASK=5 and 5
# realisations dispatched to multiprocessing.Pool inside the worker, that
# already saturates the allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PPS_N_WORKERS=${CPUS_PER_TASK}
export PPS_DTAU_MULT=\${PPS_DTAU_MULT:-2.0}
export PPS_RECORD_RENYI=1

python -m pps_qj.parallel.worker_clone_v2_pps "\${TASK_ID}" "${OUTPUT_DIR}"

echo "Done: \$(date)"
SLURM_SCRIPT

echo "Submitted job array with ${N_TASKS} elements (max ${MAX_CONCURRENT} concurrent)."
echo "Each element requests only ${CPUS_PER_TASK} cores — far easier to schedule."
echo ""
echo "Monitor:"
echo "  squeue -u \$USER -r              # show all array elements"
echo "  squeue -u \$USER --array         # alternate format"
echo "  sacct -j <JOBID> --format=JobID,State,Elapsed,Partition"
echo "  ls ${OUTPUT_DIR}/clone_*.npz | wc -l   # count completed"
