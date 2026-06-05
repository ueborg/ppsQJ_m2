#!/bin/bash
# =============================================================================
# submit_clone_ladder.sh  --  N_c-ladder campaign (the 2-day decisive run)
#
# Submits ONE N_c rung as a SLURM job array.  Each array element is a single
# 5-core cloning task (5 realisations in parallel); SLURM spreads the array
# across whatever nodes are free -> embarrassingly parallel, multi-node.
#
# Because a task's wall time = ONE realisation (~13h Nc250 / ~26h Nc500 /
# ~42h Nc800 at L=128), every rung fits inside a <48h job, and more seeds
# (blocks) cost more array elements (cores), NOT more wall time.
#
# Each rung writes to its OWN directory (the aggregator keys by (L,lam,zeta)
# and would merge rungs otherwise).  Pool the blocks afterwards with
# analysis/aggregate_ladder.py, then debias against existing clean L=32,64.
#
# Usage:
#   bash slurm/submit_clone_ladder.sh <RUNG> [CONCURRENCY] [OUTBASE]
#     RUNG         : 500 | 250 | 800   (run 500 first; 250 and 800 calibrate)
#     CONCURRENCY  : max array elements running at once (default 80).
#                    cores used = CONCURRENCY*5; nodes ~= CONCURRENCY*5/120.
#                    e.g. 80 -> ~3.3 nodes; 192 -> ~8 nodes; 288 -> ~12 nodes.
#     OUTBASE      : parent of the per-rung output dir
#                    (default /scratch/$USER/pps_qj)
#
# Recommended for <2 days on ~10-12 nodes:
#   bash slurm/submit_clone_ladder.sh 500 200    # ~26h wall, the bulk
#   bash slurm/submit_clone_ladder.sh 800  90    # ~42h wall, top rung
#   bash slurm/submit_clone_ladder.sh 250  90    # ~13h wall, bottom rung
# (submit all three at once; they queue independently and share the cluster.)
# =============================================================================
set -euo pipefail

RUNG=${1:?"give a rung: 500 | 250 | 800"}
CONC=${2:-80}
OUTBASE=${3:-/scratch/${USER}/pps_qj}

case "${RUNG}" in
  250) WALL="18:00:00" ;;
  500) WALL="36:00:00" ;;
  800) WALL="48:00:00" ;;
  *) echo "RUNG must be 250, 500, or 800 (got ${RUNG})"; exit 1 ;;
esac

OUTPUT_DIR="${OUTBASE}/pps_clone_ladder_nc${RUNG}"
LOG_DIR="${OUTBASE}/logs"
JOB_NAME="cllad_nc${RUNG}"
PARTITION="regular"
CPUS_PER_TASK=5

# Resolve this rung's contiguous task-id range from the grid.
read LO HI < <(cd "${HOME}/pps_qj" && python3 - "${RUNG}" <<'PY'
import sys
from pps_qj.parallel.grid_pps import clone_ladder_rung_ranges
lo, hi = clone_ladder_rung_ranges()[int(sys.argv[1])]
print(lo, hi)
PY
)
echo "Rung N_c=${RUNG}: array task ids ${LO}..${HI}  (wall ${WALL}, conc ${CONC})"
echo "Output dir: ${OUTPUT_DIR}"

sbatch <<SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --array=${LO}-${HI}%${CONC}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${WALL}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%A_%a.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rug.nl
set -euo pipefail

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=${CPUS_PER_TASK}
export PPS_RECORD_RENYI=1
export PPS_FORCE_RERUN=1
export PPS_BACKEND=\${PPS_BACKEND:-scalar}

mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

echo "[\$(date +%H:%M:%S)] array task \${SLURM_ARRAY_TASK_ID} (job \${SLURM_ARRAY_JOB_ID}) -> ${OUTPUT_DIR}"
python -m pps_qj.parallel.worker_clone_ladder_pps \${SLURM_ARRAY_TASK_ID} ${OUTPUT_DIR}
echo "[\$(date +%H:%M:%S)] array task \${SLURM_ARRAY_TASK_ID} done"
SLURM_SCRIPT

echo "Submitted ${JOB_NAME}: array ${LO}-${HI}%${CONC}, wall ${WALL}, out ${OUTPUT_DIR}"
