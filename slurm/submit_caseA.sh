#!/bin/bash
# =============================================================================
# submit_caseA.sh -- Case A (two competing measurements, H=0) production array.
#
# Tiered SLURM array. Each tier is a contiguous task-id range from
# pps_qj.parallel.grid_caseA.caseA_tier_ranges(). Each array element is one
# 5-core task (N_REAL=5 realisations run in parallel via PPS_N_WORKERS), so a
# task's wall time is ONE realisation. All tiers share ONE output dir: task
# ids are globally unique (0..237), so filenames never collide.
#
# Tiers:
#   zeta1     153-237 (85)  Born anchor, bias-free, CHEAP. Run FIRST; needs no
#                           calibration. L in {16,32,64,128,160}.
#   lt1_L32     0-50  (51)  zeta<1 cloning, L=32.
#   lt1_L64    51-101 (51)  zeta<1 cloning, L=64.
#   lt1_L128  102-152 (51)  zeta<1 cloning, L=128 (the expensive tier).
#
# Usage:
#   bash slurm/submit_caseA.sh <TIER> [CONCURRENCY] [OUTBASE]
#     TIER        : zeta1 | lt1_L32 | lt1_L64 | lt1_L128
#     CONCURRENCY : max array elements running at once (default 80).
#                   cores used = CONCURRENCY*5; nodes ~= cores/120.
#     OUTBASE     : parent of the output dir (default /scratch/$USER/pps_qj).
#
# *** WALL TIMES BELOW ARE PROVISIONAL. *** Before committing the cluster to a
# full zeta<1 tier, run ONE task of that tier on an interactive node and read
# the wall_time it writes to summary_caseA_<id>.json, then adjust WALL. Cost
# scales ~ N_c * T * L^4 (with L-cache effects pushing it steeper at L>=64).
#
# Calibration / trust gates (do BEFORE the zeta<1 tiers; see run-book). Both
# use env overrides into a SEPARATE output dir so they never touch production:
#   T saturation : for T in 80 120 160 200; do PPS_T_OVERRIDE=$T PPS_FORCE_RERUN=1 \
#                  python -m pps_qj.parallel.worker_caseA <L128_z0.10_lam0.5 id> \
#                  /scratch/$USER/pps_qj/pps_caseA_calib; done   # compare B_L(T)
#   N_c 2-rung   : PPS_NC_OVERRIDE=600 vs default 300 at the L=128 z=0.30
#                  lam=0.50 task, separate dir; compare B_L_mean within error.
# =============================================================================
set -euo pipefail

TIER=${1:?"give a tier: zeta1 | lt1_L32 | lt1_L64 | lt1_L128"}
CONC=${2:-80}
OUTBASE=${3:-/scratch/${USER}/pps_qj}

case "${TIER}" in
  zeta1)    WALL="24:00:00" ;;
  lt1_L32)  WALL="24:00:00" ;;
  lt1_L64)  WALL="36:00:00" ;;
  lt1_L128) WALL="48:00:00" ;;
  *) echo "TIER must be zeta1 | lt1_L32 | lt1_L64 | lt1_L128 (got ${TIER})"; exit 1 ;;
esac

OUTPUT_DIR="${OUTBASE}/pps_caseA"
LOG_DIR="${OUTBASE}/logs"
JOB_NAME="caseA_${TIER}"
PARTITION="regular"
CPUS_PER_TASK=5

read LO HI < <(cd "${HOME}/pps_qj" && python3 - "${TIER}" <<'PY'
import sys
from pps_qj.parallel.grid_caseA import caseA_tier_ranges
lo, hi = caseA_tier_ranges()[sys.argv[1]]
print(lo, hi)
PY
)
echo "Tier ${TIER}: array task ids ${LO}..${HI}  (wall ${WALL}, conc ${CONC})"
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

mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

echo "[\$(date +%H:%M:%S)] caseA ${TIER} task \${SLURM_ARRAY_TASK_ID} (job \${SLURM_ARRAY_JOB_ID}) -> ${OUTPUT_DIR}"
python -m pps_qj.parallel.worker_caseA \${SLURM_ARRAY_TASK_ID} ${OUTPUT_DIR}
echo "[\$(date +%H:%M:%S)] caseA ${TIER} task \${SLURM_ARRAY_TASK_ID} done"
SLURM_SCRIPT

echo "Submitted ${JOB_NAME}: array ${LO}-${HI}%${CONC}, wall ${WALL}, out ${OUTPUT_DIR}"
