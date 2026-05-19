#!/bin/bash
# =============================================================================
# submit_fss_test.sh  —  Targeted L=192, 256 runs for the 1/L² FSS test
#
# PURPOSE:
#   Test whether the apparent shift in lambda_c(zeta) at finite L scales
#   as 1/L², as predicted by the chiral-vertex bosonization analysis:
#
#     lambda_c(zeta, L) - lambda_c(zeta=1, L) ~ C(zeta) / L²
#
#   zeta=0.30, L=128: shift = -0.127. Predicted at L=256: -0.032 (4x smaller).
#
#   See theory/qj_marginal_chiral_correction.md §5 for the full prediction.
#
# WHY NOT THE FST GRID?
#   submit_clone_v2_fst.sh has zeta in {0.05, 0.10, 0.14, 0.18, 0.50, 1.00}
#   — zeta=0.30 is entirely absent (FST was designed for the now-retracted
#   zeta~0.143 separatrix). This script fills the gap.
#
# GRID (18 tasks):
#   zeta=0.30, L={192,256}, lam={0.20, 0.25, 0.30}   — bracket lam_c~0.237
#   zeta=0.50, L={192,256}, lam={0.30, 0.35, 0.40}   — bracket lam_c~0.334
#   zeta=1.00, L={192,256}, lam={0.30, 0.35, 0.40}   — bracket lam_c~0.364
#
# Worker: pps_qj.parallel.worker_fss_direct (L, lam, zeta passed directly)
# Output files: fss_L***_lam*_zeta*.npz  (no collision with main/FST grids)
#
# WALL TIME:
#   L=192 worst case ~22h, L=256 worst case ~31h. Request 48h to be safe.
#   Job array: each element is 5 cores, independent — much easier to schedule
#   than a single 120-core node request.
#
# Usage:
#   bash slurm/submit_fss_test.sh                                # defaults
#   bash slurm/submit_fss_test.sh /scratch/myoutput 48:00:00    # custom
# =============================================================================
set -euo pipefail

OUTPUT_DIR=${1:-/scratch/${USER}/pps_qj/pps_fss_test}
WALL_TIME=${2:-48:00:00}

JOB_NAME="fss_test"
LOG_DIR="/scratch/${USER}/pps_qj/logs"
CPUS_PER_TASK=5
MAX_CONCURRENT=18

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Generate task list: one line per task "L lam zeta"
# ---------------------------------------------------------------------------
TASK_FILE="${LOG_DIR}/${JOB_NAME}_tasks.txt"
python3 - << 'PYEOF' > "${TASK_FILE}"
import itertools
specs = [
    ([192, 256], [0.20, 0.25, 0.30], [0.30]),   # bracket lam_c(zeta=0.30)=0.237
    ([192, 256], [0.30, 0.35, 0.40], [0.50]),   # bracket lam_c(zeta=0.50)=0.334
    ([192, 256], [0.30, 0.35, 0.40], [1.00]),   # bracket lam_c(zeta=1.00)=0.364
]
for Ls, lams, zetas in specs:
    for L, lam, zeta in itertools.product(Ls, lams, zetas):
        print(f"{L} {lam:.4f} {zeta:.4f}")
PYEOF

N_TASKS=$(wc -l < "${TASK_FILE}" | tr -d ' ')
echo "Generated ${N_TASKS} tasks:"
cat "${TASK_FILE}"
echo ""

sbatch << SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --array=0-$((N_TASKS - 1))%${MAX_CONCURRENT}
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

# Read this element's L lam zeta from the task file (1-indexed sed)
read -r L LAM ZETA <<< \$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${TASK_FILE}")

echo "======================================================================"
echo "Array \${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}: L=\${L} lam=\${LAM} zeta=\${ZETA}"
echo "Node: \$(hostname)  Partition: \${SLURM_JOB_PARTITION}"
echo "Started: \$(date)"
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

python -m pps_qj.parallel.worker_fss_direct \${L} \${LAM} \${ZETA} ${OUTPUT_DIR}

echo "Done: \$(date)"
SLURM_SCRIPT

echo ""
echo "Submitted ${N_TASKS} array elements (max ${MAX_CONCURRENT} concurrent, 5 cores each)."
echo "Output dir: ${OUTPUT_DIR}"
echo "Monitor:    squeue -u \$USER -r"
echo "Progress:   ls ${OUTPUT_DIR}/fss_*.npz | wc -l"
