#!/bin/bash
# =============================================================================
# submit_validate_tcap.sh  —  T-cap (saturation-time) validation as a job
#
# For each (L, lam, zeta) point on a small grid, runs cloning at extended T
# with no burn-in and full S_history recording, then post-processes to find
# the saturation time t_sat.  Output recommends a tightened T cap if the
# production T is overly conservative.
#
# Cost note: extended-T runs are expensive at L=128.  This script targets
# L=32 and L=64 only.  L=128 saturation must be inferred from L=32, L=64
# trends (z-exponent fit) rather than measured directly.
#
# Usage:
#   bash slurm/submit_validate_tcap.sh
#
# After job completes:
#   cat /scratch/$USER/pps_qj/tcap_validation_*/aggregate.txt
# =============================================================================
set -euo pipefail

PARTITION="regular"
WALL_TIME="08:00:00"
N_PARALLEL=12
LOG_DIR="/scratch/${USER}/pps_qj/logs"
OUT_DIR="/scratch/${USER}/pps_qj/tcap_validation_$(date +%Y%m%d_%H%M)"

mkdir -p "${LOG_DIR}"

sbatch <<SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=pps_tcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${N_PARALLEL}
#SBATCH --time=${WALL_TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/tcap_%j.out
#SBATCH --error=${LOG_DIR}/tcap_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rug.nl

set -euo pipefail

echo "=================================================================="
echo "PPS T-cap (saturation) validation — Job \${SLURM_JOB_ID}"
echo "Started: \$(date)"
echo "Output:  ${OUT_DIR}"
echo "=================================================================="

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p ${OUT_DIR}

# Build (L, lam, zeta, T, seed, N_c, outdir) task list ------------------------
python <<'PY' > ${OUT_DIR}/tasks.txt
import sys
# Two grids: cheap L=32 broad coverage, and a small L=64 cross-check.
grid_L32 = {
    "L":     [32],
    "lam":   [0.30, 0.40, 0.50],
    "zeta":  [0.30, 0.50, 0.70],
    "T":     150.0,        # 2x existing L=32 cap (CONTEXT.md: T_prod=200 → use 1.5x for budget)
    "seeds": list(range(3)),
    "N_c":   200,
}
grid_L64 = {
    "L":     [64],
    "lam":   [0.40],
    "zeta":  [0.30, 0.50, 0.70],
    "T":     200.0,        # current L=64 cap is 150
    "seeds": list(range(2)),
    "N_c":   100,           # smaller N_c at L=64 to fit in walltime
}

OUT_DIR = "${OUT_DIR}"
total = 0
for grid in (grid_L32, grid_L64):
    for L in grid["L"]:
        for lam in grid["lam"]:
            for z in grid["zeta"]:
                for s in grid["seeds"]:
                    print(L, lam, z, grid["T"], s, grid["N_c"], OUT_DIR,
                          file=sys.stdout)
                    total += 1
print(f"# total tasks: {total}", file=sys.stderr)
PY

n_tasks=\$(wc -l < ${OUT_DIR}/tasks.txt)
echo "Total tasks: \${n_tasks}"
echo "Parallelism: ${N_PARALLEL}"
echo "Each task: one extended-T cloning realisation"

xargs -P ${N_PARALLEL} -L 1 \
    python -m pps_qj.tools.validate_tcap_worker < ${OUT_DIR}/tasks.txt

echo
echo "=================================================================="
echo "Aggregating saturation times"
echo "=================================================================="

python -m pps_qj.tools.aggregate_tcap ${OUT_DIR} | tee ${OUT_DIR}/aggregate.txt

echo
echo "=================================================================="
echo "Done: \$(date)"
echo "Results: ${OUT_DIR}/aggregate.txt"
echo "=================================================================="
SLURM_SCRIPT

echo "Submitted T-cap validation job."
echo "Monitor: squeue -u \${USER} -n pps_tcap"
echo "Output:  tail -f ${LOG_DIR}/tcap_<jobid>.out"
echo "Final:   cat ${OUT_DIR}/aggregate.txt"
