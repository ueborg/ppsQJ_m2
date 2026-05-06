#!/bin/bash
# =============================================================================
# submit_validate_dtau.sh  —  dτ-doubling validation as a background SLURM job
#
# For each (L, lam, zeta) point on a small grid, runs cloning at multiple
# dtau_mult values × multiple seeds and writes one JSON per realisation.
# Aggregator at the end produces a comparison table identifying the largest
# safe dτ multiplier per configuration.
#
# Usage:
#   bash slurm/submit_validate_dtau.sh
#
# After job completes:
#   cat /scratch/$USER/pps_qj/dtau_validation_*/aggregate.txt
# =============================================================================
set -euo pipefail

PARTITION="regular"
WALL_TIME="06:00:00"
N_PARALLEL=20
LOG_DIR="/scratch/${USER}/pps_qj/logs"
OUT_DIR="/scratch/${USER}/pps_qj/dtau_validation_$(date +%Y%m%d_%H%M)"

mkdir -p "${LOG_DIR}"

sbatch <<SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=pps_dtau
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${N_PARALLEL}
#SBATCH --time=${WALL_TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/dtau_%j.out
#SBATCH --error=${LOG_DIR}/dtau_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rug.nl

set -euo pipefail

echo "=================================================================="
echo "PPS dτ-doubling validation — Job \${SLURM_JOB_ID}"
echo "Started: \$(date)"
echo "Output:  ${OUT_DIR}"
echo "=================================================================="

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

# Single-thread BLAS to prevent oversubscription with xargs -P
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p ${OUT_DIR}

# Build the (L, lam, zeta, mult, seed, T, N_c, outdir) task list ----
python <<'PY' > ${OUT_DIR}/tasks.txt
import sys

# Grid covers area-law / near-critical / volume-law regimes at the ζ values
# most relevant to the MIPT.  L=32 is the primary test (cheap, comprehensive);
# L=64 is a smaller cross-check at one near-critical point.
grid_L32 = {
    "L":     [32],
    "lam":   [0.30, 0.40, 0.50],
    "zeta":  [0.30, 0.50, 0.70],
    "mult":  [1.0, 1.5, 2.0, 3.0],
    "seeds": list(range(5)),
    "T":     32.0,
    "N_c":   200,
}
grid_L64 = {
    "L":     [64],
    "lam":   [0.40],          # near-critical only at L=64
    "zeta":  [0.50],
    "mult":  [1.0, 1.5, 2.0],
    "seeds": list(range(5)),
    "T":     32.0,
    "N_c":   200,
}

OUT_DIR = "${OUT_DIR}"
total = 0
for grid in (grid_L32, grid_L64):
    for L in grid["L"]:
        for lam in grid["lam"]:
            for z in grid["zeta"]:
                for m in grid["mult"]:
                    for s in grid["seeds"]:
                        print(L, lam, z, m, s, grid["T"], grid["N_c"], OUT_DIR,
                              file=sys.stdout)
                        total += 1
print(f"# total tasks: {total}", file=sys.stderr)
PY

n_tasks=\$(wc -l < ${OUT_DIR}/tasks.txt)
echo "Total tasks: \${n_tasks}"
echo "Parallelism: ${N_PARALLEL}"
echo "Each task: one full cloning realisation"

# Run via xargs.  -L 1 = one input line per command invocation; that line's
# whitespace-separated fields become the worker's positional args.
xargs -P ${N_PARALLEL} -L 1 \
    python -m pps_qj.tools.validate_dtau_worker < ${OUT_DIR}/tasks.txt

echo
echo "=================================================================="
echo "Aggregating results"
echo "=================================================================="

python -m pps_qj.tools.aggregate_dtau ${OUT_DIR} | tee ${OUT_DIR}/aggregate.txt

echo
echo "=================================================================="
echo "Done: \$(date)"
echo "Results: ${OUT_DIR}/aggregate.txt"
echo "=================================================================="
SLURM_SCRIPT

echo "Submitted dτ validation job."
echo "Monitor: squeue -u \${USER} -n pps_dtau"
echo "Output:  tail -f ${LOG_DIR}/dtau_<jobid>.out"
echo "Final:   cat ${OUT_DIR}/aggregate.txt"
