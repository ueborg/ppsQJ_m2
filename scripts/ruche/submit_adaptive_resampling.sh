#!/bin/bash
#SBATCH --job-name=pps_adapt
#SBATCH --output=logs/adapt_%A_%a.out
#SBATCH --error=logs/adapt_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#
# ADAPTIVE-RESAMPLING PILOT. MANUAL SUBMISSION ONLY.
#
# Tests the concrete discontinuity in the current sampler: zeta=1 skips
# resampling, while every zeta<1 resamples every window even when ESS/Nc~0.97.
# Adaptive arms carry cumulative normalized weights exactly and resample only
# when ESS/Nc crosses a threshold. GESS is diagnostic only; promote an arm only
# if final-observable variance/crossing conditioning improves without a mean or
# locator shift.
#
# Cheap regime diagnostic (mid + low zeta):
#   unset OUTDIR GRIDS LS ARMS NREAL NC NSHARDS
#   export OUTDIR=$WORKDIR/pps/adaptive_diag
#   export GRIDS='0.55:0.35|0.30:0.24|0.10:0.12|0.05:0.08'
#   export LS='32,64'
#   export ARMS='baseline,adaptive97,adaptive90,adaptive75'
#   export NREAL=12; export NC=128
#   mkdir -p "$OUTDIR"
#   sbatch -p cpu_med --time=04:00:00 --array=0-7 --export=ALL \
#     scripts/ruche/submit_adaptive_resampling.sh
#
# Mid-zeta crossing-grade follow-up, only after the diagnostic:
#   export OUTDIR=$WORKDIR/pps/adaptive_mid_cross
#   export GRIDS='0.55:0.31,0.335,0.36,0.385,0.41'
#   export LS='32,48,64'
#   export ARMS='baseline,adaptive97'
#   export NREAL=24; export NC=128
#   sbatch -p cpu_med --time=04:00:00 --array=0-11 --export=ALL \
#     scripts/ruche/submit_adaptive_resampling.sh
#
# Low-zeta grids must be absolute brackets established independently from the
# relevant data. Do NOT center them on sqrt(zeta).

set -euo pipefail
: "${OUTDIR:?OUTDIR is required}"
GRIDS="${GRIDS:-0.55:0.35|0.30:0.24|0.10:0.12|0.05:0.08}"
LS="${LS:-32,64}"
ARMS="${ARMS:-baseline,adaptive97,adaptive90,adaptive75}"
NREAL="${NREAL:-12}"
NC="${NC:-128}"
TMULT="${TMULT:-1.0}"
DTAU_MULT="${DTAU_MULT:-12.0}"
SOLVER="${SOLVER:-newton}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

if [ -z "${NSHARDS:-}" ]; then
  if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && [ -n "${SLURM_ARRAY_TASK_MIN:-}" ]; then
    NSHARDS=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
  else
    NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
  fi
fi
SHARD=$(( ${SLURM_ARRAY_TASK_ID:-0} - ${SLURM_ARRAY_TASK_MIN:-0} ))
echo "SHARDING: shard $SHARD of $NSHARDS"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUTDIR"
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORKDIR/envs/pps_qj"
echo "HOST=$(hostname) COMMIT=$(git rev-parse HEAD)"

python scripts/pilot_adaptive_resampling.py \
  --outdir "$OUTDIR" --grids "$GRIDS" --Ls "$LS" --arms "$ARMS" \
  --nreal "$NREAL" --Nc "$NC" --Tmult "$TMULT" \
  --dtau-mult "$DTAU_MULT" --solver "$SOLVER" \
  --shard "$SHARD" --nshards "$NSHARDS" \
  --nworkers "${SLURM_CPUS_PER_TASK:-1}"
