#!/bin/bash
#SBATCH --job-name=pps_ncl
#SBATCH --output=logs/ncl_%A_%a.out
#SBATCH --error=logs/ncl_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#
# =============================================================================
# N_c LADDER -- scripts/pilot_nc_ladder.py
#
# Separates BIAS from VARIANCE.  lambda_c_hat(N_c) = lambda_c + b(N_c) + eps.
# Buying realisations kills eps and does nothing to b.  ARCH's red team already
# measured a finite-N_c shift in the locator at z = 3.4, so b != 0 is not
# hypothetical.  Until b is bounded, more realisations may buy a more precise
# wrong answer.
#
# Scored by lambda_c, NOT by ESS or GESS (DEC-MASTER-METRIC-001: diagnostic
# only).  Genealogy healing while lambda_c sits still is a PASS.
#
# MEMORY.  N_c = 512 at L = 64 holds 512 covariance matrices of 128x128 float64
# plus workspace, and 40 workers share the node.  120G requested; if it OOMs,
# halve --cpus-per-task rather than dropping the top rung.
#
# COST.  Exactly linear in N_c (benchmark: wall ~ N_c^1.00 +- 0.02), so the
# 64/128/256/512 ladder is 7.5x one rung at 128.  Default grid ~63 core-hours.
#
# ALWAYS export first, then --export=ALL.  sbatch splits --export on COMMAS, so
# an inline LS="32,48,64" silently becomes LS=32 -- that cost us two wasted
# submissions on 2026-08-26.
#
# RUN:
#   unset NREAL ARMS CELLS ZETAS LS NC
#   mkdir -p $WORKDIR/pps/ncladder
#   export OUTDIR=$WORKDIR/pps/ncladder
#   export ZETA=0.55
#   export LS="32,48,64"
#   export NCS="64,128,256,512"
#   export NREAL=12
#   sbatch -p cpu_med --time=04:00:00 --array=0-15 --export=ALL \
#     scripts/ruche/submit_nc_ladder.sh
#
# CHEAPER FIRST PASS, three rungs and two L, ~15 core-hours:
#   export LS="32,48"; export NCS="64,128,256"
#   sbatch -p cpu_med --time=02:00:00 --array=0-7 --export=ALL \
#     scripts/ruche/submit_nc_ladder.sh
#
# VERIFY BEFORE WALKING AWAY:
#   grep SHARDING logs/ncl_*.out          # distinct shards, same total
#   grep "shard .* :" logs/ncl_*.out      # 180 total for the default grid
#
# ANALYSE:
#   python scripts/analyse_nc_ladder.py --dir $WORKDIR/pps/ncladder
# =============================================================================

set -euo pipefail
: "${OUTDIR:?OUTDIR is required}"
ZETA="${ZETA:-0.55}"
LS="${LS:-32,48,64}"
NCS="${NCS:-64,128,256,512}"
LAMS="${LAMS:-0.31,0.335,0.36,0.385,0.41}"
NREAL="${NREAL:-12}"

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
echo "SHARDING: shard $SHARD of $NSHARDS  (array id ${SLURM_ARRAY_TASK_ID:-none},"\
     "min ${SLURM_ARRAY_TASK_MIN:-none}, max ${SLURM_ARRAY_TASK_MAX:-none})"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUTDIR"

module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORKDIR/envs/pps_qj"

echo "HOST=$(hostname)  COMMIT=$(git rev-parse HEAD)"

python scripts/pilot_nc_ladder.py \
    --outdir "$OUTDIR" --zeta "$ZETA" --Ls "$LS" --Ncs "$NCS" \
    --lams "$LAMS" --nreal "$NREAL" \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --nworkers "${SLURM_CPUS_PER_TASK:-1}"
