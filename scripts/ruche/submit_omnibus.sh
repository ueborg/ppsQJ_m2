#!/bin/bash
#SBATCH --job-name=pps_omni
#SBATCH --output=logs/omni_%A_%a.out
#SBATCH --error=logs/omni_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#
# =============================================================================
# OMNIBUS OBSERVABLE COMPARISON -- scripts/omnibus_observables.py
#
# Seven locators from the SAME trajectories and the same final covariances, so
# the only variable is the observable.  Motivated by a measured fact, not a
# hunch: the global collapse puts nu_eff at 3-5 across mid-zeta, so L^(1/nu)
# varies by 1.20 across L = 64..128 and only ~1.36 across the proposed 32..96.
# The L ladder alone cannot resolve this.  The observable has to.
#
# Ground truth on both ends: zeta = 1.00 and 0.30 survived the L-scramble
# control (ratios 3.3 and 5.2, lambda_c = 0.4364 and 0.2326); zeta = 0.55 did
# not (2.4).  An observable that reproduces both anchors AND resolves 0.55 wins.
#
# lambda windows are ABSOLUTE, not multiples of 0.5*sqrt(zeta).
#
# PARTITIONS (measured 2026-08-26): cpu_short 1:00:00, cpu_med 4:00:00,
#   cpu_prod 6:00:00, cpu_long 7-00:00:00.
#
# COST.  3 zeta x 11 lambda x 4 L x 12 real = 1584 runs.  From the Mac cost
# model at N_c=128, T=L: ~7 s (L=32), ~38 s (48), ~74 s (64), ~394 s (96).
#   => about 60 core-hours.  The S(l) profile adds ~1.5 percent.
#
# CHEAP FIRST PASS -- drop L=96, one zeta, ~3 core-hours.  Do this first:
#   sbatch -p cpu_med --time=02:00:00 --array=0-3 \
#     --export=ALL,OUTDIR=$WORKDIR/pps/omni_pilot,ZETAS=1.00,LS="32,48,64",NREAL=12 \
#     scripts/ruche/submit_omnibus.sh
#   -> must reproduce lambda_c = 0.4364 on at least the incumbent CMI, else the
#      harness is wrong and the full run is wasted.
#
# FULL RUN:
#   sbatch -p cpu_med --time=04:00:00 --array=0-19%20 \
#     --export=ALL,OUTDIR=$WORKDIR/pps/omni,ZETAS="0.30,0.55,1.00",\
# LS="32,48,64,96",NREAL=12 \
#     scripts/ruche/submit_omnibus.sh
#
# DRY RUN, costs nothing:
#   python scripts/omnibus_observables.py --outdir /tmp/o --dry-run
# =============================================================================

set -euo pipefail
: "${OUTDIR:?OUTDIR is required}"
ZETAS="${ZETAS:-0.30,0.55,1.00}"
LS="${LS:-32,48,64,96}"
NLAM="${NLAM:-11}"
NREAL="${NREAL:-12}"
NC="${NC:-128}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Ruche's Slurm does not set SLURM_ARRAY_TASK_COUNT.  Deriving NSHARDS from it
# silently gave every array task NSHARDS=1, so all ten ran identical work from
# index 0 and only the first 1/10 of the grid was ever produced (observed
# 2026-08-26: 40 of 480 records, all arm A_production).  Derive from the array
# bounds instead, and allow an explicit override.
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

python scripts/omnibus_observables.py \
    --outdir "$OUTDIR" --zetas "$ZETAS" --Ls "$LS" \
    --nlam "$NLAM" --nreal "$NREAL" --Nc "$NC" \
    --shard "$SHARD" \
    --nshards "$NSHARDS" \
    --nworkers "${SLURM_CPUS_PER_TASK:-1}"
