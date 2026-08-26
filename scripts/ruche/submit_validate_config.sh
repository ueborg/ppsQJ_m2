#!/bin/bash
#SBATCH --job-name=pps_valid
#SBATCH --output=logs/valid_%A_%a.out
#SBATCH --error=logs/valid_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#
# =============================================================================
# Ruche submit for scripts/validate_production_config.py
#
# Certifies the configuration the Cut B production campaign ACTUALLY RAN
# (PPS_DTAU_MULT=12.0, SOLVER=newton) against the range that was ever
# validated (mult 1-8 at L=32 only, brentq).  TASK-2026-08-10-SAMPLER named
# this as the blocker on every efficiency number in the project.
#
# Paired seeds are automatic: the seed depends only on (L, lam, zeta, real).
#
# NOTHING IS SUBMITTED BY ANY AGENT.  You run these commands.
#
# ENV (override with --export=ALL,VAR=...):
#   OUTDIR   output dir on GPFS (REQUIRED)
#   ARMS     comma list from A_production,D_solver_only,C_recommended,
#            B_certified,E_stride1,F_eigh
#   CELLS    comma list L:zeta, e.g. "32:0.2,32:0.9"   (empty = all four)
#   NREAL    paired seeds per arm per cell (SAMPLER specifies 40)
#   NC       clone population (production = 128)
#
# -----------------------------------------------------------------------------
# TIER 1 - cheap, L=32 only, full six-arm factorial.  Run this first.
#   Rough cost: 6 arms x 2 cells x 40 real, L=32.  Arm B (dtau=2) is 6x the
#   steps of arm A, so it dominates.  Order a few tens of core-hours.
#
#   sbatch -p cpu_med --time=02:00:00 --array=0-9 \
#     --export=ALL,OUTDIR=$WORKDIR/pps/validate,\
# ARMS="A_production,D_solver_only,C_recommended,B_certified,E_stride1,F_eigh",\
# CELLS="32:0.2,32:0.9",NREAL=40,NC=128 \
#     scripts/ruche/submit_validate_config.sh
#
# TIER 2 - the one that matters, L=96, four arms.  Only launch after Tier 1
#   completes and you have looked at it.  L=96 at dtau_mult=2 is ~6x the
#   production per-realisation cost (~2730 s at mult 12), so arm B is the
#   budget.  Order a few hundred core-hours.
#
#   sbatch -p cpu_med --time=12:00:00 --array=0-19%10 \
#     --export=ALL,OUTDIR=$WORKDIR/pps/validate,\
# ARMS="A_production,D_solver_only,C_recommended,B_certified",\
# CELLS="96:0.2,96:0.9",NREAL=40,NC=128 \
#     scripts/ruche/submit_validate_config.sh
#
# DRY RUN FIRST, on the login node, costs nothing:
#   python scripts/validate_production_config.py --outdir /tmp/v --dry-run \
#     --arms A_production,B_certified --cells 32:0.2 --nreal 4
# =============================================================================
#
# PARTITIONS.  cpu_short rejected --time=02:00:00 on 2026-08-26
# ('Requested time limit is invalid'), so its cap is below that.  cpu_med is
# known-good (the July campaign used it at 04:00:00).  Check limits with
#   sinfo -o "%20P %10l %10L %6D %C"
# and prefer the shortest partition that actually accepts your --time.
# =============================================================================

set -euo pipefail

: "${OUTDIR:?OUTDIR is required}"
ARMS="${ARMS:-A_production,D_solver_only,C_recommended,B_certified}"
CELLS="${CELLS:-}"
NREAL="${NREAL:-40}"
NC="${NC:-128}"

NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NWORKERS="${SLURM_CPUS_PER_TASK:-1}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUTDIR"

module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORKDIR/envs/pps_qj"

echo "HOST=$(hostname)  COMMIT=$(git rev-parse HEAD)  DIRTY=$(git status --porcelain | wc -l)"
python --version

python scripts/validate_production_config.py \
    --outdir "$OUTDIR" \
    --arms "$ARMS" \
    ${CELLS:+--cells "$CELLS"} \
    --nreal "$NREAL" \
    --Nc "$NC" \
    --shard "$SHARD" \
    --nshards "$NSHARDS" \
    --nworkers "$NWORKERS"
