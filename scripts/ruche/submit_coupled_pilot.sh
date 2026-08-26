#!/bin/bash
#SBATCH --job-name=pps_couple
#SBATCH --output=logs/couple_%A_%a.out
#SBATCH --error=logs/couple_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#
# =============================================================================
# PHASE 2 PILOT -- scripts/pilot_coupled_lambda.py
#
# Tests the one intervention aimed at the measured binding failure: LAMC's gate
# rejects 8-10 of 10 pairs at zeta = 0.40..0.70 on sign_change_multiplicity,
# and sixfold more data did not move it.  Coupling lambda across the scan makes
# D(lambda) smooth per realisation.  Pre-registered criterion and the FAIL
# branch are both written into the analysis script BEFORE the run.
#
# PARTITIONS (measured 2026-08-26): cpu_short 1:00:00, cpu_med 4:00:00,
#   cpu_prod 6:00:00, cpu_long 7 days.  Pick the shortest that fits.
#
# COST.  Scaling the Mac cost model (wall ~ L^3.15 at fixed T, x L for T = L):
#   L=32 ~7 s/real, L=64 ~74 s/real, L=96 ~394 s/real at N_c=128, dtau_mult=12.
#   11 lambda x 16 real x 2 modes = 352 runs per L
#   => ~0.7 + 7 + 39 = about 47 core-hours total.  Not the 20 I first quoted.
#
# RUN:
#   mkdir -p $WORKDIR/pps/couple
#   sbatch -p cpu_med --time=04:00:00 --array=0-9%10 \
#     --export=ALL,OUTDIR=$WORKDIR/pps/couple,ZETA=0.55,LS="32,64,96",NREAL=16 \
#     scripts/ruche/submit_coupled_pilot.sh
#
# CHEAPER FIRST PASS, drops L=96 and costs ~8 core-hours:
#   sbatch -p cpu_med --time=02:00:00 --array=0-3 \
#     --export=ALL,OUTDIR=$WORKDIR/pps/couple32,ZETA=0.55,LS="32,64",NREAL=16 \
#     scripts/ruche/submit_coupled_pilot.sh
#
# SCORE IT:
#   python scripts/analyse_coupled_pilot.py --dir $WORKDIR/pps/couple
# =============================================================================

set -euo pipefail
: "${OUTDIR:?OUTDIR is required}"
ZETA="${ZETA:-0.55}"
LS="${LS:-32,64,96}"
NREAL="${NREAL:-16}"
NC="${NC:-128}"
MODES="${MODES:-independent,coupled}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUTDIR"

module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORKDIR/envs/pps_qj"

echo "HOST=$(hostname)  COMMIT=$(git rev-parse HEAD)"

python scripts/pilot_coupled_lambda.py \
    --outdir "$OUTDIR" --zeta "$ZETA" --Ls "$LS" --modes "$MODES" \
    --nreal "$NREAL" --Nc "$NC" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" \
    --nshards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --nworkers "${SLURM_CPUS_PER_TASK:-1}"
