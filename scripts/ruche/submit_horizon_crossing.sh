#!/bin/bash
#SBATCH --job-name=pps_hcross
#SBATCH --output=logs/hcross_%A_%a.out
#SBATCH --error=logs/hcross_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=100G
#
# LOW-ZETA HORIZON-CROSSING PILOT -- manual submission only.
#
# Unlike submit_adaptive_pilot.sh, this is a genuine lambda grid, so it can
# measure lambda*(T=L) versus lambda*(T=2L), crossing multiplicity, and whether
# CMI_tavg50/75 help the actual boundary locator.
#
# Default zeta=0.20 grid is absolute and motivated only by the real-model
# L=16/24 diagnostic (~0.21 crossing); it assumes no sqrt(zeta) law.
#
# Array:
#   0: L=32, T=L
#   1: L=32, T=2L
#   2: L=48, T=L
#   3: L=48, T=2L
#
# MANUAL RUN:
#   cd ~/ppsQJ_m2 && git pull --ff-only origin main
#   unset OUTDIR ZETA LAMS MODES NREAL NC
#   export OUTDIR=$WORKDIR/pps/horizon_cross_z020
#   export ZETA=0.20
#   export LAMS='0.14,0.18,0.22,0.26,0.30'
#   export MODES='always,ess0.9,never'
#   export NREAL=12
#   export NC=128
#   mkdir -p "$OUTDIR"
#   sbatch -p cpu_med --time=04:00:00 --array=0-3 --export=ALL \
#       scripts/ruche/submit_horizon_crossing.sh
#
# ANALYSE:
#   python scripts/analyse_horizon_crossing.py \
#       --dir $WORKDIR/pps/horizon_cross_z020
#
# If the crossing touches an edge, broaden the ABSOLUTE bracket before any
# exponent/boundary inference.
#
set -euo pipefail
: "${OUTDIR:?OUTDIR is required}"

ZETA="${ZETA:-0.20}"
LAMS="${LAMS:-0.14,0.18,0.22,0.26,0.30}"
MODES="${MODES:-always,ess0.9,never}"
NREAL="${NREAL:-12}"
NC="${NC:-128}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

AID="${SLURM_ARRAY_TASK_ID:-0}"
case "$AID" in
  0) L=32; TMULT=1 ;;
  1) L=32; TMULT=2 ;;
  2) L=48; TMULT=1 ;;
  3) L=48; TMULT=2 ;;
  *) echo "invalid array id $AID"; exit 1 ;;
esac

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUTDIR"

module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORKDIR/envs/pps_qj"

echo "HOST=$(hostname) COMMIT=$(git rev-parse HEAD)"
echo "ARRAY=$AID L=$L TMULT=$TMULT ZETA=$ZETA LAMS=$LAMS MODES=$MODES NREAL=$NREAL NC=$NC"

python scripts/pilot_horizon_crossing.py \
    --outdir "$OUTDIR" --L "$L" --Tmult "$TMULT" \
    --zeta "$ZETA" --lams "$LAMS" --modes "$MODES" \
    --nreal "$NREAL" --Nc "$NC" \
    --nworkers "${SLURM_CPUS_PER_TASK:-1}"
