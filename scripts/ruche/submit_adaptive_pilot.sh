#!/bin/bash
#SBATCH --job-name=pps_adapt
#SBATCH --output=logs/adapt_%A_%a.out
#SBATCH --error=logs/adapt_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#
# =============================================================================
# ADAPTIVE-RESAMPLING PILOT -- scripts/exp_adaptive_cloning.py
#
# EXPERIMENTAL FORK, not production.  The fork's always-mode is gated
# BIT-IDENTICAL against pps_qj/cloning.py at job start (|dtheta| = 0, 
# max|dcov| = 0 on a matched seed); the job aborts if the gate fails, so the
# ess/never modes differ from production by exactly the trigger and the
# weighted readout, nothing else.
#
# Four arms per cell, all with weighted readout:
#   always : production schedule (resample every window)
#   ess0.9 : resample when accumulated ESS < 0.9 N_c
#   ess0.5 : resample when accumulated ESS < 0.5 N_c
#   never  : zero interaction; N_c independent guided trajectories combined by
#            self-normalised importance weights.  Unbiased reference by
#            construction; its validity is its own final ESS, which is recorded.
#
# THE TWO QUESTIONS THIS ANSWERS AT SCALE (Mac L=20 numbers in brackets):
#   1. Does adaptive resampling reduce Var[CMI] at matched cost at low zeta,
#      and is it bias-free against the never reference?   [~1.9x at zeta=0.1,
#      tau=0.5, p~0.09 -- indicative only, NOT established at L=20]
#   2. How does never-mode's final ESS fraction scale with L at zeta <= 0.2?
#      [0.29 at L=20, T=20, N_c=48.]  If it stays workable at L=48, the low-
#      zeta campaign can run INTERACTION-FREE: embarrassingly parallel, no
#      genealogy at all.
#
# One (L, zeta) block per array task.  Mapping:
#   id 0..3 -> L=32, zeta = 0.05, 0.10, 0.20, 0.55
#   id 4..6 -> L=48, zeta = 0.05, 0.10, 0.20      (no L=48 mid-zeta: costly and
#                                                  the Mac run says low zeta is
#                                                  where the payoff is)
# Rough cost per task at NREAL=24, N_c=128, 3 lambda, 4 arms:
#   L=32 ~1-2 h single-core; L=48 ~3-4 h.  Fits cpu_med.
#
# ALWAYS export first, then --export=ALL (sbatch splits --export on commas):
#   unset NREAL ARMS CELLS ZETAS LS NC LAMS
#   mkdir -p $WORKDIR/pps/adaptive
#   export OUTDIR=$WORKDIR/pps/adaptive
#   export NREAL=24
#   sbatch -p cpu_med --time=04:00:00 --array=0-6 --export=ALL \
#     scripts/ruche/submit_adaptive_pilot.sh
#
# HORIZON ARM (run as a SECOND submission after the first): the Mac test on the
# real model at zeta=0.20, L=16 vs 24, moved the tail-averaged crossing
# 0.2254 -> 0.2149 -> 0.2094 for T = L, 2L, 4L (decelerating), and cut the sign
# changes 2 -> 1 -> 1.  T=L is NOT equilibrated at low zeta.  Every record
# already carries CMI (final snapshot), CMI_tavg50 and CMI_tavg75, so the same
# JSONs answer readout AND horizon questions:
#   export TMULT=2; export OUTDIR=$WORKDIR/pps/adaptive   # same dir, new files
#   sbatch -p cpu_med --time=04:00:00 --array=0-2,4-6 --export=ALL \
#     scripts/ruche/submit_adaptive_pilot.sh              # low-zeta tasks only
#
# ANALYSE: the per-task JSONs land in $OUTDIR; compare arm means (bias vs
# never where its essF > 0.15), SEM ratios vs always, n_events, gess_root vs
# gess_recent, and essF(never) vs L.  Acceptance for promoting the algorithm is
# NOT GESS (diagnostic_only per DEC-MASTER-METRIC-001): it is bias-free vs the
# reference AND variance reduction at matched cost, and any production claim
# then still needs t_wall * var(lambda_c).
# =============================================================================

set -euo pipefail
: "${OUTDIR:?OUTDIR is required}"
NREAL="${NREAL:-24}"
NC="${NC:-128}"
TMULT="${TMULT:-1}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

AID="${SLURM_ARRAY_TASK_ID:-0}"
case "$AID" in
  0) L=32; CELLS="0.05:0.10" ;;
  1) L=32; CELLS="0.10:0.14" ;;
  2) L=32; CELLS="0.20:0.21" ;;
  3) L=32; CELLS="0.55:0.35" ;;
  4) L=48; CELLS="0.05:0.10" ;;
  5) L=48; CELLS="0.10:0.14" ;;
  6) L=48; CELLS="0.20:0.21" ;;
  *) echo "no cell for array id $AID"; exit 1 ;;
esac
echo "ARRAY $AID -> L=$L CELLS=$CELLS NREAL=$NREAL NC=$NC TMULT=$TMULT"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUTDIR"

module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORKDIR/envs/pps_qj"

echo "HOST=$(hostname)  COMMIT=$(git rev-parse HEAD)"

python scripts/exp_adaptive_cloning.py --mode study \
    --L "$L" --Nc "$NC" --nreal "$NREAL" --cells "$CELLS" --Tmult "$TMULT" \
    --out "$OUTDIR/adaptive_L${L}_T${TMULT}_task${AID}.json"
