#!/bin/bash
#SBATCH --job-name=pps_areaphase
#SBATCH --output=areaphase_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --array=0-29
# GATE 2: area-phase xi(zeta,lambda) -> phi=1/2 vs phi=1.
# Default grid = 2 L x 5 zeta x 3 offsets = 30 tasks (array 0-29).
# Each task runs N_REAL=5 cloning realisations on 5 cores (cpus-per-task=5).
# Cost ~ that of one clone task at (L,N_c,T): minutes-to-hours per task at L<=96.
#
# CONVENTION: set --partition / --account for your Habrok allocation.
# Calibration first (one task, small N_c) before the full array:
#   PPS_L_LIST=64 PPS_ZETA_LIST=0.30 PPS_LAM_OFFSETS=0.10 PPS_NC=80 \
#     python -m pps_qj.parallel.worker_areaphase_pps 0 /scratch/$USER/pps_qj/pps_areaphase_cal

set -euo pipefail
source ~/venvs/pps_qj/bin/activate
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PPS_L_LIST="${PPS_L_LIST:-64,96}"
export PPS_ZETA_LIST="${PPS_ZETA_LIST:-0.10,0.20,0.30,0.50,0.80}"
export PPS_LAM_OFFSETS="${PPS_LAM_OFFSETS:-0.06,0.10,0.14}"
export PPS_NC="${PPS_NC:-250}"
export PPS_T_MULT="${PPS_T_MULT:-3.0}"
export PPS_N_WORKERS="${SLURM_CPUS_PER_TASK:-5}"
export PPS_SEED0="${PPS_SEED0:-20260610}"

OUTDIR="${PPS_OUTDIR:-/scratch/$USER/pps_qj/pps_areaphase}"
mkdir -p "$OUTDIR"

python -m pps_qj.parallel.worker_areaphase_pps "${SLURM_ARRAY_TASK_ID:-0}" "$OUTDIR"

# N_c-bias check (run ONE rung at 2x N_c before banking phi):
#   PPS_NC=500 sbatch --array=0-4 submit_areaphase.sh   # then compare xi vs N_c=250
