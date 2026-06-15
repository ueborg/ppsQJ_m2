#!/bin/bash
# Cut B guided nu-campaign add-ons: higher size (L=160) + N_c ladder.
# Selected by the GRID env var (default guided_highL).
#
#   GRID=guided_highL  -> L=160, 15 zeta x 13 lambda = 195 tasks (N_c=250)
#   GRID=guided_ladder -> L=96,128,160 central-7 lambda, second N_c rung
#                         (96,128 -> 600 ; 160 -> 500), 315 tasks
#
# Purpose: a clean nu(zeta). highL extends the FSS lever arm to L=160; the
# ladder supplies the SECOND N_c rung so B_L at L=96/128/160 can be
# 1/N_c-extrapolated to B_inf (the prod grid gives N_c=300 at 96/128, highL
# gives N_c=250 at 160). Analysis keys by (L,lambda,zeta,N_c) and merges the
# prod/highL/ladder output dirs.
#
# SUBMISSION (size-binned; all ranges < MaxArraySize 1001):
#   # higher size L=160
#   GRID=guided_highL  sbatch --array=0-194%48   --time=08:00:00 --job-name=gB_highL  scripts/habrok/pps_scan/submit_clone_guided_nu.sh
#   # ladder, by L tier (ids: L96 0-104, L128 105-209, L160 210-314)
#   GRID=guided_ladder sbatch --array=0-104%48   --time=04:00:00 --job-name=gL_96   scripts/habrok/pps_scan/submit_clone_guided_nu.sh
#   GRID=guided_ladder sbatch --array=105-209%48 --time=08:00:00 --job-name=gL_128  scripts/habrok/pps_scan/submit_clone_guided_nu.sh
#   GRID=guided_ladder sbatch --array=210-314%32 --time=14:00:00 --job-name=gL_160  scripts/habrok/pps_scan/submit_clone_guided_nu.sh
# Recommended order if time is tight: prod -> gL_96 -> gL_128 -> gB_highL -> gL_160.
#
#SBATCH --job-name=gB_nu
#SBATCH --output=logs/gB_nu_%A_%a.out
#SBATCH --error=logs/gB_nu_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --array=0-194%48

GRID="${GRID:-guided_highL}"
REPO_DIR="${REPO_DIR:-$HOME/pps_qj}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/pps_qj}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/pps_qj/pps_clone_${GRID}}"

mkdir -p logs "$OUTPUT_DIR"
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2023.02-gfbf-2022b
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=5

export PPS_GUIDED=1
export PPS_DTAU_MULT=6
export PPS_RECORD_RENYI=1

cd "$REPO_DIR"
python -m pps_qj.parallel.worker_clone_pps \
    "$SLURM_ARRAY_TASK_ID" "$OUTPUT_DIR" --grid "$GRID"
