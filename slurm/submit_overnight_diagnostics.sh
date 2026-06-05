#!/bin/bash
# =============================================================================
# submit_overnight_diagnostics.sh  --  run the heavy diagnostics as one job
#
# Runs three diagnostics back-to-back on a dedicated full node, each using the
# whole node via ProcessPoolExecutor. Designed to finish overnight (<~9h) and
# leave JSON summaries ready in the morning:
#
#   1. t_saturation (L=32,64,128)  -> is the production T overkill? (T-trim factor)
#   2. d5 clones-vs-seeds, L=96     -> N_c knee + seeds-needed at L=96
#   3. d5 clones-vs-seeds, L=128    -> N_c knee + seeds-needed at L=128 (decisive)
#
# Failures are non-fatal: if one diagnostic dies, the others still run.
#
# Submit:  sbatch slurm/submit_overnight_diagnostics.sh
# =============================================================================
#SBATCH --job-name=overnight_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --partition=regular
#SBATCH --output=/scratch/%u/pps_qj/logs/overnight_diag_%j.out
#SBATCH --error=/scratch/%u/pps_qj/logs/overnight_diag_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=%u@rug.nl
set -uo pipefail   # NOT -e: one diagnostic failing must not abort the rest

module purge
source ${HOME}/venvs/pps_qj/bin/activate
cd ${HOME}/pps_qj

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPS_BACKEND=scalar

WORKERS=${WORKERS:-120}
OUT=outputs/diagnostics
mkdir -p ${OUT} /scratch/${USER}/pps_qj/logs

echo "================================================================"
echo "OVERNIGHT DIAGNOSTICS  job=${SLURM_JOB_ID:-NA}  workers=${WORKERS}"
echo "START $(date)"
echo "================================================================"

echo ""; echo "### [1/3] t_saturation (L=32,64,128) $(date) ###"
python3 analysis/t_saturation.py \
  --points 32:0.15:0.40,64:0.15:0.40,128:0.15:0.40,32:0.50:0.55,64:0.50:0.55,128:0.50:0.55 \
  --N_c 60 --R 6 --fracs 0.3,0.5,0.7,1.0 \
  --workers ${WORKERS} --outdir ${OUT}/tsat \
  || echo "!!! t_saturation FAILED (continuing)"

echo ""; echo "### [2/3] d5 clones-vs-seeds L=96 $(date) ###"
python3 analysis/d5_clones_vs_seeds.py \
  --L 96 --T 30 --R 20 --Ncs 25,50,100,150,200 \
  --workers ${WORKERS} --outdir ${OUT}/d5x_L96 \
  || echo "!!! d5 L96 FAILED (continuing)"

echo ""; echo "### [3/3] d5 clones-vs-seeds L=128 $(date) ###"
python3 analysis/d5_clones_vs_seeds.py \
  --L 128 --T 30 --R 20 --Ncs 25,50,100,150,200 \
  --workers ${WORKERS} --outdir ${OUT}/d5x_L128 \
  || echo "!!! d5 L128 FAILED (continuing)"

echo ""; echo "================================================================"
echo "DONE $(date)"
echo "Results:"
echo "  ${OUT}/tsat/t_saturation_summary.json"
echo "  ${OUT}/d5x_L96/d5x_summary.json"
echo "  ${OUT}/d5x_L128/d5x_summary.json"
echo "================================================================"
