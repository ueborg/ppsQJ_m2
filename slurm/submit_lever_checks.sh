#!/bin/bash
# =============================================================================
# submit_lever_checks.sh  --  measure all three cost levers in one job
#
# Runs back-to-back on a dedicated full node, each using the whole node via
# ProcessPoolExecutor. Non-fatal: if one check dies, the others still run.
#
#   1. bench_dtau         -> largest SAFE PPS_DTAU_MULT (the ~2x step lever)
#   2. bl_saturation      -> does B_L(t)/CMI(t) flatten before T_prod? (the ~2x
#                            T lever), at the two phase-diagram points we plotted
#   3. bench_blas_threads -> does threaded BLAS lower the L=128 wall floor?
#
# Together 1 and 2 give firm numbers on both halves of a potential ~4x before
# committing any campaign.
#
# Submit:  sbatch slurm/submit_lever_checks.sh
# =============================================================================
#SBATCH --job-name=lever_checks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --partition=regular
#SBATCH --output=/scratch/%u/pps_qj/logs/lever_checks_%j.out
#SBATCH --error=/scratch/%u/pps_qj/logs/lever_checks_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=%u@rug.nl
set -uo pipefail   # NOT -e: one check failing must not abort the rest

module purge
source ${HOME}/venvs/pps_qj/bin/activate
cd ${HOME}/pps_qj

# Global single-thread BLAS for the realisation-parallel checks (1 & 2).
# bench_blas_threads (3) self-spawns subprocesses that override these per run.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PPS_BACKEND=scalar

WORKERS=${WORKERS:-120}
OUT=outputs/diagnostics/levers
mkdir -p ${OUT} /scratch/${USER}/pps_qj/logs

# Saturation test points. Defaults MATCH the panels we looked at so the R=20
# curves directly confirm/refute the noisy R=6 ladders. NOTE: relaxation is
# slowest exactly at lambda_c (~0.96*sqrt(zeta) => ~0.37 and ~0.68); if these
# pass cleanly, a final confirmation at lambda_c is the conservative check.
LAM_Z015=${LAM_Z015:-0.40}
LAM_Z050=${LAM_Z050:-0.55}

echo "================================================================"
echo "LEVER CHECKS  job=${SLURM_JOB_ID:-NA}  workers=${WORKERS}"
echo "START $(date)"
echo "================================================================"

echo ""; echo "### [1/4] bench_dtau (L=32, near lambda_c, conservative) $(date) ###"
python3 analysis/bench_dtau.py \
  --L 32 --zetas 0.15,0.30 --lams 0.37,0.53 --mults 1,1.5,2,3 \
  --R 8 --N_c 200 --T 40 --workers ${WORKERS} --outdir ${OUT} \
  || echo "!!! bench_dtau FAILED (continuing)"

echo ""; echo "### [2/4] bl_saturation zeta=0.15 (L=64,128) $(date) ###"
python3 analysis/bl_saturation.py \
  --Ls 64,128 --zeta 0.15 --lam ${LAM_Z015} --N_c 50 --R 20 --n-snap 12 \
  --workers ${WORKERS} --outdir ${OUT}/blsat_z015 \
  || echo "!!! bl_saturation z0.15 FAILED (continuing)"

echo ""; echo "### [3/4] bl_saturation zeta=0.50 (L=64,128) $(date) ###"
python3 analysis/bl_saturation.py \
  --Ls 64,128 --zeta 0.50 --lam ${LAM_Z050} --N_c 50 --R 20 --n-snap 12 \
  --workers ${WORKERS} --outdir ${OUT}/blsat_z050 \
  || echo "!!! bl_saturation z0.50 FAILED (continuing)"

echo ""; echo "### [4/4] bench_blas_threads (L=64,128) $(date) ###"
python3 analysis/bench_blas_threads.py \
  --Ls 64,128 --threads 1,2,4,8 --N_c 60 --T 3 --outdir ${OUT} \
  || echo "!!! bench_blas_threads FAILED (continuing)"

echo ""; echo "================================================================"
echo "DONE $(date)"
echo "Results:"
echo "  ${OUT}/bench_dtau.json            (largest safe dtau multiplier)"
echo "  ${OUT}/blsat_z015/bl_saturation.{json,png}   (T lever, small zeta)"
echo "  ${OUT}/blsat_z050/bl_saturation.{json,png}   (T lever, large zeta)"
echo "  ${OUT}/bench_blas_threads.json    (L=128 wall floor vs threads)"
echo "================================================================"
