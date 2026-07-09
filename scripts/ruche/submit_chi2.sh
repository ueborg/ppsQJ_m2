#!/bin/bash
#SBATCH --job-name=pps_chi2
#SBATCH --output=logs/chi2_%A_%a.out
#SBATCH --error=logs/chi2_%A_%a.err
#SBATCH -p cpu_med
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --mem=80G
# chi_2(L)=d^2 I/dzeta^2|_0 via click-sector MC. Each array task = one L; it runs
# NSEED seeds across the 40 cores (one seed per core), each N2 samples, writing
# poolable JSONs. Aggregate pools+jackknifes across seeds.
#   sbatch --array=0-4 --export=ALL,OUTDIR=$WORKDIR/pps/chi2,US=0.75,CS=1.0,NSEED=40,N2=200000 \
#       scripts/ruche/submit_chi2.sh          # array index -> L in LS below
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/ppsQJ_m2}"; CONDA_ENV="${CONDA_ENV:-$WORKDIR/envs/pps_qj}"
OUTDIR="${OUTDIR:?set OUTDIR}"; US="${US:-0.75}"; CS="${CS:-1.0}"
NSEED="${NSEED:-40}"; N1="${N1:-5000}"; N2="${N2:-200000}"
LS_ARR=(${LS:-32 48 64 96 128}); L=${LS_ARR[${SLURM_ARRAY_TASK_ID:-0}]}
mkdir -p logs "$OUTDIR"
module purge; module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "L=$L US=$US CS=$CS NSEED=$NSEED N2=$N2 OUTDIR=$OUTDIR"
seq 0 $((NSEED-1)) | xargs -P "${SLURM_CPUS_PER_TASK:-40}" -I{} \
  python "$REPO_DIR/analysis/chi2_worker.py" --L "$L" --u "$US" --c "$CS" \
    --seed {} --N1 "$N1" --N2 "$N2" --outdir "$OUTDIR"
echo "L=$L done"
