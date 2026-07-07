#!/bin/bash
#SBATCH --job-name=pps_ruche
#SBATCH --output=logs/pps_%A_%a.out
#SBATCH --error=logs/pps_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
# NOTE: -p <partition>, --time and --array are set on the sbatch COMMAND LINE
#       per size-tier (see SUBMISSION block). cpus-per-task=40 = one full Ruche
#       node; control the concurrent-core footprint with --array=...%K so that
#       K*40 <= the partition's per-user core cap (cpu_med 1000 -> K<=25 ;
#       cpu_long 160 -> K<=4 ; cpu_prod 2000 -> K<=50).
#
# =============================================================================
# Ruche (Paris-Saclay Mesocentre) Slurm-array submit for the ppsQJ_m2
# boundary / nu campaign.
#
# Each array task owns a DISJOINT round-robin shard of the realization list
# (run_local_boundary.py --shard $SLURM_ARRAY_TASK_ID --nshards $N). Per-
# realization JSON checkpoints make every task resumable and idempotent, so a
# requeued/re-run task never duplicates finished work. BLAS pinned to 1 thread;
# nworkers = cpus-per-task, i.e. one realization per core, full node per task.
#
# CONFIG via env (override with --export=ALL,VAR=...):
#   OUTDIR     output dir on GPFS (REQUIRED)
#   LS         space list, e.g. "64 96 128"
#   ZETAS      space list, e.g. "0.10 0.20 0.30 0.50 1.00"
#   LAM_MULTS  space list of multipliers of lam_c(zeta)=0.5*sqrt(zeta)
#   NREAL NC DTAU_MULT SOLVER
#
# SUBMISSION (size-binned). Replace $WORK with your GPFS workdir.
#
#  # -- Tier 1  phase-boundary map  (L<=128)  -> cpu_med, 1000 cores
#  sbatch -p cpu_med --time=04:00:00 --array=0-39%25 \
#    --export=ALL,OUTDIR=$WORK/pps/boundary,LS="64 96 128",\
#ZETAS="0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.70 1.00",\
#LAM_MULTS="0.7 0.85 1.0 1.15 1.3",NREAL=12,NC=128,SOLVER=newton \
#    scripts/ruche/submit_pps_boundary.sh
#
#  # -- Tier 2  nu, L<=192  (each realization <=3h)  -> cpu_med, 1000 cores
#  sbatch -p cpu_med --time=04:00:00 --array=0-39%25 \
#    --export=ALL,OUTDIR=$WORK/pps/nu_lowL,LS="64 96 128 160 192",\
#ZETAS="0.20 0.30 0.50",LAM_MULTS="0.8 0.9 1.0 1.1 1.2",NREAL=24,NC=128,SOLVER=newton \
#    scripts/ruche/submit_pps_boundary.sh
#
#  # -- Tier 3  nu anchor L=256  (~10h/realization)  -> cpu_long, 160 cores
#  #    Keep K<=4 (160 cores). 72h walltime covers several 10h waves per task.
#  sbatch -p cpu_long --time=72:00:00 --array=0-7%4 \
#    --export=ALL,OUTDIR=$WORK/pps/nu_L256,LS="256",\
#ZETAS="0.20 0.30 0.50",LAM_MULTS="0.9 0.95 1.0 1.05 1.1",NREAL=24,NC=128,SOLVER=newton \
#    scripts/ruche/submit_pps_boundary.sh
#
#  After completion, aggregate crossings locally or on a login node:
#    python scripts/run_local_boundary.py aggregate --outdir $WORK/pps/nu_lowL
# =============================================================================

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/ppsQJ_m2}"
CONDA_ENV="${CONDA_ENV:-$WORKDIR/envs/pps_qj}"
OUTDIR="${OUTDIR:?set OUTDIR (e.g. \$WORKDIR/pps/nu_lowL)}"
LS="${LS:-64 96 128}"
ZETAS="${ZETAS:-0.25 0.30}"
LAM_MULTS="${LAM_MULTS:-0.85 0.925 1.0 1.075 1.15}"
NREAL="${NREAL:-12}"
NC="${NC:-128}"
DTAU_MULT="${DTAU_MULT:-12}"
SOLVER="${SOLVER:-newton}"

mkdir -p logs "$OUTDIR"

module purge
module load anaconda3/2023.09-0/none-none
# conda env with numpy 2.x + MKL BLAS; create once via scripts/ruche/setup_ruche.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR"

# One BLAS thread per realization; parallelism is across realizations.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))}"
[ "$NSHARDS" -lt 1 ] && NSHARDS=1
NWORKERS="${SLURM_CPUS_PER_TASK:-1}"

python -c "import numpy,scipy; print('numpy',numpy.__version__,'scipy',scipy.__version__,'|',numpy.show_config())" 2>/dev/null | head -1
echo "INFO shard $SHARD/$NSHARDS nworkers=$NWORKERS OUTDIR=$OUTDIR"
echo "INFO LS=[$LS] ZETAS=[$ZETAS] LAM_MULTS=[$LAM_MULTS] NREAL=$NREAL NC=$NC SOLVER=$SOLVER"

srun python "$REPO_DIR/scripts/run_local_boundary.py" run \
  --outdir "$OUTDIR" \
  --Ls $LS --zetas $ZETAS --lam-mults $LAM_MULTS \
  --nreal "$NREAL" --Nc "$NC" --Tmult 1.0 --dtau-mult "$DTAU_MULT" \
  --solver "$SOLVER" --entropy-stride 4 \
  --nworkers "$NWORKERS" --shard "$SHARD" --nshards "$NSHARDS"
