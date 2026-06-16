#!/bin/bash
# Guided-cloning PILOT — validate the thinned-proposal (c=zeta) cloning upgrade
# at scale before committing to a full low-zeta campaign.
#
# Method: importance-sampling proposal at intensity c*lambda (c=zeta) with the
# exact compensator weight exp[-(1-zeta)*Delta_Lambda] per window (PPS_GUIDED=1),
# combined with x8 window lengthening (PPS_DTAU_MULT=8). Local tests (L<=16):
#   - exact vs brute-force zeta^N reweighting (0.3 sigma) and vs standard cloning;
#   - at x8 windows, standard cloning ESS collapses (0.35 -> 0.11 -> worst-window
#     0.007) while guided stays healthy (0.97 -> 0.92), giving ~7-8x fewer windows
#     at maintained ESS;
#   - smaller finite-N_c bias (flatter 1/N_c extrapolation).
# Equilibration constraint: keep >~30 windows. All pilot points have win_x8>=50.
#
# Pilot grid (default v1 = make_clone_grid), zeta=0.1, lambda in {0.10,0.15,0.20}:
#   L=32  (N_c=500, T=64) : task_ids 510,525,540
#   L=64  (N_c=200, T=128): task_ids 765,780,795
#   L=128 (N_c=200, T=128): task_ids 1020,1035,1050
#
# A/B usage (run twice, different OUTPUT_DIR):
#   guided   : sbatch submit_clone_guided_pilot.sh                      (defaults)
#   standard : PPS_GUIDED=0 PPS_DTAU_MULT=1 \
#              OUTPUT_DIR=/scratch/$USER/pps_qj/pps_clone_std_pilot \
#              sbatch submit_clone_guided_pilot.sh
# Then compare S_mean / CMI / theta at matched (L,lam): must agree within error;
# compare wall time and ess_history (guided should be ~7-8x faster, higher ESS).
#
#SBATCH --job-name=pps_guided_pilot
#SBATCH --output=logs/pps_guided_pilot_%A_%a.out
#SBATCH --error=logs/pps_guided_pilot_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
# NOTE: L=128 (v1-grid ids 1020,1035,1050) EXCEED Habrok MaxArraySize=1001 and
# cannot be array indices from the v1 grid. Use the production grid (--grid
# guided, submit_clone_guided_prod.sh) for L=128. Pilot is L=32,64 only.
#SBATCH --array=510,525,540,765,780,795%6

REPO_DIR="${REPO_DIR:-$HOME/pps_qj}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/pps_qj}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/pps_qj/pps_clone_guided_pilot}"

mkdir -p logs "$OUTPUT_DIR"
module purge
module load Python/3.10.8-GCCcore-12.2.0
# Do NOT load SciPy-bundle: its numpy 1.24 lacks Generator.spawn (added in
# numpy 1.25) and shadows the venv's numpy 2.2.6 via PYTHONPATH on compute
# nodes, which silently NaNs every realisation. The venv is self-contained
# for numpy/scipy, so use it alone and pin PYTHONPATH to just the repo.
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR"
python -c "import numpy; print('using numpy', numpy.__version__, '@', numpy.__file__)"

# Single-threaded BLAS: parallelism is across the 5 realisations, not within.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=5

# Guided-cloning controls (override for the standard A/B baseline).
export PPS_GUIDED="${PPS_GUIDED:-1}"
export PPS_DTAU_MULT="${PPS_DTAU_MULT:-8}"

cd "$REPO_DIR"
python -m pps_qj.parallel.worker_clone_pps \
    "$SLURM_ARRAY_TASK_ID" "$OUTPUT_DIR"
