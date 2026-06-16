#!/bin/bash
# Cut B guided-cloning PRODUCTION scan (dense low-zeta crossing campaign).
#
# Grid: make_clone_guided_grid() in grid_pps.py  (worker --grid guided)
#   5 L {32,48,64,96,128} x 15 zeta {0.05..0.85} x 13 lambda = 975 tasks.
#   lambda windows centered on the MEASURED crossing lambda_c ~ 0.51*sqrt(zeta),
#   +/-0.08 (brackets finite-L drift). N_c = 500/400/350/300/300 by L.
#   Guided (PPS_GUIDED=1, proposal_c=zeta) + x6 window lengthening: validated
#   exact vs standard (B_L 0.66 sigma at the crossing), ESS ~0.98 vs ~0.37,
#   B_L ~4.5x tighter. nwin>=51 at every grid point (equilibration safe).
#
# Habrok MaxArraySize=1001 -> all task ids (0..974) are valid array indices.
#
# SIZE-BINNED SUBMISSION (recommended; overrides the #SBATCH defaults below):
#   sbatch --array=0-194%96   --time=02:00:00 --job-name=gB_L32  scripts/habrok/pps_scan/submit_clone_guided_prod.sh
#   sbatch --array=195-389%96 --time=03:00:00 --job-name=gB_L48  scripts/habrok/pps_scan/submit_clone_guided_prod.sh
#   sbatch --array=390-584%64 --time=05:00:00 --job-name=gB_L64  scripts/habrok/pps_scan/submit_clone_guided_prod.sh
#   sbatch --array=585-779%48 --time=08:00:00 --job-name=gB_L96  scripts/habrok/pps_scan/submit_clone_guided_prod.sh
#   sbatch --array=780-974%32 --time=12:00:00 --job-name=gB_L128 scripts/habrok/pps_scan/submit_clone_guided_prod.sh
# (Walls are generous; trim once the first tasks of each tier report timings.)
#
#SBATCH --job-name=gB_guided
#SBATCH --output=logs/gB_guided_%A_%a.out
#SBATCH --error=logs/gB_guided_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --array=0-194%96

REPO_DIR="${REPO_DIR:-$HOME/pps_qj}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/pps_qj}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/pps_qj/pps_clone_guided_prod}"

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

# Single-threaded BLAS (parallelism is across the 5 realisations).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PPS_N_WORKERS=5

# Guided cloning + x6 window lengthening + full observable set.
export PPS_GUIDED=1
export PPS_DTAU_MULT=6
export PPS_RECORD_RENYI=1

cd "$REPO_DIR"
python -m pps_qj.parallel.worker_clone_pps \
    "$SLURM_ARRAY_TASK_ID" "$OUTPUT_DIR" --grid guided
