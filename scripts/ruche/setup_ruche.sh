#!/bin/bash
# One-time Ruche environment setup for ppsQJ_m2.
# Run ONCE on a Ruche LOGIN node from the repo root:
#   cd "$HOME/ppsQJ_m2" && bash scripts/ruche/setup_ruche.sh
# Creates a conda env (numpy>=2 + scipy, MKL BLAS) under $WORKDIR, verifies
# imports, and runs a few-second end-to-end smoke test. The heavy calibration
# is a separate Slurm job (scripts/ruche/calib_ruche.sh) -- do NOT run big jobs
# on the login node.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/ppsQJ_m2}"
CONDA_ENV="${CONDA_ENV:-$WORKDIR/envs/pps_qj}"
echo "REPO_DIR=$REPO_DIR   CONDA_ENV=$CONDA_ENV"

module purge
module load anaconda3/2023.09-0/none-none

# Rocky8: keep conda pkgs/envs off the 50GB $HOME; use $WORKDIR.
export CONDA_PKGS_DIRS="$WORKDIR/.conda/pkgs"
mkdir -p "$WORKDIR/.conda/pkgs" "$(dirname "$CONDA_ENV")" "$REPO_DIR/logs"

if [ ! -d "$CONDA_ENV" ]; then
  echo "Creating conda env at $CONDA_ENV (numpy>=2 + scipy, MKL) ..."
  conda create -y -p "$CONDA_ENV" python=3.12 "numpy>=2" scipy
else
  echo "Env already exists: $CONDA_ENV (skipping create)"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "== versions =="
python -c "import numpy,scipy,numpy.random as r; assert hasattr(r.default_rng(0),'spawn'); print('numpy',numpy.__version__,'scipy',scipy.__version__,'(Generator.spawn OK)')"

echo "== import smoke =="
python -c "from pps_qj.gaussian_backend import build_gaussian_chain_model; from pps_qj.cloning import run_cloning; print('pps_qj imports OK')"

echo "== end-to-end smoke (L=32, Nc=8, T=8; a few seconds) =="
python - <<'PY'
import numpy as np, time
from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.cloning import run_cloning
L,alpha,zeta,Nc=32,0.35,0.5,8
lam=0.5*np.sqrt(zeta); dtau=12.0/max(2*alpha*(L-1),1e-6)
m=build_gaussian_chain_model(L,1-lam,lam); t0=time.time()
r=run_cloning(m,zeta,8.0,Nc,np.random.default_rng(1),delta_tau=dtau,record_entropy=True,
              proposal_c=zeta,jump_update_method="lowrank",solver_method="newton")
print(f"smoke OK: theta={r.theta_hat:.3f} ESS={r.eff_sample_size:.1f} wall={time.time()-t0:.1f}s")
PY
echo "== setup complete. Next: sbatch scripts/ruche/calib_ruche.sh (measures r). =="
