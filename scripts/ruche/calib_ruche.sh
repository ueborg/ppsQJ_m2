#!/bin/bash
#SBATCH --job-name=pps_calib
#SBATCH --output=logs/calib_%j.out
#SBATCH --error=logs/calib_%j.err
#SBATCH -p cpu_med
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# One production realization (L=128, Nc=128, T=128, lowrank+newton) on ONE core,
# to fix r = (Ruche core time)/(Mac core time). Mac baseline = 25.5 min.
# Submit from the repo root:  sbatch scripts/ruche/calib_ruche.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/ppsQJ_m2}"
CONDA_ENV="${CONDA_ENV:-$WORKDIR/envs/pps_qj}"
mkdir -p logs
module purge
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun python - <<'PY'
import time, numpy as np
from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.cloning import run_cloning
L,alpha,zeta,Nc=128,0.35,0.30,128; T=float(L)
lam=0.5*np.sqrt(zeta); dtau=12.0/max(2*alpha*(L-1),1e-6)
m=build_gaussian_chain_model(L,1-lam,lam); t0=time.time()
r=run_cloning(m,zeta,T,Nc,np.random.default_rng(20260624),delta_tau=dtau,record_entropy=True,
              proposal_c=zeta,jump_update_method="lowrank",solver_method="newton")
w=time.time()-t0
print(f"RUCHE calib L=128 Nc=128 T=128 newton: wall={w:.1f}s ({w/60:.2f} min) ESS={r.eff_sample_size:.1f}")
print(f"Mac baseline 25.5 min -> r = Ruche/Mac = {w/60/25.5:.2f}  (scale campaign estimates by r)")
PY
