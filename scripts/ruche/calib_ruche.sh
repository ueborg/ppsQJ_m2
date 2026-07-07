#!/bin/bash
#SBATCH --job-name=pps_calib
#SBATCH --output=logs/calib_%j.out
#SBATCH --error=logs/calib_%j.err
#SBATCH -p cpu_short
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# Self-pinning calibration -> fixes r = (Ruche core time)/(Mac core time).
# Threads are pinned INSIDE python (before numpy import), so MKL cannot
# oversubscribe regardless of srun env propagation; progress is streamed.
# Default runs a reduced T=32 (cloning wall is linear in T) and extrapolates
# to the production T=128 (~8-10 min instead of ~40). CALIB_T=128 = exact run.
# Mac T=128 baseline = 25.5 min.
#   sbatch scripts/ruche/calib_ruche.sh            # quick T=32
#   CALIB_T=128 sbatch scripts/ruche/calib_ruche.sh   # full, exact
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/ppsQJ_m2}"
CONDA_ENV="${CONDA_ENV:-$WORKDIR/envs/pps_qj}"
CALIB_T="${CALIB_T:-32}"
mkdir -p logs
module purge
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR"
srun python - "$CALIB_T" <<'PY'
import os, sys
# Pin BLAS threads BEFORE importing numpy -- reliable regardless of srun env.
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
print("[calib] pinned BLAS to 1 thread before numpy import", flush=True)
import time, numpy as np
from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.cloning import run_cloning
Tcal = float(sys.argv[1]); L = 128; alpha = 0.35; zeta = 0.30; Nc = 128
lam = 0.5 * np.sqrt(zeta); dtau = 12.0 / max(2 * alpha * (L - 1), 1e-6)
m = build_gaussian_chain_model(L, 1 - lam, lam)
print(f"[calib] running L={L} Nc={Nc} T={Tcal:.0f} lowrank+newton ...", flush=True)
t0 = time.time()
r = run_cloning(m, zeta, Tcal, Nc, np.random.default_rng(20260624), delta_tau=dtau,
                record_entropy=True, proposal_c=zeta, jump_update_method="lowrank",
                solver_method="newton")
w = time.time() - t0
full = w * (128.0 / Tcal)   # cloning wall is linear in T
print(f"[calib] wall(T={Tcal:.0f})={w:.1f}s ({w/60:.2f} min)  ESS={r.eff_sample_size:.1f}", flush=True)
print(f"[calib] predicted full T=128 = {full/60:.2f} min", flush=True)
print(f"[calib] Mac baseline 25.5 min  ->  r = Ruche/Mac = {full/60/25.5:.2f}", flush=True)
print("[calib] scale ALL campaign wall-time estimates by r.", flush=True)
PY
