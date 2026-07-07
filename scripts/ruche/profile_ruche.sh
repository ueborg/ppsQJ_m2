#!/bin/bash
#SBATCH --job-name=pps_profile
#SBATCH --output=logs/profile_%j.out
#SBATCH --error=logs/profile_%j.err
#SBATCH -p cpu_short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# Comprehensive profile of the guided-cloning trajectory on Ruche (1 pinned
# core). profile_code.py streams progress to this .out (phase markers +
# 10 cloning checkpoints), so `tail -f logs/profile_<jobid>.out` shows it live.
# ~5-10 min on cpu_short. Submit from the repo root:
#   sbatch scripts/ruche/profile_ruche.sh
#   PROFILE_LS="128 192 256" sbatch scripts/ruche/profile_ruche.sh
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/ppsQJ_m2}"
CONDA_ENV="${CONDA_ENV:-$WORKDIR/envs/pps_qj}"
PROFILE_LS="${PROFILE_LS:-128 192}"
mkdir -p logs
module purge
module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO_DIR"
# profile_code.py pins BLAS threads itself (before numpy import).
srun python "$REPO_DIR/scripts/ruche/profile_code.py" $PROFILE_LS
