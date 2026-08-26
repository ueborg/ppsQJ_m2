#!/bin/bash
#SBATCH --job-name=pps_bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#
# =============================================================================
# Cost model for the QJ-PPS simulator on Ruche.  scripts/benchmark_scaling.py
#
# SCOPE: timing only.  DEC-MASTER-METRIC-001 marks wall time diagnostic_only.
# This budgets a campaign and picks an L ladder.  It may NOT be used to claim
# any configuration is better; that needs t_wall * sigma^2(lambda_c).
#
# ONE CPU on purpose.  BLAS is pinned to a single thread in production, so a
# single-core measurement is the one that transfers.  Do not give it 40 cores.
#
# ENV:  OUT (required)   SWEEP (default all)   REPS (default 3)
#
# RUN THE CHEAP SWEEPS FIRST, they take minutes:
#   sbatch -p cpu_med --time=00:40:00 \
#     --export=ALL,OUT=$WORKDIR/pps/bench/bench_L.json,SWEEP=L,REPS=3 \
#     scripts/ruche/submit_benchmark.sh
#
#   ... same for SWEEP=N_c, zeta, dtau, lam, solver
#
# THE ONE THE BUDGET RESTS ON.  Lprod runs T = L at N_c = 128, i.e. genuine
# production shape, at L = 24..96.  L=96 alone is ~40 min single-core.
#   sbatch -p cpu_med --time=04:00:00 \
#     --export=ALL,OUT=$WORKDIR/pps/bench/bench_Lprod.json,SWEEP=Lprod,REPS=2 \
#     scripts/ruche/submit_benchmark.sh
#
# T-LINEARITY licenses extrapolating the short sweeps to production T:
#   sbatch -p cpu_med --time=01:00:00 \
#     --export=ALL,OUT=$WORKDIR/pps/bench/bench_T.json,SWEEP=T,REPS=3 \
#     scripts/ruche/submit_benchmark.sh
#
# DRY RUN COSTS NOTHING, do it on the login node:
#   python scripts/benchmark_scaling.py --out /tmp/b.json --sweep all --dry-run
# =============================================================================
#
# PARTITIONS.  cpu_short rejected --time=02:00:00 on 2026-08-26
# cap measured 2026-08-26: cpu_short 1:00:00, cpu_med 4:00:00, cpu_prod 6:00:00,
# known-good (the July campaign used it at 04:00:00).  Check limits with
#   sinfo -o "%20P %10l %10L %6D %C"
# and prefer the shortest partition that actually accepts your --time.
# =============================================================================

set -euo pipefail
: "${OUT:?OUT is required}"
SWEEP="${SWEEP:-all}"
REPS="${REPS:-3}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$(dirname "$OUT")"

module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORKDIR/envs/pps_qj"

echo "HOST=$(hostname)  COMMIT=$(git rev-parse HEAD)"
echo "CPU=$(lscpu | grep -m1 'Model name' | cut -d: -f2- | xargs)"
python --version

python scripts/benchmark_scaling.py --out "$OUT" --sweep "$SWEEP" --reps "$REPS"
