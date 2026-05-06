#!/bin/bash
# =============================================================================
# submit_benchmark.sh  —  Runtime benchmark vs production wall times
#
# Reruns a small set of completed production task_ids in a throwaway directory
# and compares measured times to the wall_time stored in the production .npz.
#
# Two things are tested:
#   1. Hardware match:  t_bench / t_prod ≈ 1 means interactive == compute node
#   2. Scaling model:  ratios consistent across L values means our L^4 scaling
#                      formula correctly predicts L=128 runtimes
#
# Output:  $LOG_DIR/benchmark_<jobid>.out
# Results: $BENCH_DIR/benchmark_results.txt  (tabular, for easy parsing)
#
# Usage:
#   bash slurm/submit_benchmark.sh
#
# After job completes:
#   cat /scratch/$USER/pps_qj/benchmark_*/benchmark_results.txt
# =============================================================================
set -euo pipefail

PARTITION="regular"
WALL_TIME="00:45:00"    # 6 tasks × ~5min each max; generous margin
LOG_DIR="/scratch/${USER}/pps_qj/logs"
PROD_DIR="/scratch/${USER}/pps_qj/pps_clone_v2"
BENCH_DIR="/scratch/${USER}/pps_qj/benchmark_$(date +%Y%m%d_%H%M)"

mkdir -p "${LOG_DIR}"

sbatch <<SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=pps_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=${WALL_TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --output=${LOG_DIR}/benchmark_%j.out
#SBATCH --error=${LOG_DIR}/benchmark_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rug.nl

set -euo pipefail

echo "=================================================================="
echo "PPS Runtime Benchmark  —  Job \${SLURM_JOB_ID}"
echo "Node: \$(hostname)   CPU: \$(nproc) cores   Started: \$(date)"
echo "=================================================================="

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

# Match production environment exactly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p ${BENCH_DIR}

python3 - <<'PYEOF'
import time, os, sys
import numpy as np

from pps_qj.parallel.grid_pps import task_params_clone_v2
from pps_qj.parallel.worker_clone_pps import _run_one_realisation

N_REAL    = 5
PROD_DIR  = os.environ.get("PROD_DIR",  "/scratch/{u}/pps_qj/pps_clone_v2".format(u=os.environ["USER"]))
BENCH_DIR = os.environ.get("BENCH_DIR", "/scratch/{u}/pps_qj/benchmark".format(u=os.environ["USER"]))

# ── Task selection ────────────────────────────────────────────────────────────
# Six tasks spanning (L=32,48,64) × (λ=small,large).
# All have completed in production — wall_time is stored in their .npz.
#
#  tid  L   lam   zeta   N_c   T      Expected regime
#  720  32  0.02  0.02   500  200     fast  (large δτ, few steps)
#  743  32  0.90  0.02   500  200     slow  (small δτ, many steps)
#  960  48  0.02  0.02   300  200     fast
#  983  48  0.90  0.02   300  200     slow
# 1200  64  0.02  0.02   200  150     fast
# 1223  64  0.90  0.02   200  150     slow

TASK_IDS = [720, 743, 960, 983, 1200, 1223]

# ── Load production wall times ────────────────────────────────────────────────
prod_wt = {}
for tid in TASK_IDS:
    path = f"{PROD_DIR}/clone_{tid:05d}.npz"
    if os.path.exists(path):
        try:
            prod_wt[tid] = float(np.load(path)["wall_time"])
        except Exception as e:
            print(f"  WARNING: could not load {path}: {e}")
    else:
        print(f"  WARNING: production file not found: {path}")

# ── Run benchmark ─────────────────────────────────────────────────────────────
print()
print(f"{'tid':>5}  {'L':>4}  {'lam':>6}  {'ζ':>5}  {'N_c':>5}  {'T':>5}"
      f"  {'t_prod':>8}  {'t_bench':>8}  {'ratio':>7}  status")
print("-" * 75)

results = []
for tid in TASK_IDS:
    task = task_params_clone_v2(tid)
    L, w, alpha, zeta = task["L"], task["w"], task["alpha"], task["zeta"]
    T, N_c, seed       = task["T"], task["N_c"], task["seed"]

    t0 = time.perf_counter()
    for r in range(N_REAL):
        _run_one_realisation(dict(
            L=L, w=w, alpha=alpha, zeta=zeta, T=T, N_c=N_c,
            seed=seed + r * 999_983,
        ))
    t_bench = time.perf_counter() - t0

    t_prod = prod_wt.get(tid, float("nan"))
    ratio  = t_bench / t_prod if not np.isnan(t_prod) else float("nan")

    if np.isnan(ratio):
        status = "NO_PROD"
    elif 0.70 < ratio < 1.40:
        status = "OK"
    elif ratio >= 1.40:
        status = "BENCH_SLOWER"
    else:
        status = "BENCH_FASTER"

    row = dict(tid=tid, L=L, lam=task["lam"], zeta=zeta, N_c=N_c, T=T,
               t_prod=t_prod, t_bench=t_bench, ratio=ratio, status=status)
    results.append(row)
    print(f"{tid:>5}  {L:>4}  {task['lam']:>6.3f}  {zeta:>5.2f}  {N_c:>5}  {T:>5.0f}"
          f"  {t_prod:>8.1f}  {t_bench:>8.1f}  {ratio:>7.3f}  {status}")
    sys.stdout.flush()

# ── Summary ───────────────────────────────────────────────────────────────────
print()
valid = [r for r in results if r["status"] != "NO_PROD"]
if valid:
    ratios = [r["ratio"] for r in valid]
    print(f"Ratio summary (t_bench / t_prod):")
    print(f"  mean  = {np.mean(ratios):.3f}")
    print(f"  std   = {np.std(ratios):.3f}")
    print(f"  range = [{min(ratios):.3f}, {max(ratios):.3f}]")
    print()

    # ── Scaling check: does ratio(L=32) ≈ ratio(L=48) ≈ ratio(L=64)? ─────────
    by_L = {}
    for r in valid:
        by_L.setdefault(r["L"], []).append(r["ratio"])
    print("Ratio by L (consistency check for scaling model):")
    for Lv, rlist in sorted(by_L.items()):
        print(f"  L={Lv:3d}: {[f'{x:.3f}' for x in rlist]}  mean={np.mean(rlist):.3f}")
    print()
    print("If ratios are consistent across L → scaling model is correct.")
    print("If ratios vary with L → L^4 scaling formula needs recalibration.")
    print()

    # ── Recalibrated L=128 estimate ───────────────────────────────────────────
    calib = np.mean(ratios)
    t64_by_zeta = {
        0.02:4379, 0.05:4411, 0.10:4459, 0.15:4488, 0.20:4538,
        0.30:4666, 0.50:5281, 0.70:6275, 0.85:7430, 1.00:9699,
    }
    r96_64  = (150*100*96**4) / (200*150*64**4)
    r128_64 = (100*100*128**4) / (200*150*64**4)
    t128_total = sum(t64_by_zeta[z] * r128_64 for z in t64_by_zeta) * calib
    print(f"Recalibrated L=128 total estimate (calibration={calib:.3f}×):")
    print(f"  {t128_total/3600:.1f}h  ({t128_total/60:.0f} min)")
    print(f"  Recommended sbatch --time: {int(t128_total/3600)+10}:00:00")

# Write machine-readable results
import csv
out_path = f"{BENCH_DIR}/benchmark_results.txt"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print(f"\nResults written to: {out_path}")
PYEOF

echo "=================================================================="
echo "Benchmark complete: \$(date)"
echo "Results: ${BENCH_DIR}/benchmark_results.txt"
echo "=================================================================="
SLURM_SCRIPT

echo "Submitted benchmark job."
echo "Monitor:  squeue -u \$USER -n pps_benchmark"
echo "Output:   tail -f ${LOG_DIR}/benchmark_<jobid>.out"
echo "Results:  cat /scratch/\$USER/pps_qj/benchmark_*/benchmark_results.txt"
