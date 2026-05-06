#!/bin/bash
# =============================================================================
# submit_benchmark.sh  —  Runtime benchmark vs production wall times
#
# Reruns a small set of completed production task_ids in a throwaway directory
# and compares measured times to the wall_time stored in the production .npz.
#
# Two things are tested:
#   1. Hardware match:  t_bench / t_prod ≈ 1 means compute node == omni40
#   2. Scaling model:  work ∝ N_c·T·L^3 confirmed by (32,48) pair comparison
#
# Tasks run in parallel (6 workers, one per task) so total wall time ≈ slowest
# single task (~25 min) rather than sum of all tasks (~3h serially).
#
# Output:  $LOG_DIR/benchmark_<jobid>.out
# Results: $BENCH_DIR/benchmark_results.txt  (CSV, for easy parsing)
#
# Usage:
#   bash slurm/submit_benchmark.sh
#
# After job completes:
#   cat /scratch/$USER/pps_qj/benchmark_*/benchmark_results.txt
# =============================================================================
set -euo pipefail

PARTITION="regular"
WALL_TIME="02:00:00"    # 6 tasks in parallel; slowest ~1500s, generous margin
LOG_DIR="/scratch/${USER}/pps_qj/logs"
PROD_DIR="/scratch/${USER}/pps_qj/pps_clone_v2"
BENCH_DIR="/scratch/${USER}/pps_qj/benchmark_$(date +%Y%m%d_%H%M)"

mkdir -p "${LOG_DIR}"

sbatch <<SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=pps_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=6
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

# Match production environment exactly — single-threaded BLAS per process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p ${BENCH_DIR}
export PROD_DIR="${PROD_DIR}"
export BENCH_DIR="${BENCH_DIR}"

python3 - <<'PYEOF'
import time, os, sys, csv
import concurrent.futures
import numpy as np

from pps_qj.parallel.grid_pps import task_params_clone_v2
from pps_qj.parallel.worker_clone_pps import _run_one_realisation

N_REAL    = 5
PROD_DIR  = os.environ["PROD_DIR"]
BENCH_DIR = os.environ["BENCH_DIR"]

# ── Task selection ─────────────────────────────────────────────────────────────
# Six tasks spanning (L=32,48,64) × (λ=small,large).
# Same (λ=0.02) tasks at L=32 and L=48 allow the L^3 scaling check.
# All have completed in production — wall_time is in their .npz.
#
#  tid   L   lam   zeta   N_c    T    regime
#  720   32  0.02  0.02   500   200   fast
#  743   32  0.90  0.02   500   200   slow
#  960   48  0.02  0.02   300   200   fast
#  983   48  0.90  0.02   300   200   slow
# 1200   64  0.02  0.02   200   150   fast
# 1223   64  0.90  0.02   200   150   slow
TASK_IDS = [720, 743, 960, 983, 1200, 1223]

# ── Load production wall times ─────────────────────────────────────────────────
prod_wt = {}
for tid in TASK_IDS:
    path = f"{PROD_DIR}/clone_{tid:05d}.npz"
    if os.path.exists(path):
        try:
            prod_wt[tid] = float(np.load(path)["wall_time"])
        except Exception as e:
            print(f"  WARNING: could not load {path}: {e}", flush=True)
    else:
        print(f"  WARNING: production file not found: {path}", flush=True)

# ── Benchmark function (run in subprocess) ─────────────────────────────────────
def bench_one(tid):
    """Run N_REAL realisations for task tid and return (tid, wall_time_s)."""
    import time as _time
    from pps_qj.parallel.grid_pps import task_params_clone_v2 as _params
    from pps_qj.parallel.worker_clone_pps import _run_one_realisation as _run
    task = _params(tid)
    L, w, alpha = task["L"], task["w"], task["alpha"]
    zeta, T, N_c, seed = task["zeta"], task["T"], task["N_c"], task["seed"]
    t0 = _time.perf_counter()
    for r in range(5):   # N_REAL=5
        _run(dict(L=L, w=w, alpha=alpha, zeta=zeta, T=T, N_c=N_c,
                  seed=seed + r * 999_983))
    return tid, _time.perf_counter() - t0

# ── Run all 6 tasks in parallel ────────────────────────────────────────────────
print(f"Launching {len(TASK_IDS)} tasks in parallel...", flush=True)
bench_times = {}
# spawn context avoids OpenBLAS fork deadlock (same as production)
ctx = concurrent.futures.ProcessPoolExecutor(
    max_workers=len(TASK_IDS),
    mp_context=__import__("multiprocessing").get_context("forkserver"),
)
with ctx as pool:
    futures = {pool.submit(bench_one, tid): tid for tid in TASK_IDS}
    for fut in concurrent.futures.as_completed(futures):
        tid, t = fut.result()
        bench_times[tid] = t
        print(f"  task {tid:5d} done: {t:.1f}s", flush=True)

# ── Collect and print results ──────────────────────────────────────────────────
print()
print(f"{'tid':>5}  {'L':>4}  {'lam':>6}  {'zeta':>5}  {'N_c':>5}  {'T':>5}"
      f"  {'t_prod':>8}  {'t_bench':>8}  {'ratio':>7}  status")
print("-" * 77)

results = []
for tid in TASK_IDS:
    task   = task_params_clone_v2(tid)
    t_prod = prod_wt.get(tid, float("nan"))
    t_bench = bench_times.get(tid, float("nan"))
    ratio  = t_bench / t_prod if (not np.isnan(t_prod) and not np.isnan(t_bench)) else float("nan")

    if   np.isnan(ratio):    status = "NO_PROD"
    elif 0.70 < ratio < 1.40: status = "OK"
    elif ratio >= 1.40:       status = "BENCH_SLOWER"
    else:                     status = "BENCH_FASTER"

    row = dict(tid=tid, L=task["L"], lam=task["lam"], zeta=task["zeta"],
               N_c=task["N_c"], T=task["T"],
               t_prod=t_prod, t_bench=t_bench, ratio=ratio, status=status)
    results.append(row)
    print(f"{tid:>5}  {task['L']:>4}  {task['lam']:>6.3f}  {task['zeta']:>5.2f}"
          f"  {task['N_c']:>5}  {task['T']:>5.0f}"
          f"  {t_prod:>8.1f}  {t_bench:>8.1f}  {ratio:>7.3f}  {status}")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
valid = [r for r in results if r["status"] != "NO_PROD"]
if valid:
    ratios = [r["ratio"] for r in valid]
    print(f"Hardware ratio (t_bench / t_prod):")
    print(f"  mean={np.mean(ratios):.3f}  std={np.std(ratios):.3f}"
          f"  range=[{min(ratios):.3f}, {max(ratios):.3f}]")
    print()

    by_L = {}
    for r in valid:
        by_L.setdefault(r["L"], []).append(r["ratio"])
    print("Ratio by L (scaling consistency check):")
    for Lv, rlist in sorted(by_L.items()):
        print(f"  L={Lv:3d}: {[f'{x:.3f}' for x in rlist]}  mean={np.mean(rlist):.3f}")
    print()

    # L^3 scaling check: (L=32→L=48) at same lambda=0.02
    r720  = next((r for r in results if r["tid"]==720),  None)
    r960  = next((r for r in results if r["tid"]==960),  None)
    r1200 = next((r for r in results if r["tid"]==1200), None)
    if r720 and r960:
        obs_32_48 = r960["t_prod"] / r720["t_prod"]
        pred_L3   = (300*200*48**3) / (500*200*32**3)
        pred_L4   = (300*200*48**4) / (500*200*32**4)
        print(f"L^3 scaling check (L=32→L=48, lambda=0.02, production times):")
        print(f"  Observed ratio:    {obs_32_48:.3f}")
        print(f"  L^3 prediction:    {pred_L3:.3f}  (err={abs(pred_L3-obs_32_48)/obs_32_48*100:.1f}%)")
        print(f"  L^4 prediction:    {pred_L4:.3f}  (err={abs(pred_L4-obs_32_48)/obs_32_48*100:.1f}%)")
        print()
    if r720 and r1200:
        obs_32_64 = r1200["t_prod"] / r720["t_prod"]
        pred_L3   = (200*150*64**3) / (500*200*32**3)
        pred_L4   = (200*150*64**4) / (500*200*32**4)
        print(f"L^3 scaling check (L=32→L=64, lambda=0.02, production times):")
        print(f"  Observed ratio:    {obs_32_64:.3f}")
        print(f"  L^3 prediction:    {pred_L3:.3f}  (err={abs(pred_L3-obs_32_64)/obs_32_64*100:.1f}%)")
        print(f"  L^4 prediction:    {pred_L4:.3f}  (err={abs(pred_L4-obs_32_64)/obs_32_64*100:.1f}%)")
        print()

    # Revised L=128 estimate using L^3
    t64_by_zeta = {
        0.02:4379, 0.05:4411, 0.10:4459, 0.15:4488, 0.20:4538,
        0.30:4666, 0.50:5281, 0.70:6275, 0.85:7430, 1.00:9699,
    }
    r128_64_L3 = (100*100*128**3) / (200*150*64**3)
    t128_total = sum(t64_by_zeta[z] * r128_64_L3 for z in t64_by_zeta)
    wall_h = int(t128_total/3600) + 8
    print(f"Revised L=128 estimate (L^3 scaling, 24 workers):")
    print(f"  {t128_total/3600:.1f}h total")
    print(f"  Recommended: sbatch --time={wall_h}:00:00")

# ── Write CSV ──────────────────────────────────────────────────────────────────
out_path = f"{BENCH_DIR}/benchmark_results.txt"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
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
