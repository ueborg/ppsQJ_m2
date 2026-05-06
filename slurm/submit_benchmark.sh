#!/bin/bash
# =============================================================================
# submit_benchmark.sh  —  Runtime benchmark vs production wall times
#
# Reruns a small set of completed production task_ids in a throwaway directory
# and compares measured times to the wall_time stored in the production .npz.
#
# Parallelism: xargs -P 6 launches 6 independent Python processes simultaneously.
# This is identical to how production jobs run and is known to work on Habrok.
# No multiprocessing module is used (forkserver fails silently in SLURM).
#
# Two things are tested:
#   1. Hardware match:  t_bench / t_prod ≈ 1 means compute node == omni40
#   2. Scaling model:  work ∝ N_c·T·L^3 (confirmed from partial results)
#
# Output:  $LOG_DIR/benchmark_<jobid>.out
# Results: $BENCH_DIR/benchmark_results.txt  (CSV, one row per task)
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
echo "Node: \$(hostname)   Started: \$(date)"
echo "Tasks: 720 743 960 983 1200 1223  (L=32,48,64 × fast/slow)"
echo "=================================================================="

module purge
source \${HOME}/venvs/pps_qj/bin/activate
cd \${HOME}/pps_qj

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p ${BENCH_DIR}

# ── Run 6 benchmark tasks in parallel via xargs ──────────────────────────────
# Each task_id gets its own independent Python process.
# Result written to ${BENCH_DIR}/result_XXXXX.json per task.
echo "720 743 960 983 1200 1223" | tr ' ' '\n' | \
xargs -P 6 -I{} python3 - {} ${BENCH_DIR} ${PROD_DIR} <<'PYSCRIPT'
import sys, time, json, os
import numpy as np

tid      = int(sys.argv[1])
bench_dir = sys.argv[2]
prod_dir  = sys.argv[3]

from pps_qj.parallel.grid_pps import task_params_clone_v2
from pps_qj.parallel.worker_clone_pps import _run_one_realisation

task  = task_params_clone_v2(tid)
L, w, alpha = task["L"], task["w"], task["alpha"]
zeta, T, N_c, seed = task["zeta"], task["T"], task["N_c"], task["seed"]

# Load production wall time
prod_file = f"{prod_dir}/clone_{tid:05d}.npz"
t_prod = float(np.load(prod_file)["wall_time"]) if os.path.exists(prod_file) else float("nan")

# Run benchmark
t0 = time.perf_counter()
for r in range(5):
    _run_one_realisation(dict(
        L=L, w=w, alpha=alpha, zeta=zeta, T=T, N_c=N_c,
        seed=seed + r * 999_983,
    ))
t_bench = time.perf_counter() - t0

ratio  = t_bench / t_prod if not (t_prod != t_prod) else float("nan")  # nan check
result = dict(
    tid=tid, L=L, lam=task["lam"], zeta=zeta, N_c=N_c, T=T,
    t_prod=t_prod, t_bench=t_bench, ratio=ratio,
)
out = f"{bench_dir}/result_{tid:05d}.json"
with open(out, "w") as f:
    json.dump(result, f)
print(f"task {tid:5d}: L={L:3d} lam={task['lam']:.3f}  "
      f"t_prod={t_prod:.1f}s  t_bench={t_bench:.1f}s  ratio={ratio:.3f}",
      flush=True)
PYSCRIPT

# ── Aggregate results ─────────────────────────────────────────────────────────
python3 - ${BENCH_DIR} ${PROD_DIR} <<'PYEOF'
import sys, json, glob, csv
import numpy as np

bench_dir = sys.argv[1]
prod_dir  = sys.argv[2]

files   = sorted(glob.glob(f"{bench_dir}/result_?????.json"))
results = [json.load(open(f)) for f in files]
results.sort(key=lambda r: r["tid"])

print()
print(f"{'tid':>5}  {'L':>4}  {'lam':>6}  {'zeta':>5}  {'N_c':>5}  {'T':>5}"
      f"  {'t_prod':>8}  {'t_bench':>8}  {'ratio':>7}  status")
print("-" * 77)

for r in results:
    ratio = r["ratio"]
    if   ratio != ratio:          status = "NO_PROD"       # nan
    elif 0.70 < ratio < 1.40:     status = "OK"
    elif ratio >= 1.40:           status = "BENCH_SLOWER"
    else:                         status = "BENCH_FASTER"
    r["status"] = status
    print(f"{r['tid']:>5}  {r['L']:>4}  {r['lam']:>6.3f}  {r['zeta']:>5.2f}"
          f"  {r['N_c']:>5}  {r['T']:>5.0f}"
          f"  {r['t_prod']:>8.1f}  {r['t_bench']:>8.1f}  {ratio:>7.3f}  {status}")

valid  = [r for r in results if r["ratio"] == r["ratio"]]   # not nan
ratios = [r["ratio"] for r in valid]
print()
print(f"Hardware ratio (t_bench / t_prod):")
print(f"  mean={np.mean(ratios):.3f}  std={np.std(ratios):.3f}"
      f"  range=[{min(ratios):.3f}, {max(ratios):.3f}]")

# L-scaling check on production times (independent of hardware)
by_lam = {}
for r in results:
    by_lam.setdefault(r["lam"], []).append(r)

print()
print("L^3 scaling check (production times, lambda=0.02):")
pts = sorted([r for r in results if abs(r["lam"]-0.02)<0.001], key=lambda r: r["L"])
for i in range(len(pts)-1):
    r1, r2 = pts[i], pts[i+1]
    obs  = r2["t_prod"] / r1["t_prod"]
    pred = (r2["N_c"] * r2["T"] * r2["L"]**3) / (r1["N_c"] * r1["T"] * r1["L"]**3)
    print(f"  L={r1['L']}→L={r2['L']}: observed={obs:.3f}  "
          f"L^3 pred={pred:.3f}  err={abs(pred-obs)/obs*100:.1f}%")

# Revised L=128 estimate
r128_64 = (100*100*128**3) / (200*150*64**3)
t64_by_zeta = {
    0.02:4379, 0.05:4411, 0.10:4459, 0.15:4488, 0.20:4538,
    0.30:4666, 0.50:5281, 0.70:6275, 0.85:7430, 1.00:9699,
}
t128_h = sum(t64_by_zeta[z]*r128_64 for z in t64_by_zeta) / 3600
print()
print(f"Revised L=128 estimate (L^3, 24 workers): {t128_h:.1f}h")
print(f"Recommended: sbatch --time={int(t128_h)+8}:00:00")

# Write CSV
out = f"{bench_dir}/benchmark_results.txt"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print(f"\nResults: {out}")
PYEOF

echo "=================================================================="
echo "Benchmark complete: \$(date)"
echo "=================================================================="
SLURM_SCRIPT

echo "Submitted benchmark job."
echo "Monitor: squeue -u \$USER -n pps_benchmark"
echo "Output:  tail -f ${LOG_DIR}/benchmark_<jobid>.out"
