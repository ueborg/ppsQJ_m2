#!/bin/bash
# =============================================================================
# submit_benchmark.sh  —  Runtime benchmark vs production wall times
#
# Reruns 6 completed production task_ids and compares wall times to production.
#
# Tests:
#   1. Hardware match:  t_bench / t_prod should be consistent (~0.74 on omni40)
#   2. L^3 scaling:     (32→48→64) ratios should match N_c·T·L^3 model
#
# Parallelism: xargs -P 6 — identical pattern to all production jobs.
# Worker:      python -m pps_qj.parallel.worker_bench_pps (proper module file,
#              no here-doc tricks that conflict with stdin redirection).
#
# Usage:
#   bash slurm/submit_benchmark.sh
#
# After job completes:
#   cat /scratch/$USER/pps_qj/benchmark_*/benchmark_results.txt
# =============================================================================
set -euo pipefail

PARTITION="regular"
WALL_TIME="02:00:00"
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
echo "Tasks: 720 743 960 983 1200 1223  (L=32,48,64 x fast/slow)"
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

# ── Run 6 tasks in parallel ───────────────────────────────────────────────────
echo "720 743 960 983 1200 1223" | tr ' ' '\n' | \
xargs -P 6 -I{} python -m pps_qj.parallel.worker_bench_pps \
    {} ${BENCH_DIR} ${PROD_DIR}

# ── Aggregate results ─────────────────────────────────────────────────────────
python -c "
import json, glob, csv, sys
import numpy as np

bench_dir = '${BENCH_DIR}'
files     = sorted(glob.glob(f'{bench_dir}/result_?????.json'))

if not files:
    print('ERROR: no result files found in', bench_dir)
    sys.exit(1)

results = [json.load(open(f)) for f in files]
results.sort(key=lambda r: r['tid'])

print()
print(f\"{'tid':>5}  {'L':>4}  {'lam':>6}  {'zeta':>5}  {'N_c':>5}  {'T':>5}\"
      f\"  {'t_prod':>8}  {'t_bench':>8}  {'ratio':>7}  status\")
print('-' * 77)

for r in results:
    ratio = r['ratio']
    if   ratio != ratio:          status = 'NO_PROD'
    elif 0.70 < ratio < 1.40:     status = 'OK'
    elif ratio >= 1.40:           status = 'BENCH_SLOWER'
    else:                         status = 'BENCH_FASTER'
    r['status'] = status
    print(f\"{r['tid']:>5}  {r['L']:>4}  {r['lam']:>6.3f}  {r['zeta']:>5.2f}\"
          f\"  {r['N_c']:>5}  {r['T']:>5}\"
          f\"  {r['t_prod']:>8.1f}  {r['t_bench']:>8.1f}  {ratio:>7.3f}  {status}\")

valid  = [r for r in results if r['ratio'] == r['ratio']]
ratios = [r['ratio'] for r in valid]
print()
print(f'Hardware ratio summary (t_bench / t_prod):')
print(f'  mean={np.mean(ratios):.3f}  std={np.std(ratios):.3f}'
      f'  range=[{min(ratios):.3f}, {max(ratios):.3f}]')

# L^3 scaling check using production times
print()
print('L^3 scaling check (production times, lam=0.02):')
pts = sorted([r for r in results if abs(r['lam']-0.02)<0.001], key=lambda r: r['L'])
for i in range(len(pts)-1):
    r1, r2 = pts[i], pts[i+1]
    obs  = r2['t_prod'] / r1['t_prod']
    p3   = (r2['N_c']*r2['T']*r2['L']**3) / (r1['N_c']*r1['T']*r1['L']**3)
    p4   = (r2['N_c']*r2['T']*r2['L']**4) / (r1['N_c']*r1['T']*r1['L']**4)
    print(f\"  L={r1['L']}->L={r2['L']}: obs={obs:.3f}  L^3={p3:.3f} (err={abs(p3-obs)/obs*100:.0f}%)  L^4={p4:.3f} (err={abs(p4-obs)/obs*100:.0f}%)\")

# L=128 estimate
r128 = (100*100*128**3) / (200*150*64**3)
t64  = {0.02:4379,0.05:4411,0.10:4459,0.15:4488,0.20:4538,
        0.30:4666,0.50:5281,0.70:6275,0.85:7430,1.00:9699}
t128_h = sum(t64[z]*r128 for z in t64) / 3600
print()
print(f'L=128 estimate (L^3, 24 workers): {t128_h:.1f}h -> sbatch --time={int(t128_h)+8}:00:00')

out = f'{bench_dir}/benchmark_results.txt'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
print(f'Results: {out}')
"

echo "=================================================================="
echo "Benchmark complete: \$(date)"
echo "=================================================================="
SLURM_SCRIPT

echo "Submitted benchmark job."
echo "Monitor: squeue -u \$USER -n pps_benchmark"
echo "Output:  tail -f ${LOG_DIR}/benchmark_<jobid>.out"
