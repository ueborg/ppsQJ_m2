# Running ppsQJ_m2 on Ruche (Paris-Saclay Mesocentre)

Cascade Lake CPU nodes (40 cores each). Campaign = independent cloning
realizations, 1 core each (BLAS pinned to 1 thread); parallelism is across
realizations via Slurm arrays. Code path: lowrank jump update + Newton solver
(~2.8x over the original at the cloning level).

## Filesystem
- Code + conda env -> `$HOME` (50 GB) and `$WORKDIR` (500 GB). `$WORKDIR` =
  `/workdir/$USER` = `/gpfs/workdir/$USER`. No backup on either.
- Results -> `$WORKDIR/pps/<grid>`.

## One-time setup (login node)
```bash
git clone https://github.com/ueborg/ppsQJ_m2.git "$HOME/ppsQJ_m2"
cd "$HOME/ppsQJ_m2"
bash scripts/ruche/setup_ruche.sh          # conda env (numpy2+scipy, MKL) + smoke test
sbatch scripts/ruche/calib_ruche.sh        # 1-core calibration -> prints r = Ruche/Mac
```
The calibration fixes `r`; multiply the planning numbers below by it.
Mac baseline: one L=128 realization = 25.5 min.

## Per-realization cost (r=1, ~L^5 scaling)
L=64 ~1.6m | L=96 ~8.5m | L=128 ~25-32m | L=160 ~2.2h | L=192 ~5.4h |
L=256 ~20-24h. Reduce T ONLY for the boundary tier (values saturate ~0.5L);
the nu tier needs T>=2L (slopes) -- see HANDOFF T-cap finding.

## Submitting (size-binned arrays)
Each array task owns a disjoint round-robin shard (`--shard`/`--nshards`).
Per-realization JSON checkpoints make tasks resumable/idempotent. Keep
`--array=...%K` so `K*40 <= partition core cap` (cpu_med 1000 -> K<=25,
cpu_long 160 -> K<=4, cpu_prod 2000 -> K<=50). See the header of
`submit_pps_boundary.sh` for ready-to-edit Tier 1/2/3 examples, e.g.:
```bash
sbatch -p cpu_med --time=04:00:00 --array=0-39%25 \
  --export=ALL,OUTDIR=$WORKDIR/pps/boundary,LS="64 96 128",\
ZETAS="0.10 0.20 0.30 0.50 1.00",LAM_MULTS="0.7 0.85 1.0 1.15 1.3",NREAL=12,NC=128 \
  scripts/ruche/submit_pps_boundary.sh
```

## Aggregating
```bash
python scripts/run_local_boundary.py aggregate --outdir $WORKDIR/pps/boundary
```
Writes `aggregate.csv` + B_L crossings (lambda_x, lambda_x/zeta vs L*zeta^2).
