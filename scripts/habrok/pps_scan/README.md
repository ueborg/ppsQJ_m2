# PPS phase-diagram scan on Habrok

Map the left-edge measurement-induced phase transition
(γ=0, α+w=1, λ = α/(α+w)) as a function of the partial post-selection
parameter ζ ∈ [0.1, 1.0] using the Gaussian Doob-WTMC sampler, with a
cloning cross-validation at small L.

## Grid sizes

| Scan      | L values                                       | λ points | ζ points | Total |
| --------- | ---------------------------------------------- | -------- | -------- | ----- |
| Doob-WTMC | 16, 24, 32, 48, 64, 96, 128, 192, 256 (9)      | 17       | 10       | 1530  |
| Cloning   | 8, 12, 16 (3)                                  | 17       |  4       |  204  |

Task IDs are contiguous per L (L outer, λ middle, ζ inner). L-ranges for the
Doob scan:

```
L=16  tasks    0.. 169   L= 64 tasks  680.. 849   L=128 tasks 1020..1189
L=24  tasks  170.. 339   L= 96 tasks  850..1019   L=192 tasks 1190..1359
L=32  tasks  340.. 509                            L=256 tasks 1360..1529
L=48  tasks  510.. 679
```

## One-time setup

```bash
cd $HOME/pps_qj/scripts/habrok/pps_scan
bash setup_venv_pps.sh
# Verify grid counts:
module load Python/3.10.8-GCCcore-12.2.0 SciPy-bundle/2023.02-gfbf-2022b
source $HOME/venvs/pps_qj/bin/activate
python -m pps_qj.parallel.grid_pps doob    # must print 1530
python -m pps_qj.parallel.grid_pps clone   # must print 204
```

## Submission sequence

```bash
cd $HOME/pps_qj/scripts/habrok/pps_scan

# Submit all array jobs (capture job IDs for the dependency):
SMALL=$(sbatch --parsable submit_doob_small.sh)   # L in {16,24,32,48}
MEDIUM=$(sbatch --parsable submit_doob_medium.sh) # L in {64,96}
LARGE=$(sbatch --parsable submit_doob_large.sh)   # L in {128,192,256}
CLONE=$(sbatch --parsable submit_clone_pps.sh)
echo "small=$SMALL medium=$MEDIUM large=$LARGE clone=$CLONE"

# Submit aggregation + figures with dependency:
sbatch --dependency=afterany:${SMALL}:${MEDIUM}:${LARGE}:${CLONE} \
       submit_analysis.sh
```

## Monitoring progress (partial figures any time)

```bash
module load Python/3.10.8-GCCcore-12.2.0 SciPy-bundle/2023.02-gfbf-2022b
source $HOME/venvs/pps_qj/bin/activate
cd $HOME/pps_qj
python analysis/monitor_and_plot.py /scratch/$USER/pps_qj/pps_doob_scan \
    --output-figures /scratch/$USER/pps_qj/figures_partial
```

Produces a completion heatmap + partial versions of all final figures. Safe
to re-run at any time; does not interfere with running jobs (reads
`summary_*.json`, which are written atomically).

## Walltime estimates (per array task)

| Job group      | L        | N_traj | Typical T | Walltime limit | Expected use  |
| -------------- | -------- | ------ | --------- | -------------- | ------------- |
| doob_small     | 16–48    | 1000   | 30–96     | 1 h            | 10–50 min     |
| doob_medium    | 64–96    |  500   | 128–192   | 4 h            | 30 min – 3 h  |
| doob_large     | 128–256  | 100–200| 256–512   | 8 h            | 2–7 h         |
| clone          | 8–16     | —      | 30–32     | 2 h            | 1–20 min      |
| analysis       | —        | —      | —         | 30 min         | < 5 min       |

The bisection-based WTMC sampler dominates time at large L. If L=256 jobs
time out, edit `n_traj_for_L` in `pps_qj/parallel/grid_pps.py` to drop L=256
n_traj to 50 and use `resubmit_failed.sh` to resubmit only the missing
ids.

## Resubmit missing tasks

```bash
bash resubmit_failed.sh doob  /scratch/$USER/pps_qj/pps_doob_scan  submit_doob_small.sh
bash resubmit_failed.sh clone /scratch/$USER/pps_qj/pps_clone_scan submit_clone_pps.sh
```

The first argument picks the scan type; the aggregator emits a SLURM
range spec like `0-3,17,42-58` which is passed straight to
`sbatch --array=...`.

## After completion — copy to $HOME

**Scratch is not backed up and may be purged.** Immediately after the
analysis job finishes:

```bash
rsync -av /scratch/$USER/pps_qj/ $HOME/pps_qj_results/
```

`$HOME` has quota (check `quota -s`) but IS backed up.

## Storage footprint

| Component                            | # files | Size each     | Total     |
| ------------------------------------ | ------- | ------------- | --------- |
| `doob_*.npz` (trajectory results)    |  1530   | ~15 KB        |  ~23 MB   |
| `summary_*.json` (monitoring)        |  1530   | ~1 KB         |   ~2 MB   |
| `clone_*.npz` (cloning results)      |   204   | ~10 KB        |   ~2 MB   |
| Backward passes (selective, full)    |    60   | ~1 MB–100 MB  |   ~3 GB   |
| SLURM logs (out+err)                 |  3468   | ~5 KB         |  ~17 MB   |
| **TOTAL**                            |         |               | **~3.1 GB**|

Full backward passes are saved for a representative subset of 60 points
(L ∈ {16, 32, 64, 128, 256}, λ ∈ {0.3, 0.5, 0.7}, ζ ∈ {0.3, 0.5, 0.7, 1.0}).
For any other (L, λ, ζ), rerun the worker with `--save-bwd`:

```bash
python -m pps_qj.parallel.worker_doob_pps <task_id> <output_dir> 1 --save-bwd
```

## Loading a backward pass locally

```python
from pps_qj.backward_pass_io import load_backward_pass
bwd = load_backward_pass("bwd_L064_lam0.5000_zeta0.500_T128.0.npz")
C_t, z_t = bwd.state_at(t=10.0)
print(f"Z_T = {bwd.Z_T:.6f}, theta = {bwd.theta_doob:.6f}")
```

The loaded object is a drop-in replacement for `GaussianBackwardData` in
`doob_gaussian_trajectory` (same `state_at(t)` interface).

---

## Slope-test run (added 2026-05-26)

**Goal**: discriminate Möbius slope 1/8 from naive NLSM slope 1/4 at ζ=1.
Requires λ_c(ζ) at ζ ∈ {0.7,0.8,0.9} to ~0.5% precision.

### Phase 1 — fill L≤128 gaps (fast, cheap)

ζ ∈ {0.80, 0.90} are missing from the cloning aggregate at all L.
Add them with the existing nc_for_L schedule.

Grid: L ∈ {32,48,64,96,128} × ζ ∈ {0.80,0.90} × 24 λ-points
= 240 tasks, <2h each. Cost: ~480 CPU-hours.

Purpose: (a) rough λ_c estimates at ζ=0.80,0.90; (b) narrows the λ-scan
for Phase 2 from 24 to ~12 points (halves Phase 2 cost).

### Phase 2 — high-L precision (necessary for slope test)

Grid: L ∈ {192,256} × ζ ∈ {0.70,0.80,0.90} × 24 λ-points × N_c=200
= 144 tasks, ~14h each. Cost: ~2016 CPU-hours (~2 days cluster time).

ζ=1.00 at L≫128 is anchored analytically (Carollo λ_c=1/2); do NOT rerun.
ζ=0.50 at L=192,256 already exists from Run B.

Expected discriminating power:
  λ_c(0.7): Möbius=0.456, NLSM=0.412, diff=0.044
  At N_c=200, L=256: σ(λ_c) ≈ 0.005 → 8σ discrimination if slope~1/4.

