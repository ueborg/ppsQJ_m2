# Handoff prompt — fine-grid cloning campaign with multiple observables

## Project context

You are working on `ueborg/ppsQJ_m2`, a 1D Kitaev chain MIPT project under
quantum jump (QJ) dynamics with partial post-selection (PPS) parameter ζ.
Codebase at `/Users/catlover1337/Documents/ppsQJ_m2/`. Habrok user `s4629701`.

**Read `theory/HANDOFF.md` first** for full project state.

## What this campaign is and why

The existing aggregate at L ≤ 128 (`~/Downloads/clone_aggregate(1).pkl`,
1920 entries) uses a uniform 24-point λ grid in [0.02, 0.9] and a fixed
10-point ζ grid. For finding λ_c(ζ) cleanly via Binder/CMI crossings, this
is too coarse: at small ζ where λ_c ≈ 0.10–0.15, only ~5 λ-points sit below
the crossing, making interpolation unreliable. At large ζ where N_c = 100
at L = 128 is borderline, the Binder cumulant has ~20–30% relative error in
the critical band.

Additionally, the worker computes B_L = CMI × S_AB as a single quantity, so
**the bare CMI = S_AB + S_BC − S_B − S_ABC (without the S_AB multiplier)
is not recoverable from the aggregate**, even though CMI alone is the
cleaner Binder-like observable per Li–Chen–Fisher PRB 100, 134306 (2019).
Renyi-2 and Renyi-3 entropies are computed per task and saved to .npz files
but are **dropped by the current aggregator** (`scripts/aggregate.py` or
`scripts/aggregate_runs.py` — check which is current).

## Plan in three parts

### Part 0 — Code changes (do these locally on Mac before submitting anything)

These changes are prerequisite. Estimated effort: 1–2 hours of careful work.

**0a. Modify `pps_qj/parallel/worker_clone_pps.py`**

The function `_batched_compute_B_L` currently returns only B_L. Modify it to
return a dict `{S_AB, S_BC, S_B, S_ABC, CMI, B_L}` per clone, where
CMI = S_AB + S_BC − S_B − S_ABC. Then in the realisation loop, store and
aggregate all six quantities per realisation. The .npz save call should
gain new fields: `CMI_mean`, `CMI_err`, `S_AB_mean`, `S_AB_err`, etc.

Reference: current implementation lives around line 130 of
`worker_clone_pps.py`. The four sub-covariance entropies (S_AB, S_BC, S_B,
S_ABC) are already computed there — just don't multiply them together yet,
return them separately.

**0b. Fix the aggregator (`scripts/aggregate.py` or `aggregate_runs.py`)**

Add the following fields to the per-task entry dict:
- `CMI_mean`, `CMI_err`
- `S_AB_mean`, `S_AB_err`
- `S_renyi_2_mean`, `S_renyi_2_err` (these are in .npz but not aggregated!)
- `S_renyi_3_mean`, `S_renyi_3_err`

Verify by checking the existing aggregate: `print(list(entry.keys()))`
should show these new fields after re-aggregation.

**0c. Add the fine-grid spec to `pps_qj/parallel/grid_pps.py`**

Three new grid functions, one per ζ region:

```python
# Small ζ — narrow λ range around small λ_c values
ZETA_VALS_SMALL = [0.02, 0.05, 0.08, 0.10, 0.15]
LAMBDA_VALS_SMALL = np.concatenate([
    np.linspace(0.01, 0.06, 6),      # 6 dense low-λ points
    np.linspace(0.07, 0.30, 16),     # 16 in the crossing band
    np.linspace(0.35, 0.60, 4),      # 4 in area phase for crossing
])  # 26 λ-points

# Medium ζ — middle of phase diagram
ZETA_VALS_MEDIUM = [0.20, 0.30, 0.40, 0.50]
LAMBDA_VALS_MEDIUM = np.concatenate([
    np.linspace(0.10, 0.20, 4),
    np.linspace(0.22, 0.45, 16),     # crossing band
    np.linspace(0.48, 0.65, 4),
])  # 24 λ-points

# Large ζ — near Born endpoint
ZETA_VALS_LARGE = [0.65, 0.75, 0.85, 1.00]
LAMBDA_VALS_LARGE = np.concatenate([
    np.linspace(0.20, 0.35, 4),
    np.linspace(0.38, 0.55, 16),     # crossing band around λ_c~0.45-0.50
    np.linspace(0.58, 0.70, 4),
])  # 24 λ-points

L_DENSE_SMALL = [8, 16, 24, 32, 48, 64, 96, 128]
```

N_c schedule for the dense grid: bump N_c at L=128 from 100 to 200, and at
L=96 from 150 to 250, to reduce the relative-error problem. Keep N_c=200
at L=64 and below (already adequate based on existing data).

Seeds must be unique vs the v2 grid. Use `_seed(L, lam, zeta)` (the
existing helper) — new λ values automatically avoid collision.

Provide three task dispatch functions:
- `task_params_clone_dense_small(task_id)`
- `task_params_clone_dense_medium(task_id)`
- `task_params_clone_dense_large(task_id)`

And register them in `worker_clone_pps.py`'s `_GRID_DISPATCH` dict
alongside `v1` and `slope`:
```python
_GRID_DISPATCH = {
    "v1":           task_params_clone,
    "slope":        task_params_clone_slope,
    "dense_small":  task_params_clone_dense_small,
    "dense_medium": task_params_clone_dense_medium,
    "dense_large":  task_params_clone_dense_large,
}
```

### Part 1 — L ≤ 128 fine-grid (3 SLURM scripts, ~2 days on Habrok)

Actual wall times from the existing v2 runs (median across all λ at the
old N_c), in **seconds per task** (N_REAL=5, n_workers=1):

| L | N_c (v2) | ζ=0.05 median | ζ=0.50 median | ζ=1.00 median | worst-case (max) |
|---|---|---|---|---|---|
| 8 | 2000 | 385 | 435 | 707 | 1475 |
| 16 | 1000 | 596 | 719 | 1208 | 2568 |
| 24 | 800 | 1767 | 2120 | 3469 | 5857 |
| 32 | 500 | 3145 | 3737 | 6436 | 13456 |
| 48 | 300 | 9698 | 11750 | 19278 | **31085** ← anomalous |
| 64 | 200 | 4712 | 5936 | 10221 | 20967 |
| 96 | 150 | 12178 | 15541 | 26115 | **50507** |
| 128 | 100 | 24441 | 32859 | 53136 | **82151** (~23h) |

**Important observations:**
1. L=128 worst case at ζ=1, λ=0.75 took 22.8 hours. Walltime in the new
   SLURM scripts must reflect this.
2. L=48 is anomalously slow (probably cache/BLAS — non-power-of-2 size at
   N_c=300 is in a bad regime). Consider whether to keep L=48 or skip it
   in favor of L=40 or L=56.
3. Time scales linearly with N_c. Doubling N_c at L=128 to 200 means
   worst-case ~46h, which exceeds the typical Habrok walltime cap. **Use
   intra-task parallelism** (`PPS_N_WORKERS=5`, `--cpus-per-task=5`) to
   speed up by ~5x — but verify the parallel mode works correctly first.

**Suggested walltimes for each new SLURM script (with `n_workers=5`):**

| L | walltime | partition | mem |
|---|---|---|---|
| 8–32 | 4h | regular | 16GB |
| 48–96 | 12h | regular or regularsh | 24GB |
| 128 | 24h | regularme | 32GB |

Split into three SLURM scripts: `submit_clone_dense_small.sh` (L=8–32),
`submit_clone_dense_med.sh` (L=48–96), `submit_clone_dense_large.sh`
(L=128 only). Use `cpus-per-task=5` and `export PPS_N_WORKERS=5` for the
medium and large.

**Task count:**

| Region | n_ζ × n_λ × n_L | tasks |
|---|---|---|
| Small | 5 × 26 × 8 | 1040 |
| Medium | 4 × 24 × 8 | 768 |
| Large | 4 × 24 × 8 | 768 |
| **Total L ≤ 128** | | **2576** |

**Total CPU-hours estimate** (using median × N_c_factor × n_traj_factor):
N_c bumped from old values to (200 at L≤64, 250 at L=96, 200 at L=128):

Approximate total ≈ 8000 CPU-hours. With 96 concurrent and intra-task
n_workers=5 speedup, clock time on cluster ≈ 2–3 days.

### Part 2 — L = 192, 256 high-precision (after Part 1 reveals λ_c estimates)

Goal: tight λ-scan centred on the L=128 crossing point. After Part 1
analyses, for each ζ extract λ_c^{(128)} and run a 15-point λ grid
spanning [λ_c^{(128)} − 0.08, λ_c^{(128)} + 0.08] at L = 192 and L = 256.

Cost: 13 ζ × 15 λ × 2 L = 390 tasks.

L=192 wall time (estimate from existing FST runs, N_c=80, n_workers=5):
~5h/task. At N_c=200 with same parallelism: ~12h.
L=256 at N_c=100 with n_workers=5: ~15h.

Walltime 36h, partition regularme, 5 cpus, 32GB. Total clock time ~3 days.

### Part 3 — Analysis (after Parts 1 and 2 complete)

Per ζ, three independent extractions of λ_c:
1. B_L crossings (existing observable, baseline)
2. CMI crossings (new — should be cleaner per Li–Chen–Fisher)
3. Renyi-2 crossings (new — robust to noise)

If all three agree within errors → strong evidence for that λ_c value.
Cross-check ν between the three observables → constrains universality
class.

Then plot λ_c(ζ) globally and refit the matched-NLSM exponent φ. Compare
against:
- Current best estimate: φ = 0.56 ± 0.05 (from existing data)
- Möbius prediction: λ_c = √ζ / (1 + √ζ), slope at ζ=1 → 1/8
- Naive NLSM: λ_c = ζ/(1 + ζ), slope at ζ=1 → 1/4

## Key files and where things are

- `/Users/catlover1337/Documents/ppsQJ_m2/pps_qj/parallel/grid_pps.py` —
  add new grid functions here
- `/Users/catlover1337/Documents/ppsQJ_m2/pps_qj/parallel/worker_clone_pps.py` —
  modify `_batched_compute_B_L` and the np.savez call
- `/Users/catlover1337/Documents/ppsQJ_m2/scripts/aggregate.py` (or
  `aggregate_runs.py`) — add Renyi entropies, CMI, S_AB to aggregator
- `/Users/catlover1337/Documents/ppsQJ_m2/scripts/habrok/pps_scan/` —
  new SLURM scripts go here
- `/Users/catlover1337/Downloads/clone_aggregate(1).pkl` — existing
  aggregate (use to validate new code by re-aggregating a few existing
  tasks and comparing)

## Critical reminders

1. **Don't push from Habrok** (SSH key issue). All code changes on Mac
   → commit → push GitHub → pull on Habrok.
2. **Verify the per-clone parallel mode** in `_batched_compute_B_L` works
   correctly under `n_workers=5` BEFORE submitting 2576 tasks. Run one
   task locally with `PPS_N_WORKERS=5` and check the output .npz has
   sensible numbers.
3. **Test the aggregator on a small subset** (50 tasks) before running on
   the full new dataset.
4. **L=48 anomaly**: investigate before re-running. May be worth skipping
   in favor of L=40 or L=56, or fixing whatever causes the 3x slowdown vs
   neighboring L values.
5. **Existing data is not wasted.** The new dense grid should be merged
   with the v2 aggregate. The combined dataset will have the old coarse
   grid as a backbone and the new fine grid filling in the critical band.

## Expected outcome

After Parts 1 and 2: 11 ζ values with λ_c(ζ) at 0.5% precision from three
independent observables. Should resolve:
- Whether φ = 1/2 exactly or has corrections
- The slope at ζ = 1 (Möbius vs NLSM)
- The universality class via ν from CMI and Renyi-2

This is roughly the dataset needed to upgrade from a marginal PRL to a
strong submission (or to PRX if combined with Case A implementation later).
