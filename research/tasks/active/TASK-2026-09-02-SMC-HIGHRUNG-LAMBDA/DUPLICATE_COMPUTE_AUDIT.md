# Duplicate-compute and pooling audit

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief §7.

## What was scanned

Every source of completed Cut-B runs reachable from this repository, keyed on
`(L, T, zeta, lambda, N_c, seed, dtau_mult, resample_scheme)`:

| source | runs / rows | how read |
|---|---:|---|
| `TASK-2026-08-30-SMCSTAT/scratch/*.jsonl` (11 blocks) | 1,284 | full parse |
| `TASK-2026-08-30-SMCSTAT/ruche_package/manifest.csv` | 432 | seeds |
| `TASK-2026-08-30-GENCOL/ruche_package/manifest.csv` | 96 | seeds |
| `TASK-2026-08-31-SMCCERT/ruche_package/manifest.csv` | 304 | seeds |
| `TASK-2026-09-01-SMCRUCHE-READY/arm{1,2}/manifest.csv` | 304 | seeds |
| `TASK-2026-09-01-SMCRUCHE-READY/arm{1,2}/results/*.json` | 304 | full parse |
| the historical Cut-B corpus `pps_all_realizations.csv` | 20,355 | full parse |

2,116 distinct seeds exist across the task tree; the historical corpus records
no seed at all.

## Result: nothing is repeated

### The two neighbouring lambdas have never been run

`lambda = 0.2932` and `lambda = 0.3132` return **zero rows** anywhere — not in
the corpus, not in any task, at any L, any zeta, any `N_c`. The corpus's
lambda grid at `zeta = 0.35` is
`{0.1775, 0.2219, 0.2366, 0.2479, 0.2588, 0.2662, 0.2701, 0.281, 0.2923, 0.3032,
0.3106, 0.3144, 0.3254, 0.3366, 0.3476, 0.355, 0.3588, 0.3698, 0.4141, 0.4733}`
and the new lambdas fall between existing points without coinciding with any.

### The new N_c rungs have never been run

At `L = 128, T = 128, zeta = 0.35, lambda = 0.3032` the programme has
`N_c ∈ {64, 128, 256}` (ARM2, `R = 64` each) and nothing above. `N_c = 512` and
`N_c = 1024` at L = 128 do not exist. At `L = 64, T = 64` there is no cloning
run at any `N_c` in this programme at all.

### The historical corpus is not poolable with anything here

The 36 corpus rows at `zeta = 0.35, lambda = 0.3032, L ∈ {64, 96, 128}` are
excluded from pooling on **three independent grounds**, any one of which would
be sufficient:

1. **Different discretisation.** Every corpus row has `dtau = 12.0 / (2 λ (L−1))`,
   i.e. `dtau_mult = 12.0`. This campaign runs the certified `dtau_mult = 6.0`.
   `support/instrumented.py` carries the matching in-code warning: *"dtau_mult
   defaults to the CERTIFIED production value 6, not the corpus value 12. GENCOL's
   copy of this file defaulted to 12; a default that silently deviates from the
   certified baseline is a trap."* These are different discretisations of the same
   physics and their population means are not interchangeable.
2. **Different population size.** Every corpus row is `N_c = 128`, which is not a
   rung in this campaign.
3. **No recoverable seed.** The corpus has no `seed` column. Independence from
   the new streams cannot be established, so even at a matched cell the runs
   could not be pooled without risking correlated replicates.

`shared/preflight.py` enforces (1) and refuses any manifest carrying
`dtau_mult = 12.0`.

## What IS pooled, and with what provenance

Only completed runs from **this** programme, at **exactly matched**
`(L, T, zeta, lambda, dtau_mult, resample_scheme)` and with disjoint seeds:

| pooled block | cell | N_c, R | role |
|---|---|---|---|
| ARM2 results (192 runs) | L=128, T=128, λ=0.3032 | 64, 128, 256 × R=64 | the existing ladder that `Delta_256->512` is measured against |
| ARM1 results (112 runs) | L=96, T=96, λ=0.3032 | 128, 256, 512 | context only; not used in any F1–F7 verdict |
| SMCSTAT `A-P96`, `A-BUD` (128 runs) | L=96, T=96, λ=0.3032 | 32, 64 | context only |

These 528 populations are frozen into
`frozen_inputs/predecessor_populations.csv` with a per-file `sha256`
(`INPUTS_LEDGER.md`). That snapshot reproduces the published ARM1 and ARM2 final
analyses **digit for digit** — see `VALIDATION.md`. The predecessor task
archives were read and not modified.

## The one deliberate reuse inside this campaign

`armC`'s three-point stencil at L = 128 needs a central point at
`lambda = 0.3032, N_c = 512`. That is exactly `armA512`. It is **not**
recomputed: `armC/manifest.csv` contains only `lambda ∈ {0.2932, 0.3132}` (96
rows, not 144), and `analysis/combined_analysis.py` assembles the stencil by
looking the central point up across arms. This saves 48 tasks and 241
core-hours.

`shared/preflight.py` checks `R equal across lambdas` per arm, so the reuse
cannot silently become an unequal-`R` stencil.

## Seeds

All new seeds are drawn from `[30_000_000, 31_000_000)`. The largest seed
anywhere in the existing tree is **20,384,063**, so disjointness is structural
rather than merely checked. `SEED_LEDGER.md` records the allocation; the
preflight verifies both the block bound and, redundantly, a direct set
intersection against `tools/existing_seeds.json`.
