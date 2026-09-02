# Reuse and duplicate-compute audit

TASK-2026-09-02-MOCK-PRODUCTION, brief §5.

## What was scanned

Every source of completed Cut-B runs reachable from this repository, keyed on
`(L, T, zeta, lambda, N_c, dtau_mult, resample_scheme)` and, where a seed
exists, on the seed:

| source | runs / rows | how read |
|---|---:|---|
| `TASK-2026-08-30-SMCSTAT/scratch/*.jsonl` (11 blocks) | 1,284 | full parse |
| `TASK-2026-08-30-SMCSTAT/ruche_package/manifest.csv` | 432 | seeds |
| `TASK-2026-08-30-GENCOL/ruche_package/manifest.csv` | 96 | seeds |
| `TASK-2026-08-31-SMCCERT/ruche_package/manifest.csv` | 304 | seeds |
| `TASK-2026-09-01-SMCRUCHE-READY/arm{1,2}` manifests + results | 304 | full parse |
| `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA` five manifests | 480 | seeds |
| `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/arm{A512,A1024,B}/results` | 368 | full parse |
| the historical Cut-B corpus `pps_all_realizations.csv` | 20,355 | full parse |

2,596 distinct seeds exist across the task tree, all `<= 30,500,015`. The
historical corpus records no seed at all.

## 1. The one reuse this campaign makes

**[E]** `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/armB` returned **288 completed
runs, all `status = ok`**, at exactly

```
L = 64, T = 64, zeta = 0.35, N_c = 1024, dtau_mult = 6.0,
resample_scheme = systematic,  lambda in {0.2932, 0.3032, 0.3132},
R = 96 independent populations per lambda,  seeds 30300000-30302095
```

Every one of the seven matched keys is identical to this campaign's `L = 64,
N_c = 1024` cells, and the sampler is provably the same file: both packages
bundle `instrumented.py` at sha256 `0a33c4034cda70ea…`
(`PRODUCTION_PATH_UNCHANGED.md`).

**These three cells are not recomputed.** `mockL64/manifest.csv` carries the
**ten** other grid lambdas, not thirteen — 240 rows rather than 312. The
analysis assembles the 13-point curve by reading the reused populations from
`frozen_inputs/armB_populations.csv`.

`shared/preflight.py` enforces this in both directions: it fails any
`(L=64, N_c=1024)` manifest that contains one of the three ARM-B lambdas
(`no ARM-B duplication`), and it fails any manifest whose lambdas are off the
frozen 13-point grid.

**Saved: 72 array tasks and ~39 core-hours**, and — more valuably — three grid
points that carry `R = 96` instead of `R = 24`, i.e. error bars half the size of
their neighbours', at zero cost.

The heteroscedasticity this creates is handled explicitly rather than ignored:
`analysis_spec.yaml` `heteroscedasticity_rule` requires every cross-lambda
statistic to use per-point standard errors, because a roughness or chi-square
statistic assuming equal errors would read the three precise points as anomalies
of the *curve* rather than of the *design*.

## 2. The search for any other compatible cell — result: none

**[E]** Searched across all eight sources above for any run matching this
campaign's `(zeta = 0.35, dtau_mult = 6.0, systematic)` at
`L in {32, 48, 64}` and `N_c in {128, 1024, 2048}`:

| cell class | found | usable |
|---|---|---|
| `L=64, N_c=1024`, the three ARM-B lambdas | **288 runs** | **yes — reused, §1** |
| `L=64, N_c=1024`, the other ten grid lambdas | 0 | — |
| `L=48`, any `N_c`, any lambda | 0 on Ruche; SMCSTAT `B-INJ` has 24 Mac runs at `T=32`, not `T=48` | no: `T != L` |
| `L=32`, any `N_c`, any lambda | SMCSTAT `A-MV`/`B-CHK`/`B-INJ`/`B-T32`, ~300 Mac runs | no: all `N_c <= 256`, and the `A-MV`/`B-CHK` blocks are `T = 32` but at `lambda = 0.35`, which is off this grid |
| `L=64, N_c=2048`, any lambda | 0 anywhere | — |
| `L=32/48/64, N_c=128`, on this grid | 0 anywhere | — |

**[E]** The ten new `L = 64` lambdas, and every `L = 32` and `L = 48` cell in
this campaign, return **zero rows** anywhere — not in the corpus, not in any
task, at any `zeta`, any `N_c`. Nothing else in the campaign duplicates existing
compute.

The SMCSTAT `L = 32` blocks deserve the explicit note above because they are the
nearest miss: they are the right `L`, the right `dtau_mult` and the right `T`,
but they sit at `lambda = 0.35`, which is not a point of this grid (the nearest
is 0.3532), and at `N_c <= 256`. They were also run on the Mac, not on Ruche.
They are used in this task **only** as timing evidence in `COST_MODEL.md`, never
as physics.

## 3. The historical corpus is not poolable with anything here

**[E]** The `zeta = 0.35` corpus slice is 1,200 rows across
`L in {64, 80, 96, 112, 128}`, all at `N_c = 128`, `R = 12` per cell, on the
lambda grid

```
0.1775 0.2219 0.2366 0.2479 0.2588 0.2662 0.2701 0.2810 0.2923 0.3032
0.3106 0.3144 0.3254 0.3366 0.3476 0.3550 0.3588 0.3698 0.4141 0.4733
```

It is excluded from every quantitative statement in this task on **four**
independent grounds, any one of which would suffice:

1. **Different discretisation.** Every corpus row has `dtau = 12.0/(2 λ (L−1))`,
   i.e. `dtau_mult = 12.0`. This campaign runs the certified 6.0.
   `support/instrumented.py` carries the matching in-code warning. These are
   different discretisations of the same physics and their population means are
   not interchangeable. `shared/preflight.py` refuses any manifest carrying
   `dtau_mult = 12.0`.
2. **Different lambda grid.** Exactly one corpus lambda, `0.3032`, coincides
   with a grid point of this campaign. The other twelve do not.
3. **Different population size.** `N_c = 128` throughout, which is a rung of
   the *companion* arms only, never of the main ones.
4. **No recoverable seed.** The corpus has no `seed` column, so independence
   from the new streams cannot be established and pooling could silently
   duplicate replicates.

**Consequence, stated plainly: there are ZERO exactly-compatible cells between
this campaign and the historical corpus, at any `L`, any `lambda`, any `N_c`.**
Grounds 1 and 4 hold even at `L = 64, lambda = 0.3032, N_c = 128`, the single
cell where `L`, `lambda` and `N_c` all coincide.

This is the finding that motivates the matched `N_c = 128` companion arm — see
`NC128_COMPANION_RATIONALE.md`. It is also why brief §9's Figure C and §12's
quantitative questions could not be answered at all from the archive as it
stands, and why saying so is part of the deliverable rather than a gap in it.

The corpus is used in exactly two places, both non-quantitative:

- `LAMBDA_GRID_DECISION.md`, to locate where the cross-`L` ordering reverses, so
  that the grid brackets it. Only the *sign* of a difference is borrowed.
- Figure B panel 2, drawn on its own lambda grid, dashed, with
  `DESCRIPTIVE ONLY / no exact common cell with this campaign` printed **inside
  the axes**, not in a caption.

## 4. Seeds

All new seeds are drawn from `[31,000,000, 32,000,000)`. The largest seed
anywhere else in the tree is **30,500,015**, so disjointness is structural
rather than merely checked. `SEED_LEDGER.md` records the allocation; the
preflight verifies both the block bound and, redundantly, a direct set
intersection against `tools/existing_seeds.json` (2,596 entries, the union of
the predecessor's ledger and its own allocation).
