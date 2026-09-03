# Reuse and duplication audit

## 1. What is reused

All 39 completed `N_c = 1024` cells of `TASK-2026-09-02-MOCK-PRODUCTION`:
`L = 32, 48, 64` at `lambda = 0.2332 … 0.3532`, 1,152 populations, frozen into
`frozen_inputs/predecessor_nc1024_populations.csv` with per-row provenance
(`source_task`, `source_arm`, `source_file`) and a file-level sha256 in
`INPUTS_LEDGER.md`.

Recomputing them would have cost roughly **195 core-hours** — three times this
entire campaign — and would have produced a second, slightly different
measurement of the same 39 cells, which is worse than useless: the analysis
averages populations by `(L, lambda)`, so the two would have been silently
pooled.

They are **not regenerated**. `tools/freeze_predecessor.py` reads and copies;
it never runs the sampler.

## 2. What is refused, and why the refusal is asserted

| refused | why | how it is enforced |
|---|---|---|
| `mockNC128L32/48/64` | the matched low-`N_c` companion arms were **cancelled** and returned zero results | `freeze_predecessor.py` globs their `results/` and prints the count; asserted empty |
| `mockL64nc2048` | **72 results exist** and are deliberately not read: a different population size has no place in an `N_c = 1024` curve-shape and crossing extension | the freeze script takes an explicit arm allowlist `{mockL32, mockL48, mockL64}`, not a glob; the refusal is printed |
| `historical_corpus_zeta035.csv` | `dtau_mult = 12.0` throughout; not poolable with the certified 6.0 | never opened by any file in this task |
| `armA2048_optional`, `L = 80`, `L = 96`, `L = 128` | out of scope | no manifest, no reference |

The `N_c = 2048` refusal is the one that needed asserting rather than assuming.
Those results are sitting on disk one directory away. A slightly wider glob —
`mock*/results/*.json` instead of an allowlist — would have swept 72 populations
at a different population size into the `L = 64` curve, and no file in the task
would have recorded that it had happened. The preflight's `N_c frozen [1024]`
check is the second line of defence, and negative control N08 confirms it fires.

## 3. Duplicate scan

`tools/dedup_scan.py` D1 builds the set of physical cells

```
(L, T, zeta, lambda, N_c, dtau_mult, resample_scheme)
```

from the frozen snapshot **and from every other `manifest.csv` under
`research/tasks/active/`** — 17 manifests, not just the predecessor's, so a
clash with a task nobody thought to mention is still caught — and intersects it
with this task's 288 rows.

```
scanned 17 other manifests under research/tasks/active/
distinct pre-existing physical cells: 92
duplicates: 0   (the 12 new cells exist nowhere else)
```

Independently, `shared/preflight.py` refuses any manifest containing one of the
thirteen already-measured lambdas (`no predecessor duplication`, negative
control N01) and requires all four new ones to be present (N02).

## 4. One design either side of the join

Reuse is only legitimate if the reused half and the new half are the same
measurement. `tools/dedup_scan.py` D4 checks it directly:

```
reused (13 lambdas)  zeta=[0.35] N_c=[1024] dtau=[6.0] scheme=['systematic'] T==L=True
new    ( 4 lambdas)  zeta=[0.35] N_c=[1024] dtau=[6.0] scheme=['systematic'] T==L=True
```

plus, from `INPUTS_LEDGER.md`, the same sampler bytes (`0a33c403…`) and the same
`R = 24` primary block. The three `L = 64` centre cells are the only ones at a
different `R` (96), and they are cut to block A in seed order by the analysis —
the predecessor's own rule, applied here rather than inherited pre-cut.

The preflight additionally rejects an arm whose `R` is not 24 (`R matches the
reused half`, negative control N06), because a different `R` would make the join
a change of *precision* as well as a change of lambda, and every join statistic
would then be confounded at exactly the point it is most delicate.

## 5. What reuse does not buy

The reused populations are the predecessor's **task-verified** data. Freezing
them into this package gives them provenance and reproducibility here; it does
not promote them, and nothing in `research/state/**` was written. A number
recomputed from this snapshot is a recomputation of a task-verified quantity,
not a canonical one.
