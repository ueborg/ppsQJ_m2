# Seed ledger and the disjointness proof

`TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION`. Brief §9: "exactly 288 fresh
non-overlapping seeds, zero overlap with predecessor seeds".

## The block

```
[32,000,000, 33,000,000)
```

**[E]** The largest seed anywhere else in the task tree is **31,612,047**
(`TASK-2026-09-02-MOCK-PRODUCTION/mockNC128L64`). The lowest seed allocated here
exceeds it by 387,953, so **disjointness is structural rather than merely
observed**: no arithmetic in `tools/build_arms.py` can produce a value below the
floor, the builder asserts the bound per seed, and the floor is above the
ceiling of everything that came before.

## Allocation rule

```
seed = seed_base[arm] + 1000 * grid_index + replicate_index
```

`grid_index` indexes the **frozen 17-point grid**, not a position within the
arm's own lambda list. This task uses lanes **0–3 only**; lanes 4–16 are
permanently reserved for the thirteen already-measured lambdas and can never be
handed to a new point by accident. That is the same rule the predecessor used,
extended to the longer grid — and it is why the predecessor's own reserved lanes
still line up.

Arm bases are 100,000 apart; an arm spans at most `3 * 1000 + 23 = 3,023`, so no
arm can reach the next arm's base.

| arm | base | seeds | rows |
|---|---:|---|---:|
| `lowlamL32` | 32,000,000 | 32,000,000 – 32,003,023 | 96 |
| `lowlamL48` | 32,100,000 | 32,100,000 – 32,103,023 | 96 |
| `lowlamL64` | 32,200,000 | 32,200,000 – 32,203,023 | 96 |
| | | | **288** |

## Verified

```
allocated: 288, distinct: 288, range 32000000-32203023             OK
existing seeds scanned: 5404, max 31612047, overlap = 0            OK
structural floor 32000000 > 31612047                               OK
tools/allocated_seeds.json matches the three manifests             OK
```

`tools/existing_seeds.json` (5,404 seeds) is the union of

- `TASK-2026-09-02-MOCK-PRODUCTION/tools/existing_seeds.json` (2,596 seeds
  across SMCSTAT, GENCOL, SMCCERT, SMCRUCHE-READY and SMC-HIGHRUNG-LAMBDA),
- that task's own 2,808 allocated seeds, and
- every seed **physically observed on disk** in its 864 returned JSONs and both
  of its frozen-input CSVs.

The third source is the one that matters: a ledger can be stale, but a returned
JSON is a seed that definitely ran. All 1,224 observed seeds were already in the
union — **0 new** — which is itself evidence that the predecessor's ledger was
complete.

## Checked three times, independently

1. `tools/build_arms.py` asserts `SEED_FLOOR <= s < SEED_CEIL` and no collision
   as it generates each seed;
2. `shared/preflight.py` checks the range and separately intersects against
   `tools/existing_seeds.json` — both hard failures (negative controls N04, N05);
3. `tools/dedup_scan.py` D2 re-derives all of it from the written manifests
   rather than from the builder's in-memory state.

Any one of the three would catch a hand-edited manifest.

## Reused populations carry their own seeds unchanged

The 1,152 reused populations in
`frozen_inputs/predecessor_nc1024_populations.csv` keep their original seeds,
**30,300,000 – 31,212,023**. They are in the predecessor's blocks, not this one,
which is correct: they are the predecessor's compute, reused with provenance,
not this task's. They cannot collide with anything here because this block starts
787,977 above their maximum.
