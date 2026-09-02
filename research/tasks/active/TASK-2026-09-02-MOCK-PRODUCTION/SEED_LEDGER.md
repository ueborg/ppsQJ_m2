# Seed ledger and the disjointness proof

TASK-2026-09-02-MOCK-PRODUCTION, brief §16 ("fresh non-overlapping seeds").

## The block

```
[31,000,000, 32,000,000)
```

**[E]** The largest seed anywhere else in the task tree is **30,500,015**
(`TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/armA2048_optional`). The lowest seed
allocated here exceeds it by 499,985, so **disjointness is structural rather
than merely observed**: no arithmetic in `tools/build_arms.py` can produce a
value below the floor, and the floor is above the ceiling of everything that
came before.

## Allocation rule

```
seed = seed_base[arm] + 1000 * grid_index + replicate_index
```

`grid_index` is the index into the **frozen 13-point lambda grid**, not a
position within the arm's own lambda list. That matters for `mockL64`, which
omits grid indices 6, 7 and 8 because those cells already exist as ARM B: the
skipped lanes are **not reallocated to other lambdas**, so if a later task ever
needs those cells at a different `N_c` the lane numbering still lines up, and no
seed can be reused for a different lambda by accident.

Arm bases are 100,000 apart; an arm spans at most `12 * 1000 + 47 = 12,047`, so
no arm can reach the next arm's base.

| arm | base | seeds | rows |
|---|---:|---|---:|
| `mockL32` | 31,000,000 | 31,000,000 – 31,012,023 | 312 |
| `mockL48` | 31,100,000 | 31,100,000 – 31,112,023 | 312 |
| `mockL64` | 31,200,000 | 31,200,000 – 31,212,023 | 240 |
| `mockL64nc2048` | 31,300,000 | 31,306,000 – 31,308,023 | 72 |
| `mockNC128L32` | 31,400,000 | 31,400,000 – 31,412,047 | 624 |
| `mockNC128L48` | 31,500,000 | 31,500,000 – 31,512,047 | 624 |
| `mockNC128L64` | 31,600,000 | 31,600,000 – 31,612,047 | 624 |

`mockL64nc2048` starts at 31,306,000 rather than 31,300,000 because it uses grid
indices 6, 7 and 8 — the three central lambdas — and the lane rule ties the
offset to the lambda, not to the arm-local position. That is the rule working,
not an anomaly.

## Verified

```
allocated: 2808, distinct: 2808, range 31000000-31612047        OK
existing seeds scanned: 2596, max 30500015, overlap = 0         OK
```

`tools/existing_seeds.json` is the union of
`TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/tools/existing_seeds.json` (2,116 seeds
across SMCSTAT, GENCOL, SMCCERT and SMCRUCHE-READY) with that task's own 480
allocated seeds — so the ledger this campaign checks against includes the
predecessor's freshly allocated block, which the predecessor's own ledger by
construction could not.

`shared/preflight.py` checks disjointness **twice**, independently:

1. `seeds in the fresh block` — every seed lies in `[31e6, 32e6)`;
2. `no overlap with predecessors` — a direct set intersection against
   `tools/existing_seeds.json`.

Either check alone would catch a hand-edited manifest. Both are hard failures.

## Reused populations carry their own seeds unchanged

The 288 ARM-B populations reused from `frozen_inputs/armB_populations.csv` keep
their original seeds, **30,300,000 – 30,302,095**. They are in the predecessor's
block, not this one, which is correct: they are the predecessor's compute,
reused with provenance, not this task's. They cannot collide with anything here
because this block starts 697,905 above their maximum.
