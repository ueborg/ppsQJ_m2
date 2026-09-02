# Seed ledger

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief §7 and §15.

## Allocation

All new seeds come from the block `[30_000_000, 31_000_000)`, laid out as

```
30_000_000 + 100_000 * arm_index + 1_000 * lambda_lane + replicate
```

| arm | N_c | lambda | lane | seeds | count |
|---|---:|---:|---:|---|---:|
| `armA512`  | 512  | 0.3032 | 0 | 30100000–30100047 | 48 |
| `armA1024` | 1024 | 0.3032 | 0 | 30200000–30200031 | 32 |
| `armB`     | 1024 | 0.2932 | 0 | 30300000–30300095 | 96 |
| `armB`     | 1024 | 0.3032 | 1 | 30301000–30301095 | 96 |
| `armB`     | 1024 | 0.3132 | 2 | 30302000–30302095 | 96 |
| `armC`     | 512  | 0.2932 | 0 | 30400000–30400047 | 48 |
| `armC`     | 512  | 0.3132 | 1 | 30401000–30401047 | 48 |
| `armA2048_optional` | 2048 | 0.3032 | 0 | 30500000–30500015 | 16 |
| | | | | **total** | **480** |

**480 seeds, all distinct**, asserted at generation time in
`tools/build_arms.py` and re-checked per arm by `shared/preflight.py`. The full
list is `tools/allocated_seeds.json`.

Separate lambda lanes matter statistically as well as bookkeeping-wise: the
three stencil points draw from disjoint seed ranges, so their population means
are statistically independent and the bootstrap in
`analysis/combined_analysis.py` may treat them as such when forming `d_-`,
`d_+` and `q`.

## Disjointness from every completed run

Every seed ever used in this programme was collected into
`tools/existing_seeds.json` by scanning the 11 SMCSTAT JSONL blocks, the
SMCSTAT / GENCOL / SMCCERT / SMCRUCHE-READY manifests and the 304 completed
ARM1/ARM2 result JSONs:

```
2116 distinct existing seeds,  min 1,  max 20384063
```

The new block starts at 30,100,000. **Disjointness is structural**: the lowest
new seed exceeds the highest old seed by more than 9.7 million. Two independent
checks enforce it anyway:

- `preflight.py` requires every manifest seed to satisfy
  `30_000_000 <= seed < 31_000_000`;
- `preflight.py` also intersects the arm's seeds directly against
  `tools/existing_seeds.json` and reports the collision count.

Audit result: **0 collisions across all five arms.** Reproduced in
`VALIDATION.md`, and the negative control (injecting predecessor seed
`20192000` into `armA512/manifest.csv`) makes the preflight exit 1.

## Streams are fresh, not continued

Note a deliberate difference from `TASK-2026-09-01-SMCRUCHE-READY`, whose ARM1
seeds *continued* the SMCSTAT `A-P96` stream at an identical cell so the blocks
could be pooled. Nothing here does that: every cell in this campaign is either
new (`N_c` 512/1024/2048 at L = 128; everything at L = 64; both neighbouring
lambdas) or is only ever *compared against* a predecessor rung rather than
pooled with it. So fresh non-overlapping seeds are the correct choice at every
row, and no manifest continues an existing stream.

Consequence for `numpy.random.default_rng`: distinct integer seeds in this range
give independent SeedSequence-derived streams, and `run_cell.py` spawns
`N_c` child streams per population from the row's seed. Two different manifest
rows therefore share no clone-level randomness.
