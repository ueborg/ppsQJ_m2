# SEED_LEDGER — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Every seed this campaign allocates, why it cannot collide with anything that
exists, and how a top-up avoids re-labelling an existing population.

Verified by `tools/dedup_scan.py` §3 against **every `manifest.csv` and every
result JSON in the repository**, not only against the ones that ran.

---

## 1. The block, and why disjointness is structural

```
this campaign, immediate    [33,000,000 , 33,400,000)
this campaign, conditional  [33,500,000 , 33,700,000)
repository ceiling before this task            32,203,023
```

`[E]` The highest seed allocated anywhere in the repository — including in the
manifests of arms that were **built and never run** (`mockNC128L*`, `armC`,
`armA2048_optional`), which is exactly where a collision would otherwise hide —
is `32,203,023`. This campaign's floor is `796,977` above it. `[E]` Disjointness
is therefore **structural**, not merely observed: no arithmetic accident inside
the lane rule can reach an allocated seed.

`[E]` The immediate and conditional blocks are separated by a further 100 000,
so releasing a conditional arm later cannot collide with an immediate one.

## 2. The lane rule

```
seed = arm_base + 1000 * cell_index + replicate_index

arm_base        = 33,000,000 + 20,000 * arm_ordinal      (immediate)
                = 33,500,000 + 20,000 * arm_ordinal      (conditional)
cell_index      enumerates the arm's cells in FROZEN order (lambda, then dtau_mult)
replicate_index 0 .. R-1, and for a TOP-UP starts at R_existing, not at 0
```

`[E]` 1 000 replicates per lane and 20 lanes per arm, against a maximum of 48
replicates and 7 lanes actually used. The headroom is deliberate: it means a
later top-up of any cell can extend its own lane rather than needing a new block.

## 3. Top-ups start where the existing populations stop

`[E]` This is the part that would be easy to get wrong. Five cells already hold
exact-compatible populations and are **topped up**, not recomputed. Their fresh
replicates start at `replicate_index = R_existing`:

| cell | existing R | existing source | fresh replicate indices | fresh seeds |
|---|---:|---|---|---|
| `L=64 N_c=2048 lam=0.3032` | 24 | `MOCK-PRODUCTION/mockL64nc2048` | 24–47 | 33000024–33000047 |
| `L=64 N_c=1024 lam=0.2232` | 24 | `LOWLAMBDA-EXTENSION/lowlamL64` | 24–47 | in `B_L64_cross_nc1024` |
| `L=64 N_c=1024 lam=0.2332` | 24 | `MOCK-PRODUCTION/mockL64` | 24–47 | " |
| `L=64 N_c=1024 lam=0.2432` | 24 | `MOCK-PRODUCTION/mockL64` | 24–47 | " |
| `L=32/48 N_c=1024` at the same three `lambda` | 24 each | `mockL32/48`, `lowlamL32/48` | 24–47 | in `B2_L*_nc1024` |

`[J]` The seeds themselves could not collide whatever index was used — the blocks
are disjoint. Starting at `R_existing` is about the **ledger**: a fresh
population labelled `replicate 0` in a cell that already has a `replicate 0`
makes the two indistinguishable in any later per-replicate analysis, and block
cuts (`P4`, split-half) are taken **in seed order**, so a duplicated label
would silently put two different populations in the same block position.

`[E]` The result: `A_L64_nc2048_topup` allocates `33000024–33000047`, and
`33000000–33000023` are **deliberately never allocated**. The gap is the
existing 24 populations' notional lane. It is not an error and the preflight
does not treat it as one.

## 4. Allocation, immediate group

| arm | base | lanes | rows | seed range |
|---|---:|---:|---:|---|
| `A_L64_nc2048_topup` | 33 000 000 | 1 | 24 | 33000024–33000047 |
| `A_L64_nc4096` | 33 020 000 | 1 | 48 | 33020000–33020047 |
| `A_L64_nc8192` | 33 040 000 | 1 | 48 | 33040000–33040047 |
| `B_L64_cross_nc512` | 33 060 000 | 7 | 336 | 33060000–33066047 |
| `B_L64_cross_nc1024` | 33 080 000 | 7 | 264 | 33080000–33086047 |
| `B_L64_cross_nc2048` | 33 100 000 | 7 | 336 | 33100000–33106047 |
| `B2_L32_nc512` | 33 120 000 | 7 | 336 | 33120000–33126047 |
| `B2_L32_nc1024` | 33 140 000 | 7 | 264 | 33140000–33146047 |
| `B2_L32_nc2048` | 33 160 000 | 7 | 336 | 33160000–33166047 |
| `B2_L48_nc512` | 33 180 000 | 7 | 336 | 33180000–33186047 |
| `B2_L48_nc1024` | 33 200 000 | 7 | 264 | 33200000–33206047 |
| `B2_L48_nc2048` | 33 220 000 | 7 | 336 | 33220000–33226047 |
| `C_L96_nc1024` | 33 240 000 | 1 | 24 | 33240000–33240023 |
| `C_L96_nc2048` | 33 260 000 | 1 | 24 | 33260000–33260023 |
| `D_L128_nc2048` | 33 280 000 | 1 | 16 | 33280000–33280015 |
| `E_L64_dtau_nc64` | 33 300 000 | 3 | 144 | 33300000–33302047 |
| `E_L64_dtau_nc256` | 33 320 000 | 3 | 144 | 33320000–33322047 |
| | | | **3 280** | all distinct |

## 5. Allocation, conditional group

| arm | rows | seed range |
|---|---:|---|
| `cond_D2_L128_nc4096` | 8 | 33500000–33500007 |
| `cond_M96_nc1024` | 108 | 33520000–33528011 |
| `cond_M96_nc2048` | 108 | 33540000–33548011 |
| `cond_M128_nc2048` | 72 | 33560000–33568007 |
| `cond_M128_nc4096` | 72 | 33580000–33588007 |
| `cond_LOWZ_nc64` | 48 | 33600000–33600047 |
| `cond_LOWZ_nc256` | 48 | 33620000–33620047 |
| | **464** | disjoint from the immediate 3 280 |

`[E]` The two `M96` arms and the two `M128` arms carry **different** seeds even
though they are the same physical scan at two `N_c`, so that releasing the wrong
one is visible in the data rather than silent.

## 6. Machine-readable

`tools/allocated_seeds.json` (3 280 immediate) and
`tools/conditional_summary.json` (464 conditional). Every arm's preflight checks
its own seeds against the immediate ledger and fails if one is missing from it.
