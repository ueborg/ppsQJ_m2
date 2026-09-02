# Cost model — measured on Ruche, from returned wall times

TASK-2026-09-02-MOCK-PRODUCTION, brief §8. Implemented in
`tools/cost_model.py` and re-derived at preflight time by `shared/preflight.py`.

**Every rate here is anchored on `wall_s` values actually recorded by completed
Ruche jobs of this identical production path.** None is derived from a requested
Slurm `--time`, and none is inherited from a projection.

## 1. Two corrections to the predecessor's model

Both were found by re-reading `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA`'s own
returned result JSONs rather than its summary tables.

### 1a. L = 64 is now measured

The predecessor had no Ruche run at `L = 64` and had to derive the rate by two
indirect routes, adopting 5.000 ms and flagging that "two independent
derivations agree to 1.3 %, which is reassuring and is not a measurement."

**[E]** ARM B's 288 completed runs give, by lambda:

| lambda | n_steps | median wall_s | ms per clone-window |
|---:|---:|---:|---:|
| 0.2932 | 395 | 1893 | 4.680 |
| 0.3032 | 408 | 2008 | 4.806 |
| 0.3132 | 421 | 2015 | 4.674 |

The derivation was 6 % conservative — the right direction, and no longer needed.
This model adopts **4.850 ms**, above all three measured values.

### 1b. The rate is U-shaped in N_c, and the predecessor extrapolated it flat

**[E]** At `L = 128, lambda = 0.3032`, over five completed rungs of the same
cell:

| N_c | source | runs | median wall_s | ms per clone-window |
|---:|---|---:|---:|---:|
| 64 | ARM2 | 64 | 2858 | 27.176 |
| 128 | ARM2 | 64 | 5637 | 26.805 |
| 256 | ARM2 | 64 | 9052 | **21.522** |
| 512 | ARM A512 | 48 | 19698 | **23.416** |
| 1024 | ARM A1024 | 32 | 46937 | **27.898** |

The predecessor saw only the first three, read the fall from 27.18 to 21.52 as a
small-batch inefficiency that "is gone by `N_c ~ 256`", and extrapolated the
`N_c = 256` rate flat to 512 and 1024. It said so explicitly and said that if
wrong it would be wrong optimistically. **It was: the flat model understated
`armA1024` by 30 %** (39 830 s predicted against 46 937 s observed).

**[I]** The rate is U-shaped. Below `N_c ~ 256` a per-window fixed overhead is
amortised over too few clones; above it, the live clone store
(`N_c * ((2L)^2 * 8 + 2 L^2 * 16)` bytes — 268 MB at `N_c = 256, L = 128`,
1074 MB at `N_c = 1024`) stops fitting in cache and memory traffic dominates.
Both branches matter here, because this campaign runs `N_c = 128, 1024` and
`2048`.

## 2. The model

```
rate_ms(L, N_c) = BASE_MS[L] * NC_FACTOR[N_c]

BASE_MS   = {32: 1.400, 48: 3.000, 64: 4.850, 80: 8.550, 128: 27.898}
NC_FACTOR = {128: 1.35,  1024: 1.00, 2048: 1.20}
PESSIMISTIC = 1.40
PACKING     = 1.15
```

`BASE_MS` is the rate at the reference population `N_c = 1024`.

### BASE_MS provenance, entry by entry

| L | how | value |
|---:|---|---:|
| 64 | **MEASURED**, ARM B, 288 runs at `N_c = 1024` | 4.850 (above all three lambdas' 4.674–4.806) |
| 128 | **MEASURED**, ARM A1024, 32 runs at `N_c = 1024` | 27.898 |
| 48 | derived, see below | 3.000 |
| 32 | derived, see below | 1.400 |
| 80 | interpolated, used only to reject L = 80 | 8.550 |

**The downward derivations (L = 32, 48).** Three candidate `L`-scaling
exponents are available, each from a pair measured at the **same** `N_c`, so
that the whole `N_c` dependence cancels identically:

| pair | same N_c | exponent |
|---|---:|---:|
| L=64 vs L=128, Ruche | 1024 | 2.563 |
| L=96 vs L=128, Ruche | 512 | 2.469 |
| L=32 vs L=64, SMCSTAT Mac (`A-MV` / `A-HV`) | 256 | 2.339 |

**[I]** A *larger* exponent predicts a *smaller* rate at low `L`, so using a
measured exponent here would be the optimistic choice. The model instead uses
`p = 2.0` — below every measured exponent — and then rounds up:

```
L = 48:  4.850 * (48/64)^2 = 2.728  ->  adopt 3.000   (+10 %)
L = 32:  4.850 * (32/64)^2 = 1.213  ->  adopt 1.400   (+15 %)
```

Cross-checked by an independent Mac→Ruche transfer. **[E]** The Mac→Ruche ratio
is now itself measurable: SMCSTAT `A-HV` gives `L = 64, N_c = 256` at 2.969 ms
on the Mac against ARM B's 4.850 ms on Ruche, a ratio of 1.633 — and the same
ratio computed at `L = 96` from `A-BUD` against ARM 1 is 1.587, agreeing to
3 %. Applying 1.633 to the Mac rates:

```
L = 48:  1.282 (B-INJ, N_c=64)  * 1.633 = 2.094 ms
L = 32:  0.587 (A-MV,  N_c=256) * 1.633 = 0.959 ms
```

The adopted 3.000 and 1.400 are above both routes at both `L`.

### NC_FACTOR provenance

**`N_c = 2048` → 1.20.** **[E]** The only measured doubling of this kind is
`L = 128, N_c = 512 → 1024`: 23.416 → 27.898, i.e. **+19.1 %**. The smaller
doubling `256 → 512` at the same `L` cost only **+8.8 %**. This campaign's
doubling is at `L = 64` and takes the working set from 268 MB to 537 MB — the
*smaller* of those two transitions — so the true factor is expected between 1.09
and 1.19. **1.20 is adopted, above both.**

**[J] This is the single largest modelling judgement in the package.** If it is
wrong it is most likely wrong conservatively, and the `mockL64nc2048` arm has a
4 h wall limit against a 1.39 h prediction, so it would take a factor of 2.9
error to threaten the job.

**`N_c = 128` → 1.35.** **[E]** Small-batch penalties against the same `L`'s
`N_c = 256` rate: 26.805/21.522 = **1.245** on Ruche at `L = 128`, and
0.682/0.587 = **1.162** on the Mac at `L = 32`. Against the `N_c = 1024`
reference the penalty is larger still, since the reference itself sits on the
memory branch. 1.35 is above every measured comparison. These arms are 21 % of
the campaign's core-hours and their wall limits are 25× their predictions, so
over-provisioning them costs nothing.

### PESSIMISTIC = 1.40

Inherited unchanged. **[E]** Now known to be ample: ARM B's observed max/median
wall spread was 1.077, and its whole 288-task array finished in 2.76 h against a
2.84 h central prediction.

### PACKING = 1.15

**[E]** Elapsed time for an `--array=...%C` job is throughput-bound at
`core_h / C`, times a packing factor. Calibrated on ARM B, the only completed
multi-wave array in the programme: 288 tasks, cap %64, **157.8 core-hours
actually consumed**, `157.8 / 64 = 2.47 h`, **observed span 2.76 h** →
factor **1.118**. The model adopts 1.15.

Reconstructed from the arrays' own `.out` timestamps, which also establish the
concurrency regime — see §5.

## 3. Back-test against every completed Ruche cell

Predicted / observed ≥ 1 means the model is conservative.

| cell | source | predicted | observed | ratio |
|---|---|---:|---:|---:|
| L=64, N_c=1024, λ=0.2932 | ARM B | 1962 s | 1893 s | **1.036** |
| L=64, N_c=1024, λ=0.3032 | ARM B | 2026 s | 2008 s | **1.009** |
| L=64, N_c=1024, λ=0.3132 | ARM B | 2091 s | 2015 s | **1.038** |
| L=128, N_c=1024, λ=0.3032 | ARM A1024 | 46 936 s | 46 937 s | **1.000** |

## 4. n_steps — exact, not estimated

```
delta_tau = dtau_mult / (2 * lambda * (L - 1)),   n_steps = ceil(T / delta_tau)
```

which is `support/instrumented.py` lines 127–128 verbatim (`alpha == lambda` on
this cut). Verified against measured values at `L = 64` (395 / 408 / 421),
`L = 96` (922) and `L = 128` (1643), all exact. `n_steps` depends on lambda, so
the 13 grid points do not cost the same; the low-lambda end is the cheapest.

| L | n_steps at λ=0.2332 | at λ=0.2932 | at λ=0.3532 |
|---:|---:|---:|---:|
| 32 | 78 | 97 | 117 |
| 48 | 176 | 221 | 266 |
| 64 | 314 | 395 | 475 |

## 5. Concurrency — what the allocation actually granted

**[E]** Reconstructed from the start/finish timestamps in the predecessor's
`.out` logs on 2026-09-02:

| array | tasks | cap | max concurrent observed | span |
|---|---:|---|---:|---:|
| armB | 288 | %64 | **64** | 2.76 h |
| armA512 | 48 | %64 | 48 (all of them) | 5.86 h |
| armA1024 | 32 | %64 | 32 (all of them) | 13.21 h |

All three started within 12 seconds of each other and ran **144 tasks
concurrently**. So on that occasion the allocation granted the full **64 slots
per array**, not 64 in total — the question the predecessor's runbook had to
leave open. That is the regime the elapsed figures below assume, and
`RUCHE_RUNBOOK.md` §2 gives the command to re-confirm it, because one
observation of an allocation is not a guarantee about it.

This campaign asks for more arrays than that test exercised (seven, capped at
%64 each). The runbook gives the throttle and the ordering to use if the slots
turn out to be shared.

## 6. Per-arm cost table

| arm | L | N_c | λ | R | tasks | rate ms | slowest task | core-h | elapsed at %64 | peak MB | request | partition | --time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `mockL32` | 32 | 1024 | 13 | 24 | 312 | 1.400 | 0.05 h | 12.1 | 0.22 h | 262 | 1G | cpu_short | 01:00:00 |
| `mockL48` | 48 | 1024 | 13 | 24 | 312 | 3.000 | 0.23 h | 58.8 | 1.06 h | 430 | 1G | cpu_med | 02:00:00 |
| `mockL64` | 64 | 1024 | 10 | 24 | 240 | 4.850 | 0.66 h | 129.3 | **2.32 h** | 665 | 2G | cpu_med | 03:00:00 |
| `mockL64nc2048` | 64 | 2048 | 3 | 24 | 72 | 5.820 | 1.39 h | 97.3 | 1.75 h | 1202 | 3G | cpu_med | 04:00:00 |
| **main subtotal** | | | | | **936** | | | **297.5** | | | | | |
| `mockNC128L32` | 32 | 128 | 13 | 48 | 624 | 1.890 | 0.01 h | 4.1 | 0.07 h | 145 | 1G | cpu_short | 01:00:00 |
| `mockNC128L48` | 48 | 128 | 13 | 48 | 624 | 4.050 | 0.04 h | 19.9 | 0.36 h | 166 | 1G | cpu_short | 01:00:00 |
| `mockNC128L64` | 64 | 128 | 13 | 48 | 624 | 6.548 | 0.11 h | 57.3 | 1.03 h | 195 | 1G | cpu_short | 01:00:00 |
| **companion subtotal** | | | | | **1872** | | | **81.3** | | | | | |
| **TOTAL** | | | | | **2808** | | | **378.8** | | | | | |

Pessimistic total: **530.3 core-hours**.

## 7. Wall-clock — both numbers the brief asked for

**Core-hours: 378.8** predicted, 530.3 pessimistic.

**Elapsed, all seven arrays submitted together at %64 each** (the regime
observed in §5): the campaign finishes when its slowest arm does, which is
`mockL64` at **2.32 h predicted, 3.25 h pessimistic**. Every other arm is
finished by then; the second-longest is `mockL64nc2048` at 1.75 h.

**The ≤ 3 h requirement in brief §8 is met on the predicted figure and is
marginal on the ×1.40 pessimistic band.** Both are stated rather than only the
favourable one. The pessimistic band has never been needed on this code path:
ARM B, the only completed multi-wave array, came in at 2.76 h against a 2.84 h
central prediction — 3 % under, not 40 % over.

**If the allocation turns out to give 64 slots in TOTAL** rather than per array,
elapsed becomes throughput-bound at `378.8 / 64 * 1.15 = 6.8 h`. That is not a
3-hour campaign, and the runbook's fallback is to submit the four main arms
first (297.5 core-h → 5.3 h) and the three companion arms afterwards.

## 8. Honest limits of this model

- **`L = 32` and `L = 48` have never been run on Ruche at all.** They are
  downward extrapolations from a measured `L = 64`, deliberately using an
  exponent below every measured one and then rounded up, and cross-checked by a
  Mac→Ruche transfer whose ratio is itself now measured at two values of `L`.
  They are together 19 % of the campaign's core-hours and their wall limits are
  9–20× their predictions, so an error here cannot kill a job.
- **`NC_FACTOR[2048] = 1.20` is an extrapolation of one measured doubling at a
  different `L`.** §2 gives the bracket it sits in.
- **Ruche partition limits are inherited** (`cpu_short` 1 h, `cpu_med` 4 h,
  `cpu_long` 7 d) and were **not** re-verified against the live cluster; this
  session has no cluster access and must not obtain any. `RUCHE_RUNBOOK.md` §2
  gives the commands to confirm them before submitting.
- **Nothing here bounds queue wait time**, which is not a property of the
  package.
