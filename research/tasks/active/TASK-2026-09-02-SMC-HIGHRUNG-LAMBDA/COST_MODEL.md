# Cost model — measured, not modelled

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief §6. Implemented in
`tools/cost_model.py` and re-derived at preflight time by
`shared/preflight.py`.

## The improvement over the predecessor

`TASK-2026-09-01-SMCRUCHE-READY` had to warn, in its own `submit.slurm`, that
"THE COST IS THE LEAST RELIABLE NUMBER IN THIS PACKAGE" and to treat its L = 128
rate as ±50 %, because it was derived from a Mac timing probe. That warning was
justified: **the derived figure was low by 45 %.**

| L | predecessor's assumed rate | actually measured on Ruche | error |
|---:|---:|---:|---:|
| 96  | 6.59 ms/clone-window | **11.51** ms | −43 % |
| 128 | 14.83 ms/clone-window (derived) | **21.52** ms | −31 % |

Every rate this campaign uses is now measured on Ruche, from a completed run of
this identical production path.

## Measured Ruche wall times

From the 304 completed ARM1/ARM2 result JSONs (median over each rung):

| L | N_c | R | n_steps | median wall_s | min | max | ms per clone-window |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 96  | 128  | 32 | 922  | 1381.4 | 1169.8 | 1466.4 | 11.705 |
| 96  | 256  | 32 | 922  | 2388.7 | 2214.2 | 2722.4 | 10.120 |
| 96  | 512  | 48 | 922  | 5433.3 | 4482.9 | 5659.5 | 11.510 |
| 128 | 64   | 64 | 1643 | 2857.6 | 2127.9 | 3102.6 | 27.176 |
| 128 | 128  | 64 | 1643 | 5637.2 | 4387.3 | 6032.2 | 26.805 |
| 128 | 256  | 64 | 1643 | 9052.2 | 8390.8 | 10093.6 | 21.522 |

## Rate scaling with N_c — the one modelling choice, stated

The per-clone-window rate **falls** with `N_c` at L = 128 (27.18 → 26.81 →
21.52) and then flattens at L = 96, where the three rungs give 11.71, 10.12 and
11.51 with no trend. The natural reading is a small-batch inefficiency that is
gone by `N_c ~ 256`.

The model therefore extrapolates to `N_c = 512, 1024, 2048` **from the
`N_c = 256` rate**, not from the smaller rungs, and not from a fitted power law
in `N_c` (the three L = 128 points do not support one: successive log-log slopes
are 0.980 then 0.683). This is the single place a judgement was made, and if it
is wrong it is wrong in the *optimistic* direction, which is what the ±40 %
pessimistic band exists for.

## Rate scaling with L, and the L = 64 figure

No Ruche run at L = 64 exists, so its rate is derived by two independent routes:

1. **Within-Ruche L-scaling.** The exponent from the two measured Ruche points
   is `ln(21.522/11.510)/ln(128/96) = 2.174`, giving
   `11.510 * (64/96)^2.174 = 4.773` ms.
2. **Mac → Ruche transfer.** The SMCSTAT local blocks measure L = 64 at 2.969
   ms (`A-HV`, `N_c = 256`) and L = 96 at 7.253 ms (`A-BUD`, `N_c = 64`), so the
   Ruche/Mac ratio at L = 96 is `11.510/7.253 = 1.587`, giving
   `2.969 * 1.587 = 4.712` ms.

They agree to 1.3 %. The model adopts **5.000 ms**, above both.

```
RATE_RUCHE_MS = {64: 5.000, 96: 11.510, 128: 21.522}
PESSIMISTIC   = 1.40
```

The ±40 % band covers the observed within-rung max/median spread (1.115 at
L = 128, `N_c = 256`) and the residual extrapolation uncertainty in `N_c`.

## n_steps — exact, not estimated

```
delta_tau = dtau_mult / (2 * lambda * (L - 1)),   n_steps = ceil(T / delta_tau)
```

which is `support/instrumented.py` lines 127–128 verbatim (`alpha == lambda` on
this cut). Verified against the measured values: L = 96 → 922 and L = 128 →
1643, both exact. Note `n_steps` depends on lambda, so the stencil's three
lambdas do not cost the same — `lambda_-` is the cheapest.

| L | lambda_- | lambda_0 | lambda_+ |
|---:|---:|---:|---:|
| 64  | 395 | 408 | 421 |
| 96  | 892 | 922 | 953 |
| 128 | 1589 | 1643 | 1698 |

## Memory

```
per_clone = (2L)^2 * 8  +  (2L * L) * 16   bytes
peak_MB   = 128 + 2 * N_c * per_clone / 1e6
```

the same formula the predecessor validated against ARM2's real footprint. The
request is ≥ 1.5× the estimate and `preflight.py` fails if it is not.

## Per-arm cost table

| arm | L | N_c | lambdas | R | tasks | slowest task | pessimistic | core-h | pessimistic | peak MB | request | partition | --time |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| **A** `armA512`  | 128 | 512  | 0.3032 | 48 | 48 | 5.03 h | 7.04 h | 241.4 | 338.0 | 1202 | 3G | cpu_long | 12:00:00 |
| **A** `armA1024` | 128 | 1024 | 0.3032 | 32 | 32 | 10.06 h | 14.08 h | 321.9 | 450.6 | 2275 | 5G | cpu_long | 24:00:00 |
| **B** `armB`     | 64  | 1024 | all three | 96 | 288 | 0.60 h | 0.84 h | 167.1 | 234.0 | 665 | 2G | cpu_med | 03:00:00 |
| **C** `armC`     | 128 | 512  | 0.2932, 0.3132 | 48 | 96 | 5.20 h | 7.28 h | 482.9 | 676.1 | 1202 | 3G | cpu_long | 12:00:00 |
| | | | | | **464** | | | **1213.3** | **1698.7** | | | | |
| *optional* `armA2048_optional` | 128 | 2048 | 0.3032 | 16 | 16 | 20.12 h | 28.16 h | 321.9 | 450.6 | 4423 | 9G | cpu_long | 48:00:00 |
| *rejected* ARM B2 (L = 96 bridge) | 96 | 512 | all three | 64 | 192 | 1.56 h | 2.18 h | 289.9 | 405.8 | 732 | — | — | — |

**Recommended overnight total: 464 tasks, 1213 core-hours** (1699 pessimistic).
**Optional extension: +322 core-hours** (451 pessimistic), 16 tasks.

## Elapsed wall-clock, and the concurrency question

This is the number most likely to be misread, so both readings are given.

**If each array gets its own 64 slots (192 concurrent across the three arms):**
elapsed is the slowest arm, which is `armA1024` at **10.1 h predicted, 14.1 h
pessimistic**. `armA512` and `armC` finish in ~5.2 h (7.3 h), `armB`'s 288 tasks
take 4.5 waves of 64 at 0.60 h ≈ **2.7 h** (3.8 h). That fits an overnight run,
with `armA1024` as the long pole — which is why the runbook says to queue it
first.

**If the allocation gives 64 slots in TOTAL across all arms:** elapsed is
throughput-bound at `1213 / 64 = 19.0 h` predicted, **26.5 h pessimistic**. That
is *not* an overnight run. In that case submit `armA1024`, `armA512` and `armB`
tonight (730 core-h → 11.4 h) and `armC` tomorrow.

The researcher should check which regime applies before submitting; the runbook
gives the command.

## Honest limits of this model

- The `N_c = 512/1024/2048` L = 128 rates are extrapolations from `N_c = 256`.
  They are the best available — a measured rate on the same machine, same code,
  same cell — but no run at those `N_c` exists yet anywhere.
- The L = 64 rate has never been measured on Ruche at all. Two independent
  derivations agree to 1.3 %, which is reassuring and is not a measurement.
- `armB`'s projected variance is scaled from `A-HV` at L = 64, **T = 32**, with
  a factor 2 applied for the doubled horizon. That factor is a judgement, not a
  measurement, and `armB`'s achieved SEM could differ from 0.0056 by a fair
  margin in either direction. It does not affect the design decision:
  `delta_lambda = 0.010` survives a factor of 2 in either direction at L = 64.
- Nothing here bounds queue wait time, which is not a property of the package.
