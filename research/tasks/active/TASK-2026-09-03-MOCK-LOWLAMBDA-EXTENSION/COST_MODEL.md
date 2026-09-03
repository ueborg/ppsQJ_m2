# Cost model

Every number in this task is fitted to `wall_s` **actually recorded by
completed Ruche jobs** of this identical code path, at this identical `N_c`, on
this identical lambda family. Never to a requested Slurm `--time`. Never to a
predecessor projection.

Implementation: `tools/cost_model.py`. It is **refitted from the frozen data by
the preflight on every run**, which fails non-zero if the literals have drifted
from the measurements by more than 0.5 % where they are used
(`tools/negative_controls.py` N18).

---

## 1. The measured runtime distribution

From `frozen_inputs/predecessor_nc1024_populations.csv`, which is the
predecessor's own returned JSONs. `wall_s` in seconds.

| L | n | min | median | mean | p95 | max | max/median | total core-h |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 312 | 119.8 | 151.1 | 147.9 | 163.6 | 175.9 | 1.165 | 12.8 |
| 48 | 312 | 554.6 | 619.8 | 637.2 | 712.0 | 746.9 | 1.205 | 55.2 |
| 64 | 240 | 1535.0 | 1886.4 | 1901.4 | 2190.5 | 2261.5 | 1.199 | 126.8 |

(The `L = 64` row is `mockL64`'s 240 runs. The cost fit below additionally uses
the 288 reused ARM-B runs at the same `L` and `N_c`, giving `n = 528`.)

Per-clone-window rates over each arm's own 13-lambda span:

| L | min ms | median ms | max ms |
|---:|---:|---:|---:|
| 32 | 1.294 | 1.479 | 1.888 |
| 48 | 2.330 | 2.839 | 3.354 |
| 64 | 3.840 | 4.809 | 5.527 |

**They are not constant.** A per-clone-window rate is what the predecessor's
model assumed, and the returned data say the rate drifts systematically with
`n_steps` rather than scattering — because a run carries a fixed per-run cost
(interpreter start, the `pps_qj` import, lattice and population construction,
the final observable pass) that a strictly proportional model must smear into
the rate.

---

## 2. The adopted model

Least squares over every `N_c = 1024` population at that `L`:

| L | fit | resid sd | n | `n_steps` span fitted |
|---:|---|---:|---:|---|
| 32 | `wall_s = 0.815551 · n_steps + 68.43` | 7.1 s | 312 | 78–117 |
| 48 | `wall_s = 1.588743 · n_steps + 286.09` | 24.3 s | 312 | 176–266 |
| 64 | `wall_s = 2.723572 · n_steps + 850.23` | 99.9 s | 528 | 314–475 |

`n_steps` is read from the JSON each run wrote, not recomputed, so the
regressor is exactly what the sampler did. The discretisation is

```
dtau    = dtau_mult / (2 · lambda · (L - 1))
n_steps = ceil(T / dtau)
```

verified against the recorded `n_steps` at all 13 old lambdas and all three `L`.

**The prediction adopted is the larger of two measured-data models:**

1. the affine fit above;
2. `RATE_MAX_MS[L] · 1e-3 · N_c · n_steps`, using the **worst** per-clone-window
   rate any completed run of that arm actually showed (1.888 / 3.354 / 5.527 ms).

Taking the max is not double-counting: the two disagree in different directions
at different `n_steps`, and a `--time` limit should be sized by whichever says
the job is slower.

---

## 3. The extrapolation, and which way it errs

The four new lambdas are **below** the fitted range, so `n_steps` extrapolates
downward:

| L | fitted over | used at | ratio to the fitted floor |
|---:|---|---|---:|
| 32 | 78–117 | 64–74 | 0.82× |
| 48 | 176–266 | 146–168 | 0.83× |
| 64 | 314–475 | 260–300 | 0.83× |

Nothing reaches into a regime the campaign has not already measured within
20 %.

A positive intercept makes the affine model predict **more** time at low
`n_steps` than a proportional model does. At `L = 64, lambda = 0.1932` the
affine model says 1558 s where a median-rate proportional model says 1280 s —
22 % more conservative, in the direction that matters for a time limit. That is
deliberate.

---

## 4. Per-arm cost

`n_steps`, then the two models, then the adopted value, per new lambda:

### `lowlamL32` — `L = 32`, `T = 32`

| lambda | `n_steps` | affine (s) | worst-rate (s) | **adopted (s)** |
|---:|---:|---:|---:|---:|
| 0.1932 | 64 | 120.6 | 123.7 | **123.7** |
| 0.2032 | 68 | 123.9 | 131.5 | **131.5** |
| 0.2132 | 71 | 126.3 | 137.3 | **137.3** |
| 0.2232 | 74 | 128.8 | 143.1 | **143.1** |

96 tasks · **3.57 core-h** (5.00 pessimistic). Slowest task **2.4 min**
(3.3 min pessimistic). Peak 262 MB.

### `lowlamL48` — `L = 48`, `T = 48`

| lambda | `n_steps` | affine (s) | worst-rate (s) | **adopted (s)** |
|---:|---:|---:|---:|---:|
| 0.1932 | 146 | 518.0 | 501.4 | **518.0** |
| 0.2032 | 153 | 529.2 | 525.5 | **529.2** |
| 0.2132 | 161 | 541.9 | 553.0 | **553.0** |
| 0.2232 | 168 | 553.0 | 577.0 | **577.0** |

96 tasks · **14.51 core-h** (20.31 pessimistic). Slowest task **9.6 min**
(13.5 min pessimistic). Peak 430 MB.

### `lowlamL64` — `L = 64`, `T = 64`

| lambda | `n_steps` | affine (s) | worst-rate (s) | **adopted (s)** |
|---:|---:|---:|---:|---:|
| 0.1932 | 260 | 1558.4 | 1471.5 | **1558.4** |
| 0.2032 | 274 | 1596.5 | 1550.7 | **1596.5** |
| 0.2132 | 287 | 1631.9 | 1624.3 | **1631.9** |
| 0.2232 | 300 | 1667.3 | 1697.9 | **1697.9** |

96 tasks · **43.23 core-h** (60.52 pessimistic). Slowest task **28.3 min**
(39.6 min pessimistic). Peak 665 MB.

**Campaign total: 288 tasks, 61.31 core-hours (85.83 pessimistic).**

---

## 5. Wall-clock

The predecessor's elapsed model was `max(core_h / C · PACKING, slowest_h)`.
That is right for a 240–624 task array where many waves average out. **These
arrays are 96 tasks at `%64` — exactly two waves** — so the wave floor, not
throughput, is what binds:

```
elapsed = max( core_h / 64 · 1.15 ,  ceil(96/64) · slowest_task )
```

| arm | throughput bound | two-wave floor | **expected** | pessimistic ×1.40 |
|---|---:|---:|---:|---:|
| `lowlamL32` | 3.9 min | 4.8 min | **4.8 min** | 6.7 min |
| `lowlamL48` | 15.6 min | 19.2 min | **19.2 min** | 26.9 min |
| `lowlamL64` | 46.6 min | 56.6 min | **56.6 min** | 79.2 min |

All three run concurrently, so the campaign's elapsed time is the long pole:

- **expected 0.94 h (56.6 min)**
- **pessimistic 1.32 h (79.2 min)**

**Queue wait is excluded from every figure above and is expected to dominate
all of them.** These are three ~1 h `cpu_med` requests; how long they sit
pending is a property of the cluster on the day, not of this package. The
`%64` cap is a concurrency cap only.

`PACKING = 1.15` is inherited: calibrated on ARM B, the only completed
multi-wave array in the programme — 288 tasks, `%64`, 157.8 core-hours
consumed, 2.47 h throughput-bound, 2.76 h observed span → 1.118, adopted 1.15.

`PESSIMISTIC = 1.40` is inherited and is now known to be ample: this
campaign's own three arms showed max/median wall spreads of 1.165, 1.205 and
1.199.

---

## 6. Slurm limits

The smallest safe request with substantial margin, on `cpu_med` in every case
(`SCHEDULER_DECISION.md`):

| arm | `--time` | pessimistic slowest | margin | `--mem` | 1.5 × peak | margin |
|---|---|---:|---:|---|---:|---:|
| `lowlamL32` | `00:20:00` | 3.3 min | **6.0×** | `1G` | 0.38 G | 2.6× |
| `lowlamL48` | `00:45:00` | 13.5 min | **3.3×** | `1G` | 0.63 G | 1.6× |
| `lowlamL64` | `02:00:00` | 39.6 min | **3.0×** | `2G` | 0.97 G | 2.1× |

The preflight recomputes all of this and exits non-zero if `--time` falls below
1.40 × the slowest predicted task, if `--mem` falls below 1.5 × the estimated
peak, or if `--time` exceeds the partition's `MaxTime`
(`tools/negative_controls.py` N12, N13, N14).

The memory formula is the predecessor's, validated there against ARM2's real
footprint. `L` and `N_c` are unchanged here, so the footprint is unchanged;
lambda does not touch memory.

---

## 7. What this model does not claim

- It is not an `L`-scaling law. It is three independent one-dimensional fits,
  each used only within and just below its own measured span. No exponent is
  extracted and none is needed.
- It says nothing about `L = 80`, `96` or `128`. The predecessor's
  `L80_RUNTIME_GATE.md` and `L128_NC2048_HANDOFF.md` are unaffected.
- It says nothing about `N_c` other than 1024. The predecessor's `NC_FACTOR`
  table is not carried over, because this task has only one population size.
