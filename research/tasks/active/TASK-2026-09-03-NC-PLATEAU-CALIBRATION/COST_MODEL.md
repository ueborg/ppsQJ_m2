# COST_MODEL — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Implemented in `tools/cost_model.py`, re-derived at preflight time, and printed
by every arm's preflight before anything is queued.

**Every rate is a per-clone-window wall time actually recorded by a completed
Ruche job of this identical code path.** Never a requested `--time`, never a
laptop extrapolation, never a predecessor's projection. Provenance: the 1 896
raw result JSONs enumerated in `EXISTING_POPULATION_INVENTORY.csv`.

Labels `[E]` `[I]` `[C]` `[J]`.

---

## 1. `K` is exact, not estimated

```
delta_tau = dtau_mult / (2 * lambda * (L - 1))        alpha == lambda on this cut
K = n_steps = ceil(T / delta_tau) = ceil(2 lambda (L-1) T / dtau_mult)
```

`[E]` `support/instrumented.py` lines 127–128 verbatim. `[E]` Verified against
the `n_steps` every completed run recorded for itself: **exact in all 1 896
cases**, at every `(L, lambda, dtau_mult)` in the corpus. `[E]` Campaign E's
`K = 816 / 408 / 204` is recomputed from this, not quoted from the design.

`[E]` Note `delta_tau` as *recorded by the sampler* is the actual step `T/K`,
not the nominal `dtau_mult/(2 lambda (L-1))` it was derived from — the `ceil`
rounds it down. The smoke test asserts this, because recording the nominal value
would trap anyone reconstructing the schedule from a result file.

## 2. Measured rates

Per-clone-window wall time, ms. `n` is the number of completed populations
behind each rung.

| L | `N_c` | n | median | p90 | **max (adopted)** |
|---:|---:|---:|---:|---:|---:|
| 32 | 1024 | 408 | 1.501 | 1.735 | **1.922** |
| 48 | 1024 | 408 | 2.881 | 3.237 | **3.457** |
| 64 | 1024 | 624 | 4.841 | 5.370 | **5.769** |
| 64 | 2048 | 72 | 4.788 | 4.888 | **5.075** |
| 96 | 128 | 32 | 11.705 | 12.320 | **12.426** |
| 96 | 256 | 32 | 10.120 | 11.354 | **11.534** |
| 96 | 512 | 48 | 11.510 | 11.907 | **11.989** |
| 128 | 64 | 64 | 27.176 | 28.967 | **29.506** |
| 128 | 128 | 64 | 26.805 | 28.483 | **28.683** |
| 128 | 256 | 64 | 21.522 | 22.423 | **23.998** |
| 128 | 512 | 48 | 23.416 | 24.874 | **25.080** |
| 128 | 1024 | 32 | 27.898 | 28.137 | **28.260** |

`[J]` The **max**, not the median, is adopted: `--time` protects the slowest task
in an array, not the typical one.

## 3. The correction that matters — the `N_c` direction reversed

`[E]` `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA` had three `L = 128` rungs
(27.18, 26.81, 21.52 at `N_c = 64, 128, 256`), read them as small-batch
inefficiency "gone by `N_c ~ 256`", and extrapolated every larger `N_c`
**flat from the `N_c = 256` rate**. It wrote, correctly for its own evidence:
*"if it is wrong it is wrong in the optimistic direction."*

`[E]` It is wrong, and in that direction. Two rungs have since completed and the
rate turns back up: 23.42 at 512 and **27.90** at 1024. Flat-from-256 predicts
21.52 at `N_c = 1024` against a measured 27.90 — **30 % low**, in exactly the
regime this campaign enters.

`[E]` A log-log fit over the three rungs where the small-batch regime is over
(`N_c >= 256`) gives

```
rate ~ N_c ** G ,   G = 0.1871
```

`[E]` `G` is applied to every extrapolation **above the largest measured rung at
each `L`**. `[J]` It is deliberately conservative where it is applied outside its
own `L`: at `L = 96` the three measured rungs show no trend, and at `L = 64` the
measured `1024 -> 2048` step is slightly **negative** (5.769 → 5.075 worst-case).
`G` is applied anyway. A cost model that is wrong should be wrong upward.

`[E]` The rate is also made **monotone non-decreasing in `N_c`**: the measured
envelope wobbles (`L = 128` dips at 256; `L = 64` dips at 2048), and a dip is not
a licence to request less wall time for a larger population. `rate()` returns
the running maximum over all measured rungs at or below the request, and
extrapolates from that.

`[E]` **Small-batch penalty below the smallest measured rung.** Going down in
`N_c` the corpus shows the rate rising: `L = 128` gives 29.506/23.998 = 1.230 at
`N_c = 64` and 1.195 at 128; `L = 96` gives 1.077 at 128. Campaign E runs at
`N_c = 64` and 256 at `L = 64`, where the smallest measured rung is 1024, so a
penalty of 1.30 / 1.15 is applied rather than assumed away.

**Adopted `rate(L, N_c)`, ms per clone-window:**

| L | 64 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2.499 | 2.210 | 2.018 | 1.922 | 2.188 | 2.491 | 2.836 |
| 48 | 4.494 | 3.976 | 3.630 | 3.457 | 3.936 | 4.481 | 5.101 |
| 64 | 7.500 | 6.634 | 6.057 | 5.769 | 5.769 | 6.568 | 7.477 |
| 96 | — | 12.426 | 12.426 | 14.147 | 16.106 | 18.336 | 20.875 |
| 128 | 29.506 | 29.506 | 29.506 | 29.506 | 33.592 | 38.243 | 43.539 |

`[C]` **Where this model is a conjecture and not a measurement**: every entry at
`N_c` above the largest measured rung — `L = 64` above 2048, `L = 96` above 512,
`L = 128` above 1024. `G` was fitted at one `L` on three points and is being
used at three `L`. If it is wrong it is most likely wrong at `L = 64`,
`N_c = 8192`, which is the furthest extrapolation (two doublings) and where the
measured local trend points the other way. The ±40 % pessimistic band exists for
this and `--time` is set from the pessimistic figure.

## 4. Memory — a defect found in the inherited model

`[E]` Every predecessor package sized `--mem` from

```
peak = 128 + 2 * N_c * ((2L)^2 * 8 + (2L) * L * 16)
```

and the coefficient 2 **was never checked against a running process**.
`TASK-2026-09-01-SMCRUCHE-READY` describes its output as *"the measured 732 MB
peak"*; 732 MB is exactly what that formula returns for `L = 96, N_c = 512`, and
no `MaxRSS` from any Ruche job appears anywhere in this repository.

`[E]` `tools/mem_probe.py` reads `ru_maxrss` from a real run of the bundled
sampler. These are the first peak-RSS measurements of this sampler in the
repository — 15 cells, including every large cell this campaign runs. Where a
cell was probed more than once, every probe is listed and **the maximum is
adopted**:

| L | `N_c` | measured peak MB | old formula MB | old formula is |
|---:|---:|---:|---:|---|
| 32 | 64 | 90.8 | 136 | over |
| 32 | 256 | 149.2 | 162 | over |
| 32 | 1024 | 276.4 | 262 | **under** |
| 32 | 2048 | 592.1 | 396 | **under 1.49×** |
| 32 | 4096 | 1063.9 | 665 | **under 1.60×** |
| 32 | 8192 | 1709.5 | 1202 | **under 1.42×** |
| 64 | 128 | 236.3 | 195 | **under** |
| 64 | 512 | 566.0 | 396 | **under 1.43×** |
| 64 | 2048 | 1694.2 | 1202 | **under 1.41×** |
| 64 | 4096 | 2032.6, **2747.1** | 2276 | **under 1.21× on the higher probe** |
| 64 | 8192 | 3547.0, **4593.8** | 4423 | **under 1.04× on the higher probe** |
| 96 | 128 | 430.6 | 279 | **under 1.54×** |
| 96 | 1024 | 2006.3, **2139.8** | 1336 | **under 1.60×** |
| 96 | 2048 | 2200.8 | 2544 | over |
| 128 | 2048 | 3482.5, 3521.7, **6275.9** | 4423 | **under 1.42× on the highest probe** |

`[E]` `L = 64`, `N_c = 2048` — the cell `MOCK-PRODUCTION/mockL64nc2048` ran with
`--mem=2G` — really needs 1694 MB, i.e. **21 % headroom**, not the ~1.7× its own
comment implied. `[J]` It never broke and it was closer to breaking than anyone
knew.

`[E]` The adopted model takes, per cell, the larger of the direct measurement
(where one exists), a conservative `K_MEM = 4.5` per-clone model (where one does
not), and the old formula **always, as a floor**, so this campaign never requests
less than a predecessor did for a comparable cell. The window-indexed genealogy
arrays (`anc_matrix` and `idxs_history`, both `K x N_c` intp) are added
analytically, because a short-`T` probe cannot see them.

`[E]` **Every immediate arm's memory is now measurement-based.** The table below
gives the model, the request, and whether that cell's peak was probed directly:

| arm | `L`, `N_c` | probes, MB (max adopted) | model MB (with genealogy) | request | basis |
|---|---|---|---:|---|---|
| `A_L64_nc2048_topup` | 64, 2048 | 1694 | 1708 | 3G | **measured** |
| `A_L64_nc4096` | 64, 4096 | 2033, **2747** | 2749 | 4G | **measured**, 2 probes |
| `A_L64_nc8192` | 64, 8192 | 3547, **4594** | 4647 | **7G** | **measured**, 2 probes |
| `B_L64_cross_nc512` | 64, 512 | 566 | 569 | 1G | **measured** |
| `B_L64_cross_nc1024` | 64, 1024 | — | 665 | 2G | modelled |
| `B_L64_cross_nc2048` | 64, 2048 | 1694 | 1707 | 3G | **measured** |
| `B2_L32_*` | 32, 512–2048 | 149–592 | ≤595 | 1G | **measured** |
| `B2_L48_*` | 48, 512–2048 | — | ≤1515 | 1–2G | modelled |
| `C_L96_nc1024` | 96, 1024 | 2006, **2140** | 2155 | 3G | **measured**, 2 probes |
| `C_L96_nc2048` | 96, 2048 | 2201 | 2574 | 4G | **measured**, 1 probe (floor binds) |
| `D_L128_nc2048` | 128, 2048 | 3483, 3522, **6276** | 6330 | **9G** | **measured**, 3 probes |
| `E_L64_dtau_nc*` | 64, 64–256 | 90–149 (`L = 32` proxy) | ≤240 | 1G | modelled |

`[E]` **Repeated probes of the same cell are NOT reproducible, and at every cell
probed twice the SECOND probe came in HIGHER.** That is the second finding here:

| cell | probes, MB | spread | note |
|---|---|---:|---|
| `L = 128`, `N_c = 2048` | 3 482.5 · 3 521.7 · **6 275.9** | **1.80** | three probes; the first two agree to 1.1 % and are still 1.80× below the third |
| `L = 64`, `N_c = 8192` | 3 547.0 · **4 593.8** | 1.30 | |
| `L = 64`, `N_c = 4096` | 2 032.6 · **2 747.1** | 1.35 | |
| `L = 96`, `N_c = 1024` | 2 006.3 · **2 139.8** | 1.07 | |
| `L = 96`, `N_c = 2048` | 2 200.8 | — | probed once; a second run's output line was truncated by a concurrent write and its peak is unrecoverable, so it is **not** counted |

`[E]` **Every cell probed more than once varies.** The `L = 128` triple carries
the sharper lesson: two probes agreeing to 1.1 % were still 1.80× below a third,
so **agreement between two probes is not evidence of a bound either**.

`[I]` Nothing about the sampler changed between any of them. `ru_maxrss` is a
high-water mark over the whole process and depends on when the allocator happens
to release the transient copies selection makes.

`[I]` **A single probe of a cell is not a bound.** Treating one as a bound is how
a 31-hour job dies at hour 20 with OOM. The model therefore takes the **maximum
over probes** at each cell, keeps the old formula as a floor, and applies the
1.35x margin on top of both. `D_L128_nc2048` requests **9G** — not the 6G the
lower probe alone would have justified, and not the 14G a purely model-based
estimate would have asked for.

`[J]` Had only the first probe been taken at each cell, this package would have
shipped **6G against a cell observed to reach 6.3 GB**, and 6G rather than 7G at
`A_L64_nc8192`. The check that caught it was running the probe twice, and it is
the single most useful thing in this section.

`[E]` **Honest limitation.** These are macOS `ru_maxrss` and the cluster is
Linux with a different allocator. `[J]` That is why the old formula is kept as a
floor and a 1.35× margin sits on top — under either allocator the request is
above every observation. `[E]` The first thing worth doing on the day is
`sacct -j <jobid> --format=JobID,MaxRSS` on one completed `D_L128_nc2048` task:
it would be the first `MaxRSS` measurement of this sampler on the cluster in
existence, it settles the question for the whole programme, and it costs
nothing. It is in the runbook.

`[E]` Three cells in the immediate group remain **modelled** rather than
measured — `B_L64_cross_nc1024`, the four `B2_L48_*` arms, and campaign E's
`N_c = 64`/`256` cells. All three are small (model ≤ 1 515 MB, request 1–2G)
and all sit well inside the conservative branch.

## 5. Slurm memory-unit parsing, re-verified

`[E]` `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` asked for this to be checked
again. The parser is `preflight.gib()` and its unit table is re-run by
`tools/negative_controls.py`:

| `--mem` | parsed GiB | note |
|---|---:|---|
| `2G`, `2g` | 2.000000 | |
| `512M` | 0.500000 | |
| `600M` | 0.585938 | the value the predecessor's parser raised on |
| `2048` | **2.000000** | **no suffix means MEGABYTES**; the old parser read 2048 GiB and failed OPEN |
| `1024K` | 0.000977 | |
| `1T` | 1024.0 | |
| `` , `nonsense` | 0.0 | fails closed |

`[E]` Negative control **N12** additionally injects `--mem=200` into a real arm
and requires the preflight to reject it.

## 6. `--time`, and how it is chosen

`[E]` `--time >= 1.6 x` the **pessimistic** slowest task, snapped up to a
readable limit. The partition is then chosen to fit the time — never the other
way round. `[E]` The preflight recomputes both and fails on a mismatch, and
negative control **N10** injects a short `--time` and requires rejection.

`[E]` **Partition rule**: `cpu_med` if `--time` fits its 4 h `MaxTime`,
`cpu_long` otherwise. **`cpu_short` is never used at any `--time`.**
`TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/SCHEDULER_DECISION.md` records that it
is effectively serialised for this account by `QOSMaxJobsPerUserLimit`, so its
`%N` concurrency cap is not real. Negative control **N13** injects `cpu_short`
and requires rejection.

## 7. Per-arm cost — the immediate group

`PESSIMISTIC = 1.40`. `PACKING = 1.15`. Elapsed is at `%64` and **excludes queue
wait**, which will dominate the short arms.

| campaign | arm | tasks | rate ms | core-h | pess | slowest h | pess | elapsed h | partition | `--time` | `--mem` |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---|
| A | `A_L64_nc2048_topup` | 24 | 5.769 | 32.1 | 45.0 | 1.34 | 1.88 | 1.34 | cpu_med | 03:00:00 | 3G |
| A | `A_L64_nc4096` | 48 | 6.568 | 146.3 | 204.9 | 3.05 | 4.27 | 3.05 | cpu_long | 08:00:00 | 4G |
| A | `A_L64_nc8192` | 48 | 7.477 | 333.2 | 466.5 | 6.94 | 9.72 | 6.94 | cpu_long | 18:00:00 | 6G |
| B | `B_L64_cross_nc512` | 336 | 6.057 | 90.8 | 127.2 | 0.29 | 0.40 | 1.73 | cpu_med | 01:00:00 | 1G |
| B | `B_L64_cross_nc1024` | 264 | 5.769 | 136.0 | 190.4 | 0.55 | 0.77 | 2.74 | cpu_med | 02:00:00 | 2G |
| B | `B_L64_cross_nc2048` | 336 | 5.769 | 346.1 | 484.5 | 1.10 | 1.53 | 6.58 | cpu_med | 03:00:00 | 3G |
| B2 | `B2_L32_nc512` | 336 | 2.018 | 7.5 | 10.5 | 0.02 | 0.03 | 0.14 | cpu_med | 01:00:00 | 1G |
| B2 | `B2_L32_nc1024` | 264 | 1.922 | 11.2 | 15.7 | 0.04 | 0.05 | 0.23 | cpu_med | 01:00:00 | 1G |
| B2 | `B2_L32_nc2048` | 336 | 2.188 | 32.5 | 45.5 | 0.10 | 0.14 | 0.62 | cpu_med | 01:00:00 | 1G |
| B2 | `B2_L48_nc512` | 336 | 3.630 | 30.5 | 42.7 | 0.10 | 0.14 | 0.58 | cpu_med | 01:00:00 | 1G |
| B2 | `B2_L48_nc1024` | 264 | 3.457 | 45.7 | 63.9 | 0.18 | 0.25 | 0.92 | cpu_med | 01:00:00 | 2G |
| B2 | `B2_L48_nc2048` | 336 | 3.936 | 132.3 | 185.2 | 0.42 | 0.59 | 2.51 | cpu_med | 01:00:00 | 2G |
| C | `C_L96_nc1024` | 24 | 14.147 | 89.0 | 124.7 | 3.71 | 5.19 | 3.71 | cpu_long | 12:00:00 | 3G |
| C | `C_L96_nc2048` | 24 | 16.106 | 202.7 | 283.8 | 8.45 | 11.83 | 8.45 | cpu_long | 24:00:00 | 4G |
| **D** | `D_L128_nc2048` | 16 | 33.592 | **502.4** | **703.3** | **31.40** | **43.96** | **31.40** | cpu_long | 72:00:00 | 9G |
| E | `E_L64_dtau_nc64` | 144 | 7.500 | 9.1 | 12.8 | 0.11 | 0.15 | 0.33 | cpu_med | 01:00:00 | 1G |
| E | `E_L64_dtau_nc256` | 144 | 6.634 | 32.3 | 45.3 | 0.39 | 0.54 | 1.16 | cpu_med | 01:00:00 | 1G |
| | **total** | **3 280** | | **2 180.0** | **3 051.9** | | | | | | |

`[E]` **Cost by campaign**: B 26.3 % · A 23.5 % · D 23.0 % · C 13.4 % ·
B2 11.9 % · E 1.9 %.

`[J]` **What dominates.** No single campaign dominates: A, B and D are within
3 percentage points of each other. But **D is 16 tasks** and A+B are 1 056, so
per *answer* D is by far the most expensive — 502 core-hours for one screening
number that cannot certify convergence. `[J]` E is 1.9 % of the campaign and is
the only arm whose **both** outcomes kill a mechanism.

`[E]` Campaign D's ~31.4 h central / ~44.0 h pessimistic per population is
independently consistent with the ~31 h / low-40s figure the brief carried, and
was reached here from the `N_c = 1024` measured rate plus `G`, not inherited.

## 8. Conditional-group cost, if every arm were released

| arm | tasks | core-h | pess | slowest task | `--time` | `--mem` |
|---|---:|---:|---:|---:|---|---|
| `cond_D2_L128_nc4096` | 8 | 572 | 801 | **71.5 h** (100.1 pess) | 168:00:00 | 26G |
| `cond_M96_nc1024` | 108 | 308 | 432 | 3.2 h | 08:00:00 | 3G |
| `cond_M96_nc2048` | 108 | 702 | 983 | 7.3 h | 18:00:00 | 4G |
| `cond_M128_nc2048` | 72 | 1 739 | 2 435 | 27.3 h | 72:00:00 | 9G |
| `cond_M128_nc4096` | 72 | **3 960** | 5 545 | **62.1 h** (86.9 pess) | 144:00:00 | 26G |
| `cond_LOWZ_nc64` | 48 | 2.6 | 3.6 | 0.05 h | 01:00:00 | 1G |
| `cond_LOWZ_nc256` | 48 | 9.2 | 12.9 | 0.19 h | 01:00:00 | 1G |
| | **464** | **7 294** | | | | |

`[E]` **They will not all be released.** The two `M96` arms are the same scan at
two `N_c` and are mutually exclusive; so are the two `M128` arms.

`[E]` **A hard limit worth seeing before it is hit.** `cond_D2_L128_nc4096` is
`--time=168:00:00`, which is `cpu_long`'s `MaxTime` **exactly**. Its pessimistic
slowest task is 100 h, so there is 1.68× headroom and **no room to add any**. A
single task that overruns is lost entirely, with no checkpointing in this code
path. `[I]` `L = 128` at `N_c = 4096` is at the edge of what this cluster can run
as one job, and `N_c = 8192` at `L = 128` is **not runnable at all** under this
architecture. `[J]` That, more than cost, is why the locator route matters at
`L = 128`.
