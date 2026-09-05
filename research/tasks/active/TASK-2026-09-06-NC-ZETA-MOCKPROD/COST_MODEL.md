# COST_MODEL — TASK-2026-09-06-NC-ZETA-MOCKPROD

Implemented in `tools/cost_model.py`, **re-derived from raw data by every
preflight** (`P10`), and printed in every `submit.slurm` header. Labels `[E]`
`[I]` `[C]` `[J]`.

Brief §13: *use measured Ruche runtime scaling from the recent campaigns rather
than the old incorrect `N_c^0.1871` model.* That instruction is discharged in
§3, with the refutation.

---

## 1. The identity

```
K = n_steps  = ceil(2 * lambda * (L-1) * T / dtau_mult),   T = L
task_seconds = K * N_c * rate(L, N_c, lambda, zeta) / 1000
rate         = rate35(L, N_c) * small_batch(N_c) * rho(zeta) * f_lam(lambda, zeta)
```

`[E]` `K` is **exact**, not estimated: `support/instrumented.py` lines 127–128,
verified against the recorded `n_steps` of every completed population in the
corpus.

`[I]` `K` depends on `lambda`, and `lambda` moves with `zeta`, so **cost and
statistical difficulty are entangled** and run in opposite directions — see §7.

## 2. Measurement versus conjecture, line by line

| input | status |
|---|---|
| `K` | **MEASURED / exact** |
| `rate35(L, N_c)`, `L in {32,48,64}` | **MEASURED on Ruche**, this code path, `zeta = 0.35`, `dtau_mult = 6` |
| `small_batch(N_c)` at `N_c = 128, 256` for `L = 32, 48` | **CONJECTURE**, applied upward |
| `rho(zeta)` | **MEASURED RATIO**, local, ratios only; absolute level always from the Ruche table |
| `f_lam(lambda, zeta)` | **INFERRED**, and the weakest input in this package |
| memory | **MEASURED envelope x 1.45**; still no Ruche `MaxRSS` anywhere in this repository |

## 3. `rate35(L, N_c)` — and the refutation of `N_c^0.1871`

`[E]` Per-clone-window wall time, ms, from every completed `zeta = 0.35`,
`dtau_mult = 6` population in `research/tasks/**/results/`:

| `L` | `N_c` rungs measured | worst-case rate at each rung | adopted |
|---:|---|---|---:|
| 32 | 512, 1024, 2048 | 1.951, 1.922, 1.872 | **1.951** |
| 48 | 512, 1024, 2048 | 3.303, 3.457, 3.391 | **3.457** |
| 64 | 64, 256, 512, 1024, 2048, 4096, 8192 | 5.111, 5.011, 5.733, 5.769, 5.693, 5.048, 5.107 | **5.769** |

`[J]` The **maximum**, not the median, is adopted: `--time` protects the slowest
task in an array, not the typical one.

`[E]` **`TASK-2026-09-03-NC-PLATEAU-CALIBRATION` fitted `rate ~ N_c^G`,
`G = 0.1871`, on exactly three `L = 128` rungs (21.52, 23.42, 27.90 at
`N_c = 256, 512, 1024`) and applied it above the largest measured rung at every
`L`.** `[E]` That campaign has since returned the rungs that test it:

| cell | `G = 0.1871` predicts | **measured** | error |
|---|---:|---:|---:|
| `L = 64`, `N_c = 4096` | 6.57 ms | **5.05** | +30 % high |
| `L = 64`, `N_c = 8192` | 7.48 ms | **5.11** | +46 % high |
| `L = 128`, `N_c = 2048` | 31.7 ms | **22.45** | +41 % high |

`[E]` **The law is refuted by the data that was collected to test it.** Over
`N_c = 512..8192` at `L in {32, 48, 64}` the measured rate is flat to within
±9 % with no monotone trend, and at `L = 64` it is slightly *lower* at 8192 than
at 1024. `[I]` `G = 0.1871` was an `L = 128`-specific three-point artifact.
`[I]` It erred **upward**, which is the safe direction for `--time` and the
expensive direction for planning a campaign — the previous programme's error ran
the other way and cost a campaign, so both directions are now on record.

`[E]` `cost_model.refit()` re-derives this table on every preflight run and
prints the `G` comparison, so the refutation is re-checked rather than
remembered.

`[E]` **Small-batch penalty.** `N_c = 128` and `256` are unmeasured at `L = 32`
and `48`. The corpus shows the rate rising as `N_c` falls at large `L`
(`L = 128`: 1.230 at `N_c = 64`, 1.195 at 128; `L = 96`: 1.157 at 128), so 1.25
and 1.15 are applied. `[E]` **The investigator disagrees** and would carry the
flat plateau value down, on the grounds that at `L = 64` the measured `N_c = 64`
and `256` rates (5.111, 5.011) sit *below* the adopted `L = 64` constant.
`[J]` The penalty is kept: it errs upward, it affects only the four cheapest
arms in the design (`M_z010_nc128/256`, `M_z020_nc128/256`, 36 core-hours
between them), and no `--time` in the package depends on which reading is right.
The disagreement is recorded rather than resolved.

## 4. `rho(zeta)` — the only `zeta`-resolved timing that exists

`[E]` No Ruche timing off `zeta = 0.35` exists anywhere in this repository. The
only `zeta`-resolved timing is
`TASK-2026-08-11-ALGRD/results/b0_L{32,64,96,128}.json`
(`sec_per_clone_window` at `zeta in {0.05, 0.15, 0.30, 0.70}`). It is local, its
`N_c` is confounded with `L`, and its `lambda` follows the historical
`0.51*sqrt(zeta)` line.

`[E]` Only the **ratio at fixed `L`** is used (`N_c` is constant within each `L`
block, so the ratio is clean), log-log interpolated in `zeta`, and **maximised
over the four `L`**, so the model errs upward:

| `zeta` | `L=32` | `L=64` | `L=96` | `L=128` | **adopted** |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.4486 | 0.4033 | 0.3651 | 0.3419 | **0.4486** |
| 0.20 | 0.6391 | 0.6041 | 0.5913 | 0.5868 | **0.6391** |
| 0.70 | 2.3260 | 2.3720 | 2.1927 | 2.0850 | **2.3720** |

`[E]` Independently re-derived here from the raw JSON and agreeing with
`TASK-2026-09-05-NC-ZETA-CALIBRATION`'s table to better than 0.1 %.
`[C]` The ALGRD probes use tiny `n_steps` and an intercept fit at `N_c <= 500`,
a different measurement regime from production. Treat `rho` as directional.

`[E]` **Correction: `rho(0.70)` errs DOWNWARD, and the "maximum over `L` errs
upward" argument does not cover it.** The red team found that the log-log
`rate(zeta)` curve is **convex** — successive slopes 0.42, 0.64, 1.22 — so the
chord used to interpolate the `zeta = 0.35` denominator sits **above** the true
curve, which inflates the denominator and deflates the ratio. `rho(0.70)` is
understated by roughly 7–9 %.
`[J]` Taking the maximum over the four `L` errs upward; the log-log
interpolation of the denominator errs downward; they are different operations
and the first does not compensate for the second. The net effect on
`zeta = 0.70` is that its arms are **under**-costed by up to ~9 %, which is
absorbed by the `PESSIMISTIC = 1.40` band and by `--time` margins of 2.2x or
better, but it is not the direction §2 implies and the earlier wording was
wrong.

## 5. `f_lam` — the confound correction, and the weakest input here

`[E]` `rho` is a rate ratio measured **along** `lambda = 0.51*sqrt(zeta)`. This
task's grids are not on that line, and at fixed `zeta = 0.35` the Ruche corpus
shows the rate **falling** with `lambda` over `0.1932..0.3532`:

| cell | log-log slope, lead (LSQ over 21 points) | investigator (endpoints) |
|---|---:|---:|
| `L=32, N_c=1024` | −0.434 | −0.48 |
| `L=48, N_c=1024` | −0.367 | — |
| `L=64, N_c=1024` | −0.357 | −0.53 |
| `L=64, N_c=2048` | −0.360 | — |

`[E]` **A draft of this file adopted `A_LAM = -0.35` under a comment claiming it
was the conservative choice. That is backwards** — the grids sit *below* the
reference line, so `(lambda/lambda_ref)^A_LAM > 1` and grows as `A_LAM` becomes
*more* negative, making −0.35 the *least* conservative estimate available. Caught
by `agent_reports/numerics.md` Q2. **`A_LAM = -0.50` is adopted**: the
investigator's independent estimate, and the upward-erring one.

`[E]` Correction size at the extremes:

| cell | `lambda / lambda_ref` | `f_lam` |
|---|---:|---:|
| `zeta=0.10, lambda=0.040` | 0.248 | **2.008** |
| `zeta=0.10, lambda=0.160` | 0.992 | 1.004 |
| `zeta=0.20, lambda=0.080` | 0.351 | 1.688 |
| `zeta=0.70, lambda=0.250` | 0.586 | 1.306 |
| `zeta=0.70, lambda=0.520` | 1.219 | 0.906 |

`[C]` **At `zeta = 0.10, lambda = 0.040` this is a 4x extrapolation below the
window where the −0.5 slope was calibrated (0.19–0.35).** It is a plausible
direction, not a measured number. `[E]` The cap `F_LAM_CAP = 2.5` sits **above**
the largest value the design reaches (2.008) so that it cannot silently truncate
this campaign's own worst cell — a cap that binds is a cap that hides.

`[J]` The exposure is small in absolute terms: the cells where `f_lam` is
largest are the cheapest in the campaign, and the arms it most affects
(`M_z010_*`, 102 core-hours, 5.1 % of the design) carry `--time` margins of
2.2x or better. The investigator's recommended probe — one short local timing
run at `zeta = 0.10, L = 64, N_c = 512, lambda = 0.040` — is carried to
`RECOMMENDATION.md` as a Gate-A option, not run here.

## 6. Memory

`[E]` `mem = 1.45 * (128 + 2 * N_c * ((2L)^2*8 + (2L)*L*16) / 1e6)` MB. The 1.45
exists because the inherited formula **under-predicts** real peak RSS by up to
1.6x, measured directly at 15 cells by `NC-PLATEAU-CALIBRATION`.
`[E]` Requests are `>= 1.35 x` that model (`P5`). `[I]` `zeta` does not enter
memory: the genealogy term is tens of MB against a GB-scale base.

`[E]` **Every memory figure in this programme still descends from local
`ru_maxrss` probes.** There is no Ruche `MaxRSS` anywhere in this repository.
`RUCHE_RUNBOOK.md` §5 asks for the one `sacct` line that would end that, and it
costs nothing.

## 7. Per-arm cost

`PESSIMISTIC = 1.40`, `PACKING = 1.15`, concurrency cap `%64`. Elapsed is a FIFO
list-scheduling simulation of the packs onto 64 slots, floored by the slowest
pack and by the throughput bound. **Queue wait is excluded from every figure and
is expected to dominate the short arms.**

| arm | pops | packed tasks | core-h | pessimistic | slowest population | slowest pack | elapsed | partition | `--time` | `--mem` |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `M_z010_nc128` | 432 | 23 | 4.0 | 5.5 | 1.5 min | 11.4 min | 0.22 h | cpu_med | `00:45:00` | `1G` |
| `M_z010_nc256` | 432 | 41 | 7.3 | 10.2 | 2.8 min | 12.4 min | 0.24 h | cpu_med | `00:45:00` | `1G` |
| `M_z010_nc512` | 432 | 65 | 12.6 | 17.7 | 4.8 min | 14.4 min | 0.28 h | cpu_med | `00:45:00` | `1G` |
| `M_z010_nc1024` | 432 | 110 | 25.3 | 35.4 | 9.6 min | 19.2 min | 0.60 h | cpu_med | `00:45:00` | `2G` |
| `M_z010_nc2048` | 432 | 213 | 50.6 | 70.8 | 19.2 min | 19.2 min | 1.09 h | cpu_med | `00:45:00` | `3G` |
| `M_z020_nc128` | 432 | 47 | 8.7 | 12.2 | 3.2 min | 12.9 min | 0.25 h | cpu_med | `00:45:00` | `1G` |
| `M_z020_nc256` | 432 | 84 | 16.1 | 22.5 | 5.9 min | 14.4 min | 0.43 h | cpu_med | `00:45:00` | `1G` |
| `M_z020_nc512` | 432 | 124 | 28.0 | 39.2 | 10.3 min | 19.7 min | 0.61 h | cpu_med | `00:45:00` | `1G` |
| `M_z020_nc1024` | 432 | 226 | 56.0 | 78.3 | 20.6 min | 20.6 min | 1.19 h | cpu_med | `01:00:00` | `2G` |
| `M_z020_nc2048` | 432 | 298 | 111.9 | 156.7 | 41.3 min | 41.3 min | 2.40 h | cpu_med | `02:00:00` | `3G` |
| `M_z070_nc128` | 432 | 237 | 67.3 | 94.2 | 23.1 min | 23.8 min | 1.39 h | cpu_med | `01:00:00` | `1G` |
| `M_z070_nc256` | 432 | 318 | 123.9 | 173.4 | 42.5 min | 42.5 min | 2.71 h | cpu_med | `02:00:00` | `1G` |
| `M_z070_nc512` | 432 | 352 | 215.4 | 301.6 | 73.9 min | 73.9 min | 4.62 h | cpu_med | `03:00:00` | `1G` |
| `M_z070_nc1024` | 432 | 408 | 430.8 | 603.1 | 147.9 min | 147.9 min | 9.22 h | cpu_long | `06:00:00` | `2G` |
| **`M_z070_nc2048`** **CONDITIONAL** | 432 | 432 | **861.6** | **1206.3** | 295.8 min | 295.8 min | 18.41 h | cpu_long | `12:00:00` | `3G` |
| `E_dtau_z010` | 32 | 11 | 2.5 | 3.5 | 7.5 min | 15.1 min | 0.29 h | cpu_med | `00:45:00` | `1G` |

```
COMMITTED     15 arms   6 080 populations   2 557 array tasks   1 160.3 core-h  (1 624.4 pessimistic)
CONDITIONAL    1 arm      432 populations     432 array tasks     861.6 core-h  (1 206.3 pessimistic)
FULL DESIGN                                                     2 021.9 core-h
```

## 8. Where the cost is, and the one rung held back

`[E]` Cost by `zeta` over the full design: **`zeta = 0.70` is 84.0 %**
(1 699 core-h), `zeta = 0.20` is 10.9 % (221), `zeta = 0.10` is 5.1 % (102), the
discretisation control is 0.12 % (2.5).

`[E]` **`M_z070_nc2048` alone is 861.6 core-hours = 42.6 % of the full design**,
for one rung at one `zeta`. Brief §4 permits making exactly this rung
conditional provided the saved core-hours are reported clearly:

> **Holding `M_z070_nc2048` saves 861.6 core-hours (1 206.3 pessimistic), which
> is 42.6 % of the full design. The committed campaign is 1 160.3 core-hours.**

`[E]` It is shipped, complete and preflight-clean, in `conditional/`, behind the
interlock in `CONDITIONAL_SUBMISSION.md`. `[J]` The interlock is a real decision
point, not a formality: if `512 -> 1024` at `zeta = 0.70` already reads
`ROUGHLY STABLE`, the 2048 rung buys a confirmation the survey does not need,
and the researcher can spend those 862 core-hours on the `R` that the
`zeta = 0.35` anchor showed to be the actually-binding budget.

`[I]` **Cost runs opposite to statistical difficulty.** `rho` says a population
at `zeta = 0.70` costs 5.3x one at `zeta = 0.10` per clone-window, and `lambda`
is ~3.9x larger there so `K` is larger too — while the tilt prefactor
`(1-zeta)^2` says the sampler has *least* statistical work to do at high `zeta`.
`[J]` So the cheapest `zeta` to survey is the one most likely to need a large
`N_c`, and the most expensive is the one most likely not to. That asymmetry is
the single most useful planning fact in this file.

## 8b. The model is inflated overall, by about 25 %, and the ratios are not

`[E]` `RATE35_MAX` is the maximum per-clone-window rate attained anywhere at that
`L` — and in this corpus the maximum is attained at the corpus's **lowest**
`lambda`, because the rate falls with `lambda`. `f_lam` then applies a
`lambda` correction **again**. The red team measured the resulting double count
at roughly **25 %** beyond what the max-not-median convention alone would give.

`[E]` **Every absolute core-hour figure in this package is therefore high**, and
`[E]` **every ratio is unaffected**, because the same factor multiplies both
sides. `[J]` The kills of C2, C3 and C4 rest on ratios and are safe; the budget
headline is conservative, which is the direction a `--time` limit should err.
`[C]` It is not corrected here: doing so would change every `--time` in the
package after the red team had validated them, for a change that only makes the
campaign cheaper. F13 will measure it.

## 9. What this model does not claim

- `[E]` It is not an `L`-scaling law. Three per-`L` constants, each used only
  within or just outside its own measured span.
- `[E]` It says nothing about `L = 96` or `128`, or about `N_c > 2048`.
- `[E]` It does not fit `N_c^req(zeta)` and forbids anyone doing so from these
  numbers.
- `[C]` `f_lam` at `zeta = 0.10` is an extrapolation and is flagged as this
  package's weakest input, not smoothed over.
