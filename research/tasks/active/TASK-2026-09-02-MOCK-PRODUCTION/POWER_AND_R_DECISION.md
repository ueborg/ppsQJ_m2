# Choosing R — the power calculation, from the ACTUAL ARM-B data

TASK-2026-09-02-MOCK-PRODUCTION, brief §4 and §6.

The brief supplies `N_c = 1024, R = 24` as a default and says not to accept it
blindly. This file is the verification. **It concludes that R = 24 is correct
and needs no increase anywhere in the proposed lambda range.**

## 1. What ARM B actually measured

**[E]** Recomputed here from the 288 result JSONs of
`TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/armB`, frozen into
`frozen_inputs/armB_populations.csv` and reproducing the published ARM-B block
digit for digit (`INPUTS_LEDGER.md`):

| lambda | R | mean CMI | across-population sd | SEM | sd / I |
|---:|---:|---:|---:|---:|---:|
| 0.2932 | 96 | 0.40307 | 0.03658 | 0.00373 | 0.0908 |
| 0.3032 | 96 | 0.35316 | 0.02850 | 0.00291 | 0.0807 |
| 0.3132 | 96 | 0.31612 | 0.02598 | 0.00265 | 0.0822 |

The brief's stated figures — SEM ≈ 0.0027–0.0037 and neighbouring
`delta_lambda = 0.01` changes of ≈ 0.037–0.050 — are **confirmed exactly**:

| increment | d | SEM(d) at R=96 | r |
|---|---:|---:|---:|
| 0.2932 → 0.3032 | −0.04991 | 0.00473 | 10.55 |
| 0.3032 → 0.3132 | −0.03704 | 0.00394 | 9.41 |

The single most useful number in the table is the last column of the first one:
**the relative across-population spread is 0.084 ± 0.005 and is very nearly
constant in lambda.** That is what makes the projection below possible.

## 2. R = 24 at the three measured lambdas

Rescaling the measured variances by `sqrt(96/24) = 2`:

| increment | d | SEM(d) at R=24 | r |
|---|---:|---:|---:|
| 0.2932 → 0.3032 | −0.04991 | 0.00947 | **5.27** |
| 0.3032 → 0.3132 | −0.03704 | 0.00787 | **4.71** |

**[E]** Reducing to `R = 24` resolves the neighbouring points at 4.7–5.3 sigma.
The brief asked whether that is "comfortable". It is: five sigma on a
neighbouring grid step, with an error bar about a fifth of the step itself.

**Under the matched-R amendment this is not merely a projection — it is what
the primary analysis actually uses.** The three reused cells are cut to their
block A, so the reused points enter every primary statistic at `R = 24`, the
same as their ten new neighbours. Their full `R = 96` remains available as a
secondary high-precision mean. See `MATCHED_R_AMENDMENT.md`.

## 3. R = 24 across the whole 13-point grid

The shape of `|d(lambda)|` is taken from the corpus quadratic at `L = 64`
(`LAMBDA_GRID_DECISION.md`) and the *scale* of the noise from the measured
relative sd of 0.0845. Both inputs are named; neither is invented.

| increment | \|d\| | SEM(d), R=24 | r |
|---|---:|---:|---:|
| 0.2332 → 0.2432 | 0.0887 | 0.0217 | **4.09** |
| 0.2432 → 0.2532 | 0.0837 | 0.0196 | 4.28 |
| 0.2532 → 0.2632 | 0.0787 | 0.0176 | 4.48 |
| 0.2632 → 0.2732 | 0.0737 | 0.0157 | 4.69 |
| 0.2732 → 0.2832 | 0.0687 | 0.0140 | 4.92 |
| 0.2832 → 0.2932 | 0.0637 | 0.0124 | 5.15 |
| 0.2932 → 0.3032 | 0.0587 | 0.0109 | 5.40 |
| 0.3032 → 0.3132 | 0.0537 | 0.0095 | 5.66 |
| 0.3132 → 0.3232 | 0.0487 | 0.0082 | 5.91 |
| 0.3232 → 0.3332 | 0.0437 | 0.0071 | 6.15 |
| 0.3332 → 0.3432 | 0.0387 | 0.0061 | 6.34 |
| 0.3432 → 0.3532 | 0.0337 | 0.0052 | 6.46 |

The worst case is the low-lambda end, at **r ≈ 4.1**.

### How wrong that projection can be, checked where it is checkable

**[E]** At the two increments where ARM B measured the answer, the projection
gives r = 5.40 and 5.66 and the measurement gives 5.27 and 4.71 — so the
projection is optimistic by 2 % and 17 %. **[I]** Deflating the worst case by
the larger of the two errors gives a realistic floor of **r ≈ 3.4**, still well
clear of the M2 threshold of 2 and far from the "unnecessarily fine" regime.

**[C]** At `L = 32` and `L = 48` there is no measurement of either the slope or
the spread — the corpus does not reach below `L = 64`. The extrapolated corpus
slope grows slowly *with* `L` (−5.78 at 64 to −7.89 at 128), so at smaller `L`
it should be slightly shallower; the relative spread at fixed `N_c` and smaller
`L` should be equal or smaller, since a population of 1024 clones represents a
32-site chain at least as well as a 64-site one. Both effects are small and they
partly cancel. R = 24 is carried across all three `L` unchanged, and the
projection is recorded in `analysis_spec.yaml` under M2 so that a large
disagreement is itself a reportable outcome rather than a surprise.

**Verdict: R = 24. No increase is needed anywhere in the proposed lambda range,
and none is applied.** The goal was a realistic production-quality
configuration, not cosmetic error bars, and 4–6 sigma per grid step with 24
independent populations is exactly that.

## 4. R for the N_c = 2048 shape check (brief §6)

`Delta_N(lambda) = I_{N_c=2048}(lambda) − I_{N_c=1024}(lambda)` at the three
central lambdas. The `N_c = 1024` side is ARM B's existing `R = 96` and costs
nothing. The question is the `N_c = 2048` side.

**[E]** Across-population sd falls with `N_c`. From the completed ARM2 ladder at
`L = 128, lambda = 0.3032`: sd = 0.1995 / 0.1883 / 0.1343 at `N_c` = 64 / 128 /
256, i.e. roughly `N_c^-0.28` over that range. **[I]** One further doubling
therefore scales sd by ≈ 0.82.

| R | SEM(Delta_N) at the three lambdas | SEM of the end-to-end lambda variation | 2-sigma detectable variation |
|---:|---|---:|---:|
| 16 | 0.0084 / 0.0066 / 0.0060 | 0.0103 | 0.0206 |
| 21 | 0.0076 / 0.0059 / 0.0054 | 0.0093 | 0.0185 |
| **24** | **0.0072 / 0.0056 / 0.0051** | **0.0088** | **0.0176** |

Cost and schedule, from `tools/cost_model.py`:

| R | tasks | core-h | elapsed at %64 |
|---:|---:|---:|---:|
| 16 | 48 | 64.8 | 1.39 h |
| 21 | 63 | 85.1 | 1.53 h |
| **24** | **72** | **97.3** | **1.75 h** |

**R = 24 is chosen.** The decisive fact is that this arm is **not on the
critical path at any R in the brief's 16–24 range**: the campaign's long pole is
`mockL64` at 2.32 h, and even R = 24 leaves this arm finishing 34 minutes
earlier. So the trade is 32.5 core-hours against a 17 % tighter bound on the
lambda-dependence of `Delta_N`, with no wall-clock cost at all. **[J]** Taking
the precision is the right call when the schedule does not have to pay for it.

R = 24 also keeps the population count uniform across every `N_c = 1024` and
`N_c = 2048` cell in the campaign, which is one fewer thing for the bootstrap
and split-half machinery to special-case.

**Scale check.** **[I]** A detectable lambda-variation of 0.0176 should be read
against one lambda-grid step in CMI at `L = 64`, which the ARM-B data put at
0.037–0.050. So the arm can distinguish a `Delta_N` that tilts by more than
about 40 % of a grid step from one that does not. It cannot resolve a tilt much
smaller than that, and `analysis_spec.yaml` pre-registers verdict **C
(unresolved)** as a real possible outcome rather than a failure.

## 5. R for the matched N_c = 128 companion arm

`R = 48`. **[I]** At `N_c = 128` the across-population sd is larger — the
historical `R = 12` corpus at `L = 64` has a median SEM of 0.0206, implying
sd ≈ 0.071, about twice ARM B's — so `R = 48` returns SEM ≈ 0.0103, comparable
to the `N_c = 1024` arms' 0.006–0.009 and sufficient for the difference
`Delta = I_1024 − I_128` to be dominated by neither side.

`R = 48` also has two further uses. It contains **two disjoint `R = 24` blocks**,
so the matched-R primary comparison has a full block A while block B remains an
independent replication of the whole comparator curve. And it contains **four
disjoint `R = 12` subsets**, so the analysis can reproduce the historical
corpus's own precision by subsampling and ask directly whether `R = 12` is what
made the old scan look jagged (brief §12.4). Neither question can be asked at
`R = 12` itself.

`R = 48` is kept rather than cut to 24: only the *primary comparison* is matched
at 24. Nothing is recomputed and nothing is discarded — see
`MATCHED_R_AMENDMENT.md`.

The whole companion group costs 81.3 core-hours, 21 % of the campaign, and is
off the critical path. See `NC128_COMPANION_RATIONALE.md` for why it exists.
