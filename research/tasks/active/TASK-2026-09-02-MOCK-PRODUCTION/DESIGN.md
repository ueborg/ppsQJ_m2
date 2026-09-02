# Design

TASK-2026-09-02-MOCK-PRODUCTION — a numerical child task of
`TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA`.

**Terminal state: `READY_FOR_HUMAN_SUBMISSION`.** No agent submitted anything.
`research/RESOURCE_POLICY.md` §4 forbids it unconditionally; nothing in this
package contains an executable scheduler call.

---

## 1. The question

The predecessor answered it for one cell. **[E]** ARM B ran three lambdas at
`L = 64, N_c = 1024, R = 96` and its frozen criteria came back `F4 SUPPORTED`
(S1–S4 all pass), spacing verdict *approximately appropriate* at `r = 9.43`.
So a single high-population cell behaves.

The question now is one step up:

> **If we use a realistically large population, can we obtain an entire
> finite-size `CMI(lambda)` scan — three system sizes, thirteen lambdas — that
> is clean, reproducible, and suitable in character for the final production
> analysis?**

This is **algorithm validation**. It is not physics, and `L = 32, 48, 64` sit at
or below the programme's own corpus floor of `L = 64`. Nothing here may be read
as a phase boundary. `SUCCESS_CRITERIA.md` closes with that constraint and the
analysis prints it at the end of every run.

## 2. The fixed slice

`zeta = 0.35`, `T = L`, `dtau_mult = 6.0` (certified), systematic resampling,
`L in {32, 48, 64}`. **[E]** The production convention was inspected rather than
assumed: every arm in the predecessor uses `T = L` and `dtau_mult = 6.0`, and
`shared/preflight.py` now enforces `T == L` as a hard failure. The sampler is
byte-identical to the predecessor's (`PRODUCTION_PATH_UNCHANGED.md`); nothing in
it is changed.

## 3. The lambda grid

**13 points, `delta_lambda = 0.010`, from 0.2332 to 0.3532, identical at every
`L`.** Derived from where the measured `zeta = 0.35` corpus says the cross-`L`
ordering reverses — not from any critical law. The three ARM-B lambdas are grid
indices 6, 7, 8 and are reused rather than recomputed.

Full derivation, including the wave arithmetic for 13 versus 11 points and the
risk the grid does not cover: `LAMBDA_GRID_DECISION.md`.

## 4. The arms

| arm | L | N_c | λ | R | tasks | core-h | elapsed at %64 | why |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `mockL32` | 32 | 1024 | 13 | 24 | 312 | 12.1 | 0.22 h | the curve |
| `mockL48` | 48 | 1024 | 13 | 24 | 312 | 58.8 | 1.06 h | the curve |
| `mockL64` | 64 | 1024 | **10** | 24 | 240 | 129.3 | **2.32 h** | the curve, minus the three ARM-B lambdas |
| `mockL64nc2048` | 64 | 2048 | 3 | 24 | 72 | 97.3 | 1.75 h | `Delta_N(lambda)` shape check |
| `mockNC128L32` | 32 | 128 | 13 | 48 | 624 | 4.1 | 0.07 h | matched low-`N_c` comparator |
| `mockNC128L48` | 48 | 128 | 13 | 48 | 624 | 19.9 | 0.36 h | matched low-`N_c` comparator |
| `mockNC128L64` | 64 | 128 | 13 | 48 | 624 | 57.3 | 1.03 h | matched low-`N_c` comparator |
| | | | | | **2808** | **378.8** | **2.32 h** | |

Plus **288 reused populations** at `L = 64, N_c = 1024`, λ ∈ {0.2932, 0.3032,
0.3132}, `R = 96` each, frozen from ARM B and not recomputed.

The three `mockNC128*` arms are **an addition to the brief's arm list**, flagged
as such, and separable. `NC128_COMPANION_RATIONALE.md` gives the reason: there
are **zero exactly-compatible cells** between this campaign and the historical
`N_c = 128` corpus, so without them brief §9C and the quantitative half of §12
have nothing to measure, and M3 has no comparator.

`L = 80` is **rejected**: 7.5–8.4 h elapsed even on the optimistic rate, against
a 3 h budget. `L80_RUNTIME_GATE.md`.

## 5. Why R = 24 and not something else

**[E]** From the ARM-B measurement, `R = 24` resolves neighbouring grid points
at 4.7–5.3 sigma where it can be checked directly, and 4.1–6.5 sigma projected
across the whole grid — with a realistic floor of ~3.4 after deflating the
projection by the 17 % it is optimistic by. That is a production-quality
configuration, not cosmetic error bars, and **no increase is needed anywhere in
the proposed range**. `POWER_AND_R_DECISION.md` has the arithmetic.

`R = 24` on the `N_c = 2048` arm rather than 16 because that arm is **not on the
critical path at any R in the brief's 16–24 range** — it finishes 34 minutes
before `mockL64` even at 24 — so the 17 % tighter bound on the
lambda-dependence of `Delta_N` costs 32 core-hours and no wall-clock.

`R = 48` on the companion arms so their SEM is comparable to the main arms';
because it splits into two disjoint `R = 24` blocks, giving the matched-R primary
comparison a full block A and an independent replication in block B; and because
it contains four disjoint `R = 12` subsets — the historical corpus's own
precision — which is what makes brief §12's question 4 answerable.

## 6. What parallelism costs, and what it buys

Every arm is an independent Slurm array with **no scientific dependency on any
other**. That is a deliberate trade and it is worth naming:

- **It buys wall-clock.** Submitted together, the campaign finishes when
  `mockL64` does: **2.32 h predicted, 3.25 h pessimistic**.
- **It costs the option of gating.** A sequential design would run `mockL64`
  first and cancel the rest if its curve failed M1. That would have saved up to
  250 core-hours in the bad case. The brief asked for wall-clock, so the arms
  run in parallel and that saving is forgone.

If the allocation turns out to grant 64 slots *in total* rather than per array,
elapsed becomes 6.8 h and the runbook's fallback is to submit the four main arms
first. `COST_MODEL.md` §5 and §7; `RUCHE_RUNBOOK.md` §2.

## 7. The matched-R rule, and the asymmetry it removes

Ten of the thirteen `L = 64` points carry `R = 24`; three carry `R = 96`; the
`N_c = 128` comparator carries `R = 48`. Left alone, that asymmetry would leak
into every cleanliness statistic, because roughness, increment significance and
crossing counts all measure scatter against error bars that scale as
`1/sqrt(R)`.

**The primary analysis is therefore matched at `R = 24` everywhere**: cells with
more populations are cut into disjoint blocks of 24 in **seed order**, and
block A is primary. The rule is observable-blind and asserted by
`tools/test_matched_r.py`. Full-`R` means, replicate blocks B/C/D and `R = 12`
subsets are secondary. `MATCHED_R_AMENDMENT.md` states it and shows that it
changes M3's verdict on synthetic data — the confound was real, not theoretical.

The reused points are interior to the grid rather than at an edge, so if the
reuse were somehow invalid it would show as a visible discontinuity in `q_i` at
indices 5–9 — which is `FALSIFICATION_PLAN.md` X1.

## 8. What the returned data will produce

Four figures (A: the curves; B: matched and historical low-`N_c`; C: the `N_c`
difference on exact common cells; D: the `N_c = 2048` shape check), the §10
curve-quality battery, the §11 crossing protocol per `L`-pair and per `N_c`
class, the §12 displacement decomposition, and the seven pre-registered M1–M7
verdicts. All of it from one script,
`analysis/mock_production_analysis.py`, which was exercised end to end on
fabricated data before submission and which found two real defects in the
process (`VALIDATION.md` §7).

## 9. What this task is not allowed to conclude

No `lambda_c(zeta)`. No finite-zeta exponent. No global phase-boundary law. No
`1/N_c` bias law. No `N_c(L, zeta, lambda)` law. No conversion of VIF into a
bias rule. No imposed monotonicity. The historical `dtau_mult = 12` corpus is
never pooled, averaged or interpolated into any quantitative statement.

The full list is `prohibited_conclusions` in `analysis_spec.yaml`, and the
analysis prints it at the end of every run — because prose guards are the ones
that get skipped.
