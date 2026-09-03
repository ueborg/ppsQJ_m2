# Design

`TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION`. A cheap extension of the completed
`TASK-2026-09-02-MOCK-PRODUCTION`.

---

## 1. What is actually being asked

The predecessor produced three clean 13-point `CMI(lambda)` curves and then
could not answer its own crossing question, because the crossing structure sat
on the lower edge of the scan:

```
                lambda:  0.2332 ..................................... 0.3532
  I48 - I32:            -0.0202  -0.0259 ...  -0.1086      no sign change
  I64 - I32:            -0.0051  -0.0514 ...  -0.1714      no sign change
  I64 - I48:            +0.0151  -0.0255 ...  -0.0628      ONE, at 0.23691
                        ^^^^^^^
                        the first scanned point
```

`I64 - I48` changes sign in the very first interval. `I48 - I32` and
`I64 - I32` never change sign on the grid at all, yet their bootstrap crossing
mass piles up at `0.2333–0.2378`, i.e. jammed against the boundary. Both
symptoms have the same cause: the interesting behaviour is at or below
`lambda = 0.2332`, and the scan starts there.

The fix is not statistical. It is four more lambda points.

---

## 2. What changes and what does not

**Changed: nothing except the lambda range.**

| | predecessor | here |
|---|---|---|
| zeta | 0.35 | 0.35 |
| `N_c` | 1024 | 1024 |
| `R` | 24 | 24 |
| `L` | 32, 48, 64 | 32, 48, 64 |
| `T` | `L` | `L` |
| `dtau_mult` | 6.0 (certified) | 6.0 (certified) |
| resampling | systematic | systematic |
| sampler | `instrumented.py` sha256 `0a33c403…` | **the same bytes** |
| `delta_lambda` | 0.010 | 0.010 |
| lambda range | `0.2332 … 0.3532` | `0.1932 … 0.3532` |
| grid points | 13 | 17 (4 new + 13 reused) |

Holding everything else fixed is the whole design. A change of `R`, of `N_c` or
of `dtau_mult` across the join would make the extended curve a comparison of
two measurements rather than one measurement over a longer interval, and every
statistic downstream — the increments, the second differences, the roughness,
the crossings — would then be confounded at exactly the point they are most
delicate.

---

## 3. The arms

Three arms, one per `L`, each computing only the four new lambdas:

```
lowlamL32   X32   L=32  T=32   4 lambdas x R=24 =  96 tasks
lowlamL48   X48   L=48  T=48   4 lambdas x R=24 =  96 tasks
lowlamL64   X64   L=64  T=64   4 lambdas x R=24 =  96 tasks
                                                  ----------
                                                  288 tasks
```

No arm recomputes any of the thirteen already-measured lambdas. The preflight's
`no predecessor duplication` check is a hard failure, not a note, and
`tools/dedup_scan.py` re-derives the same conclusion independently by scanning
every manifest under `research/tasks/active/`.

---

## 4. What is deliberately NOT here

- **No `N_c = 128` companion.** The predecessor prepared three such arms and
  they were **cancelled**; they returned zero results. This task neither
  revives them nor reads them. `tools/freeze_predecessor.py` asserts they are
  empty rather than assuming it.
- **No `N_c = 2048` arm.** The predecessor's `mockL64nc2048` returned 72
  results, and they are **deliberately not read**. A different population size
  has no place in a curve-shape and crossing extension at `N_c = 1024`; pooling
  it would be exactly the kind of silent contamination the matched-`R` rule
  exists to prevent. The freeze script prints that refusal explicitly.
- **No historical `dtau_mult = 12` corpus.** Not poolable, and this task makes
  no use of it at all — not even descriptively. The predecessor drew it in one
  figure panel marked `DESCRIPTIVE ONLY`; there is no counterpart here, because
  the question is about a join between two halves of one certified grid and the
  corpus is not on that grid.
- **No `L = 80`, `L = 96` or `L = 128`.** Out of scope. The predecessor's
  `L80_RUNTIME_GATE.md` and `L128_NC2048_HANDOFF.md` still stand unchanged.
- **No second extension.** Whatever the outcome, the grid is not extended again
  automatically. Pre-registered as `FALSIFICATION_PLAN.md` Y6.

---

## 5. The analysis, and the one thing it adds

`analysis/lowlambda_analysis.py` re-runs the predecessor's whole battery on the
17-point grid — per-cell means and across-population SEMs, VIF as a variance
diagnostic only, adjacent increments and their SEMs, second finite differences,
the roughness statistic with its bootstrap CI, split-half and leave-one-out
stability, the maximum standardized population outlier, and the weighted
quadratic yardstick — and then the full crossing protocol on all three pairs.

It adds exactly one new thing: **the join test**.

The join is where this design could fail invisibly. Four points computed on a
different day, on different nodes, could in principle sit slightly off the
curve they are being attached to, and the resulting kink would then be read as
structure. So `J1`, `J2` and `J3` (`SUCCESS_CRITERIA.md` §3) ask, three
different ways, whether the new points join continuously:

- **J1** compares the two second-difference triples that straddle the join
  against the bootstrap distribution of the curve's *own* worst roughness — so
  the join is judged against this curve's noise level, not against zero.
- **J2** fits a weighted quadratic to the **five lowest already-measured
  points only**, and asks it to predict the four new ones. The fit never sees a
  new point; this is a genuine out-of-sample check.
- **J3** asks whether the single increment straddling the join steps away from
  the local trend of the three increments on either side.

None of the three is allowed to remove a point. A `FAIL` licenses saying the
join is not smooth. It does not license making it smooth.

---

## 6. Interiority: the falsification target

The crossing protocol is the predecessor's, unchanged, plus a pre-registered
**interiority** classification frozen before any datum exists
(`analysis_spec.yaml` → `crossings.interiority`):

```
I1  the raw crossing is not in the first or the last interval
I2  it SURVIVES DELETING THE FIRST LAMBDA POINT (0.1932)
I3  the bootstrap 95 % interval is clear of both endpoints by >= delta_lambda/2
```

`I2` is the one the brief pre-registered in words — *"an interior crossing must
not depend on the first lambda point"* — and it is implemented literally: the
entire crossing machinery is re-run on the 16-point grid with `0.1932` removed,
and the crossing must survive at a location inside the full grid's bootstrap
interval.

Four outcome classes, and only four:

| class | meaning |
|---|---|
| `INTERIOR` | `I1` and `I2` and `I3` — the extension worked |
| `STILL_BOUNDARY` | a crossing exists but fails one of them: the locator moved *with* the boundary |
| `BELOW_GRID` | no sign change on 17 points and the bootstrap mass piles at the lower end |
| `NONE` | no crossing, no lower-end accumulation |

`STILL_BOUNDARY` is reachable **only** when a raw crossing exists. Routing "no
crossing at all" into it would report a boundary artefact where there is no
locator to be an artefact of — a defect the smoke test caught and
`VALIDATION.md` §6 records.

`tools/smoke_test.py` proves on synthetic data that this classification really
does distinguish the three cases, including the `edge` case where the sign
change is forced into the first interval. An interiority test that could not
fail would be worthless, so it is shown failing.

---

## 7. Cost

`61.3` core-hours predicted, `85.8` pessimistic; `56.6` minutes elapsed for the
long pole at `%64`, excluding queue wait. This is roughly **one sixth** of the
predecessor's main-arm cost, for four points that answer its open question.

Every figure is fitted to `wall_s` recorded by that campaign's own completed
Ruche jobs, and the preflight refits it from the frozen data and fails if the
literals have drifted. Full arithmetic: `COST_MODEL.md`.

---

## 8. Terminal state

`READY_FOR_HUMAN_SUBMISSION`. No agent submits. The researcher types the
commands in `RUCHE_RUNBOOK.md`.
