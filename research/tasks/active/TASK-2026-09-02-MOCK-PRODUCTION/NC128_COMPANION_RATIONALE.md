# The matched N_c = 128 companion arm — an addition to the brief, and why

TASK-2026-09-02-MOCK-PRODUCTION.

**This is the one place the package departs from the arm list in brief §16.**
The brief asks for `MOCK-L32`, `MOCK-L48`, `MOCK-L64`, `MOCK-L64-NC2048` and an
optional `MOCK-L80`. This package adds three more:
`mockNC128L32`, `mockNC128L48`, `mockNC128L64`.

They are flagged rather than folded in, and they are separable: dropping them
costs the campaign 81.3 core-hours, no wall-clock at all, and exactly two things
— brief §9's Figure C and the quantitative half of brief §12.

## The finding that forced the decision

**[E]** Brief §9 Figure C asks for the high-`N_c` minus low-`N_c` CMI
difference "wherever exact compatible cells exist", and brief §12 — marked
*crucial* — asks six quantitative questions about the move from `N_c = 128` to
`N_c = 1024`, with the instruction to "use only exact common cells for
quantitative differences".

Inspecting the actual archive rather than the summaries
(`REUSE_AND_DEDUP_AUDIT.md`, reproducible with `tools/dedup_scan.py`):

> **There are zero exactly-compatible cells between this campaign and the
> historical `N_c = 128` corpus. At any L, any lambda, any N_c.**

Four independent reasons, each sufficient:

1. every corpus row is `dtau_mult = 12.0`; this campaign is the certified 6.0;
2. exactly one corpus lambda (0.3032) is a point of this campaign's grid;
3. the corpus has no `L = 32` and no `L = 48` at all — it starts at `L = 64`;
4. the corpus has no `seed` column, so independence cannot be established.

Reasons 1 and 4 hold even at the single cell where `L`, `lambda` and `N_c` all
coincide (`L = 64, lambda = 0.3032, N_c = 128`, 12 rows).

**[I]** So brief §9C and the quantitative part of §12 are, against the archive
as it stands, **unanswerable as specified**. The brief anticipated this — it
says "wherever exact compatible cells exist" and permits an explicitly-labelled
descriptive comparison — but the consequence is that:

- Figure C would be emitted empty;
- §12's questions 1, 2, 5 and 6 would have no matched measurement behind them;
- **M3 would have no comparator at all.** M3 asks whether cross-`L` locator
  structure is "materially cleaner than in the old `N_c = 128` scan"; the old
  scan has no `L = 32`, no `L = 48`, a different lambda grid and a different
  discretisation, so there is no cross-`L` crossing analysis on it to compare
  against.

## What the companion arm buys

Three arms at `N_c = 128`, `R = 48`, on the **identical** 13-point grid, at
`L = 32, 48, 64`, with `dtau_mult = 6.0`, systematic resampling, `T = L`, and
the same bundled sampler. **The only variable that differs from the main arms is
`N_c`.**

That yields **39 exact common cells** (13 lambdas × 3 L), and with them:

- **Figure C** as specified, on measured cells rather than on interpolation.
- **§12 questions 1, 2, 5, 6** answered by a controlled comparison. The
  displacement decomposition in `analysis_spec.yaml` — constant vs linear vs
  quadratic vs irregular, per `L`, with chi-square — is only meaningful on
  matched cells.
- **§12 question 4** ("does the independent-population uncertainty explain the
  apparent old jaggedness?") answered directly, and additionally by subsampling
  `R = 48` down to four disjoint `R = 12` subsets, which is exactly the
  historical corpus's own precision. That question cannot be posed at `R = 12`.
- **M3** given a real, matched comparator, so that "materially cleaner" is a
  measurement rather than a comparison across two different discretisations.
- **A byproduct at `L = 64`**: the corpus also has `L = 64, N_c = 128`, so
  running the same cell at `dtau_mult = 6` isolates the discretisation
  systematic from the population-size effect. Nothing in the programme has done
  that yet.

## What it costs

| | |
|---|---|
| core-hours | 81.3 predicted (113.8 pessimistic), **21 % of the campaign** |
| array tasks | 1,872 in three arrays of 624 |
| elapsed at %64 | 0.07 h, 0.36 h, 1.03 h — all **below** the 2.32 h long pole |
| effect on campaign wall-clock | **none**; the critical path is `mockL64` |
| slowest single task | 0.11 h against a 1 h wall limit |

**[J]** 21 % of the core-hours, no wall-clock, and it converts two sections of
the brief and one success criterion from "cannot be evaluated" to "measured". I
judge that a good trade and recommend running it.

## What it is not

- It is **not** a physics arm. `N_c = 128` is known to be far from converged —
  the ARM2 ladder at `L = 128` shows the mean still moving by 0.10–0.12 per
  doubling below `N_c = 256`. These curves are a **methodological control**,
  not a measurement of CMI.
- It does **not** replace the historical corpus in the record. Figure B keeps
  the corpus panel, dashed and labelled `DESCRIPTIVE ONLY`, because the brief
  asked for the historical comparison and because the corpus is the thing the
  programme's intuitions were actually formed on.
- It does **not** license a `1/N_c` law. Two populations sizes at three `L` and
  one `zeta` support no such thing, and `analysis_spec.yaml`
  `prohibited_conclusions` forbids it explicitly.

## If the human declines it

Drop the three `mockNC128*` arms from the submission. Everything else is
unaffected: no other arm depends on them, `tools/build_arms.py` regenerates
without them by deleting their three `ARMS` entries, and the analysis degrades
correctly and by design — Figure C is emitted **empty with the reason printed on
it** rather than filled by interpolating the corpus, and **M3 returns
INCONCLUSIVE with the explicit reason that it has no matched comparator and that
comparing against `dtau_mult = 12` data instead is refused.**

That refusal is deliberate and is coded, not merely written down.
