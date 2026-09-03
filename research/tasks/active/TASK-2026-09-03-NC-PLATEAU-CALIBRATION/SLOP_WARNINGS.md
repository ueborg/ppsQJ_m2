# SLOP WARNINGS — TASK-2026-09-03-NC-PLATEAU-CALIBRATION   (Charter §6)

Explicit verdict on **all twelve**, for the surviving candidates as a group and
per-candidate where they differ. A flagged direction is **not** discarded
silently: the reasoning and any surviving reformulation are recorded.

`clear` · `flagged` · `fatal`

| # | warning | verdict | reasoning |
|---|---|---|---|
| 1 | established method on a routine new dataset / model / topology / application | **flagged** | `[J]` This is honestly what campaigns A, C and D are: an existing estimator, run at larger `N_c`, on the same cells. It is not a method contribution and the package never calls it one. `[E]` What keeps it from being routine is that the criteria (P1–P5, `tau_I` derived from `tau_lambda`) were frozen before the data and two of the six arms are declared unable to satisfy their own headline criterion. **Reformulation that survives:** the campaign's value is the *decision* — the smallest defensible `N_c` per `L` — not the measurement. |
| 2 | two known techniques combined with no nontrivial interaction | **clear** | `[E]` Nothing is combined. One sampler, one observable, one axis per campaign. |
| 3 | metric is a monotone transform, weighted sum, or rename | **clear** | `[E]` `Delta_N = I_{2N} - I_N` and `B_eff = -2N Delta_N` are not renames of the observable: `B_eff` is constant in `N` **iff** the pure `1/N` form holds, which is the property under test. `[E]` `tau_I` is explicitly *derived* from `tau_lambda` through a **measured** slope, and the derivation is written out rather than asserted. |
| 4 | another constraint on a familiar optimization, no conceptual change | **clear** | `[E]` No optimisation is performed anywhere in this task. |
| 5 | architecture swap for a small benchmark gain | **clear** | `[E]` The sampler is byte-identical (sha256 gated) and demonstrated bit-identical on re-execution. Nothing was swapped. |
| 6 | theorem whose assumptions largely encode the conclusion | **clear** | `[E]` No theorem is proved. `[J]` The nearest risk was importing the predecessor's suggestion that multiplicative correction survives where additive fails; `SUCCESS_CRITERIA.yaml` gives it explicitly zero prior weight and tests H1 and H2 on equal terms. |
| 7 | regime constructed because it makes the method outperform | **flagged, and the flag was acted on** | `[E]` Campaign B's `lambda` window comes from **observed** interior sign changes in already-measured curves, not from any law — not `sqrt(zeta)`, not `zeta^(1/3)`. `[E]` But a narrow window is exactly where a correction is most likely to look flat, which would flatter candidate C2. **Acted on:** the analysis reports increments and second differences, not only the fitted constant, and `CANDIDATES.md` C2/A9 records this as the simplest competing explanation of a flat result. |
| 8 | weak, obsolete, or informationally disadvantaged baseline | **clear** | `[E]` The comparison at every step is the *same* estimator at smaller `N_c`, from the byte-identical sampler with disjoint seeds. `[E]` Where existing rungs carry a larger `R` than new ones, matched-`R` blocks cut in seed order are the primary statistic and full-`R` views carry no verdict authority. |
| 9 | computational scale treated as scientific depth | **flagged — the sharpest one here** | `[J]` The campaign is 2 180 core-hours and its headline move is "run it at bigger `N_c`". That is scale. `[E]` What stops it being scale-as-depth: `N_c` and `R` are budgeted and reported **separately**, every verdict names which one binds, and `UNRESOLVED_R_LIMITED` exists precisely so that a bigger `N_c` cannot be mistaken for an answer to a question `R` controls. `[E]` The single most decision-relevant result derivable here — that absolute-level certification at `L = 128` is unreachable at any affordable `R` — was obtained from the **existing** variances with no new compute at all. **Reformulation:** the campaign's justification is the frozen tolerance and the power calculation, not the core-hour figure. |
| 10 | runnable code treated as evidence a problem exists | **clear** | `[E]` The problem is evidenced by measurements: a resolved `−0.060 ± 0.023` step at `L = 128`, a `1/N` rejection at `p = 0.0056`, and a structural obstruction derived in a predecessor task. `[E]` The code came after. |
| 11 | silo-breaking claimed from terminology alone | **clear, by refusal** | `[E]` No cross-field claim is made. `[J]` C3's structure (a common offset cancels in a difference) plausibly appears elsewhere, and **no external search was performed**, so `NOVELTY_MATRIX.md` and `ASSESSMENT_AH.md` §E both decline to claim cross-silo value and record that the search is owed. `BRIDGE_AUDIT.md` is absent because there is no bridge to audit. |
| 12 | paper drafted around an artifact before the claim exists | **clear** | `[E]` No manuscript was touched. `[E]` `FALSIFICATION_RESULTS.md` records Y1–Y8 as **not yet attempted** and refuses to write outcomes for data that do not exist. |

## Per-candidate divergences

`[E]` **C6 (campaign E) is `clear` on warning 1** where the others are flagged:
it is not "an established method on a new dataset" but a manipulation that leaves
the target measure exactly invariant while moving the algorithmic schedule, and
both of its outcomes kill a named mechanism.

`[E]` **C5 (campaign D) is the worst offender on warning 9**: 502 core-hours,
23 % of the campaign, for 16 tasks and one screening number that cannot certify
convergence. `[J]` It survives because both of its outcomes change what is
submitted next and because it is the cheapest way to learn whether the
absolute-level route at `L = 128` is dead. `[E]` `NOVELTY_GATE.md` records that a
predecessor costed this design at 901 core-hours and **declined** it, and that
the decline was correct on its own (mechanism-discrimination) terms.

`[E]` **C3 (locator convergence) is the one flagged on warning 11**, and the
flag is left standing rather than resolved.

## Overall

**flagged on 3 of 12 (1, 7, 9), fatal on none.**

`[J]` The three flags are the real ones for a calibration campaign and none is
argued away: warning 1 is accepted (this is not a method contribution), warning
7 was acted on in the analysis design, and warning 9 is answered by the fact
that the campaign's most decision-relevant result cost nothing to obtain.
