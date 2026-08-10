# RESEARCH_MEMO — TASK-2026-08-10-AMP096           (Charter Stage 9)

Not a manuscript (§4.3). Statement classes: [E] evidence, [I] inference,
[C] conjecture, [J] judgment.

**This task was run to validate `/research` v1 on a historical case. Its
scientific output is deliberately small and its findings are not merged.**

## 1. The question investigated

Why did the Cut B amplitude A ≈ 0.96 appear, is the recorded reason for its
withdrawal correct, and what does canonical state permit us to conclude now?

## 2. Why it matters

[J] Not for the physics — `CB-AMP-096-001` is `withdrawn` and nothing depends on
it. It matters because a *correct verdict resting on a wrong diagnosis* will
reproduce the error, and because project memory re-asserted the number in August
2026 after the June correction.

## 3. What was previously known

[E] `CB-AMP-096-001` is `withdrawn`, superseded by `CB-AMP-001` (A ≈ 0.49,
`provisional`). [E] The withdrawal reason on file: "was an r_c-type prefactor
mislabelled as lambda_c". [E] `CB-PARAM-001` (`supported`) establishes that
lambda_c/r_c exponent gaps are pure Jacobian. [E] `METH-EXTRAP-001`
(`withdrawn`, `review_status: stale`) already recorded that the 1/sqrt(L)
extrapolation choice was "circular on an unmeasured nu", with the four-form
chi²/dof table.

## 4. Which candidates were eliminated, and why

[E] Four of five, all by Stage 8. C1 (the "never was an r_c prefactor" clause)
is **false as history**. C2 is a **rediscovery of `METH-EXTRAP-001`** — and the
lead had labelled it "the finding of the task". C3 is a tautology on an
unverified convention. C5's headline number is a **window-edge artifact**: the
crossings sat at the last sampled lambda with B_L collapsed to numerical zero,
and restricting to the clean interior reproduces `CB-AMP-001` exactly. Full kill
record in `CANDIDATES.md`.

## 5. What survived

[E] One scoped result (C4): `OBS-BLPROD-001` and `OBS-BLKMR-001` agree to a
median 0.31% on crossing position and ~1% on amplitude, on
`agg_pps_clone_guided_prod.pkl` **only**. This partially discharges
`OBS-BLKMR-001`'s `open_task` on one dataset.

[E] Three canonical fields are false and can be corrected without any scientific
status change:
1. `CB-AMP-096-001.evidence_note` says the original analysis was "not located on
   disk". It is on disk and tracked at `6c9c843`:
   `analysis/lambda_c_phi_analysis.md`.
2. `OBS-BLKMR-001.recomputable_from` asserts `S_AB_mean` is in every guided
   aggregate. It is absent from `agg_caseB_combined.pkl` — the dataset
   `CB-AMP-001` rests on.
3. `SRC-KMR-2023.invoked_for` attributes to KMR a statement of the form
   "lambda_c ~ A·r_c^{1/2} with A of order one half". **Full-text inspection
   found no such statement**; the word "diffusiv" does not occur in the paper,
   and KMR's `r_c` is a detector readout threshold (Eq. 20, values −2 to −0.5),
   not lambda_c/(1−lambda_c).

[E] Two sources were promoted: `SRC-KMR-2023` from title-only on the B_L
definition to body-verified, and `SRC-FULGA-2012` from `not_inspected` with an
unknown arXiv id to identified and read (arXiv:1205.1441). `DEC-CITATION-001`
item 3 is now verified from both sides.

## 6. The evidence

Three independent investigator reports (`agent_reports/`), a Stage 8 report that
validates clean (`REDTEAM.yaml`, 0 errors, `lead_summary_seen: false`), and the
reviewer's own read-only recomputation on `results/boundary_aggregate.csv`.
[I] The strongest single piece is the reviewer's, because it was produced while
trying to destroy the candidates rather than support them.

## 7. The remaining uncertainty

[E] The zeta convention is unverified against `SRC-LMR-2025`. [E] The source of
lambda_c(1) = 1/2 is still unfound after a 182-PDF library search and four web
queries. [E] No defensible L-extrapolation exists on either dataset: L ranges are
too narrow and per-zeta intercepts are non-monotone. [I] Therefore
`CB-AMP-001`'s own falsifier **cannot currently be evaluated** — it did not
fire, and it also cannot be made to fire or not fire on data we hold.

## 8. The actual contribution

[J] Stated without inflation: **a bookkeeping correction and a withdrawn
citation.** No amplitude, no exponent and no dispute position moved. The task's
real product is a demonstration that the engine's Stage 8 catches the lead
over-claiming, which is infrastructure, not physics.

## 9. Reusable artifacts produced

The task directory; three raw investigator reports; a validated Stage 8 report;
and `/research` v1 itself, exercised end to end.

## 10. The next human decision

Whether to accept the three field corrections in `proposed/` at the merge gate,
and whether the p = 1/2 extrapolation question — **not investigated here** —
should be opened as its own task bearing on `DISP-PHI-001`.

---

## Status report (§12)

1. **What was established.** Three canonical metadata fields are false. Two
   sources promoted to inspected. `OBS-BLKMR-001`'s open task partially
   discharged on one dataset. KMR contains no "diffusive prefactor" statement.
2. **What was refuted.** Four of the lead's five candidates, including the one
   the lead called the task's finding. The apparent tension with `CB-AMP-001`
   was a window-edge artifact; **`CB-AMP-001` came out of this stronger, not
   weaker.**
3. **Assumptions introduced or removed.** None introduced. The single-power
   premise underlying C3 was *identified* as an assumption the record already
   denies.
4. **Uncertainties remaining.** The zeta convention; the source of
   lambda_c(1) = 1/2; whether p = 1/2 is still load-bearing for
   `CB-PHI-HALF-001`; whether any L-extrapolation is defensible on current L
   ranges.
5. **Files and artifacts changed.** Only `research/tasks/active/TASK-2026-08-10-AMP096/`.
   **`research/state/**` is byte-identical** — SHA-256 of the sorted file
   digests is `76da2c75…2ecfd` before and after.
6. **Decision requiring human judgment.** The three field corrections, and
   whether to open the p = 1/2 task.

**Activity is not progress.** 450k subagent tokens produced one scoped result
and three metadata fixes. That is the honest accounting.
