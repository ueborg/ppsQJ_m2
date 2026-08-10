# TASK-2026-08-10-AMP096 — task charter

**This is an infrastructure validation run, not a research result.** It was
opened to exercise `/research` v1 against a *historical, already-settled* case.
Nothing it produces may be merged into `research/state/**`, and its findings are
provenance for the workflow's behaviour, not support for any physics claim.

Written by the lead 2026-08-10, before any worker was dispatched.

## Research question

What happened to the historical claim that the Cut B phase-boundary amplitude
was A ≈ 0.96 — why did it appear, how do the parameterization (`lambda_c` vs
`r_c`) and observable (`OBS-BLPROD-001` vs `OBS-BLKMR-001`) conventions bear on
it, what evidence superseded or weakened it, and what does canonical state
currently permit us to conclude?

## Why this question, now

It decides nothing about the physics — the correction is already recorded. What
it decides is whether the research engine can reconstruct a superseded claim
without re-asserting it, and whether it distinguishes a convention mismatch from
a physical disagreement. Project memory re-asserted A ≈ 0.96 in August 2026
after the June correction, and the number is in the submitted M1 report, so the
failure mode is live.

## Canonical state in scope

| ID | kind | why it bears on the question |
|---|---|---|
| `CB-AMP-096-001` | claim, `withdrawn` | the historical claim itself |
| `CB-AMP-001` | claim, `provisional` | the superseding claim |
| `CB-PARAM-001` | claim, `supported` | the λ_c vs r_c Jacobian identity |
| `CB-WINDOW-001` | claim | window drift of fitted exponents |
| `OBS-BL-001` | observable, `retired` | the label both claims were written against |
| `OBS-BLPROD-001` | observable, `active` | our average-of-products locator |
| `OBS-BLKMR-001` | observable, `active_not_yet_computed` | KMR's product-of-averages |
| `DEC-OBS-001` | decision | B_L biases exponents low; locator vs exponent use |
| `EV-EXEC-AUDITREPRO-001` | evidence, `fully_reproducible` | the audit reproduction |
| `EV-DATA-AGGCASEB-001` | evidence | the Cut B aggregate the reproduction ran on |
| `DISP-PHI-001`, `DISP-WINDOW-001` | disputes | live, and must stay live |

## Competing hypotheses

1. **H1 — convention artifact.** A ≈ 0.96 and A ≈ 0.49 are the *same*
   measurement expressed under different conventions (`r_c` vs `lambda_c`,
   and/or a different B_L definition). Nothing physical separates them.
2. **H2 — measurement supersession.** A ≈ 0.96 was a genuinely different and
   worse measurement — different data, method, or extrapolation — that later
   analysis corrected on its own terms.
3. **H3 — both, partially.** A convention mismatch generated the discrepancy and
   an independent methodological weakness (no L-extrapolation, window drift)
   makes *neither* number a defensible asymptotic amplitude.

## Kill criterion

Stated before any evidence was seen: if canonical state does not record a
traceable reason for the withdrawal — i.e. if `CB-AMP-096-001` were withdrawn
without a documented basis — the task returns `Infrastructure first` and
proposes nothing.

## Compute tier

T0, read-only. No simulation, no HPC. The run stops at Human Gate A.

## What workers were NOT told

- Which hypothesis the lead favours, or that H1 is the recorded resolution.
- The values 0.96, 0.49, or the existence of a superseding claim.
- The contents of `CB-AMP-096-001.provenance_note`, which names the correction.

Workers received only the question, the IDs in scope, and the charter rules.
