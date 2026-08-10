# ASSESSMENT A–H — TASK-2026-08-10-AMP096          (Charter §5)

Each dimension assessed separately. **No aggregate score is produced**, and none
may be inferred by combining these verdicts.

Assessed against what the task produced **after** Stage 8, not before.

## A. Consequential bottleneck
**Verdict:** WEAK, and weaker than the lead judged before the red team.
**Reasoning:** The limitation addressed was "the reason recorded for a
withdrawal may be wrong". Real, but the withdrawal itself was never in doubt and
no downstream claim depends on `CB-AMP-096-001`. What remains is two false
metadata fields. The genuine bottleneck — that no Cut B amplitude is
estimator-stable — was already recorded in `CB-WINDOW-001`, `DEC-OBS-001` and
`METH-EXTRAP-001` before this task opened.

## B. Mechanistic contribution
**Verdict:** NONE.
**Reasoning:** No causal, operational or mathematical explanation changed. The
one candidate that looked mechanistic (C2: the p = 1/2 choice co-generating both
numbers) was already stated in `METH-EXTRAP-001`. Recovering an existing
mechanism is not contributing one.

## C. Discriminability
**Verdict:** ADEQUATE for the provenance question, ABSENT for the physics.
**Reasoning:** The provenance question had a decisive test and it was run: does
the May document form r_c? It does not. The physics question has no
discriminating test available on current data — the smallest decisive test
(`FALSIFICATION_PLAN.md`) needs wider L than either dataset provides.

## D. Dependency significance
**Verdict:** LOW as physics, MODERATE as bookkeeping.
**Reasoning:** Nothing depends on the withdrawn claim. But `SRC-KMR-2023`'s
`invoked_for` field carries a statement that is not in KMR, and that field is
citable by future work, so leaving it uncorrected propagates.

## E. Cross-silo value
**Verdict:** NOT APPLICABLE, and deliberately so.
**Reasoning:** No cross-field claim was made, so no `BRIDGE_AUDIT.md` was
required. The one external-comparison statement in scope ("twice the diffusive
prefactor") was found to have **no referent in the cited source**, which is the
opposite of cross-silo value: it is a citation that must be withdrawn.

## F. Robustness
**Verdict:** FAILED for four of five candidates.
**Reasoning:** C5's central number did not survive a window-edge check the lead
did not think to run. C3 depends on an unverified convention. C1's historical
clause is false. Only C4 survived, and only on one dataset.

## G. Informative failure
**Verdict:** STRONG. This is the dimension the task actually scores on.
**Reasoning:** The failure is informative twice over. Scientifically: the
apparent tension with `CB-AMP-001` was an artifact, so the live claim is
*confirmed* rather than weakened. Infrastructurally: it demonstrated that a lead
synthesis can promote a rediscovery to "the finding of the task", and that an
independent Stage 8 catches it.

## H. Infrastructure value
**Verdict:** STRONG, but this is a property of the run, not of the physics.
**Reasoning:** Three canonical fields were shown to be false, two sources were
promoted to inspected, and `DEC-CITATION-001` item 3 was verified from both
sides. The `/research` machinery itself was exercised end to end.
