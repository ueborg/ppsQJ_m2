# PROBLEM_MEMO — TASK-2026-08-10-AMP096            (Charter Stage 1)

Statement classes: [E] evidence, [I] inference, [C] conjecture, [J] judgment.

## 1. The observed or formal problem

[E] Canonical state holds two Cut B boundary amplitudes for the same functional
form lambda_c = A*zeta^phi: `CB-AMP-096-001` (A = 0.96 +/- 0.05,
`epistemic_status: withdrawn`) and `CB-AMP-001` (A ~ 0.49, `provisional`), the
second recorded as superseding the first.

[E] `CB-AMP-096-001` records the reason for withdrawal as "was an r_c-type
prefactor mislabelled as lambda_c", and records `evidence: []` with
`evidence_note: "Original supporting analysis not located on disk."`

[E] The claim file also records that the number was retained because project
memory re-asserted it in August 2026 and because it appears in the submitted M1
report. So the failure mode is live, not historical.

## 2. The smallest precise research question

Why did A ≈ 0.96 appear; is the recorded reason for its withdrawal correct; and
what does canonical state permit us to conclude about the Cut B amplitude today?

## 3. Why current approaches do not resolve it

[E] The state file asserts a cause (parameterization mislabel) without citing
evidence for that cause: its `evidence` list is empty. [I] A cause recorded
without support is exactly the kind of claim the charter's traceability rule
exists to catch, and it had not been checked against the primary document.

## 4. The theoretical or operational decision affected

[J] Whether the project can trust the *reasons* recorded in its own knowledge
plane, not only the verdicts. A correct verdict resting on a wrong diagnosis
will reproduce the error the next time the same situation arises.

## 5. Relevant constraints and information structure

[E] Two parameterizations (lambda_c, r_c = lambda_c/(1-lambda_c)) and two
locators (`OBS-BLPROD-001` average-of-products, `OBS-BLKMR-001`
product-of-averages) are in play, plus an estimator axis (L-extrapolated versus
not) that canonical state does not record for this comparison.

## 6. Strongest case that the problem matters

[J] The project's headline result (`DISP-PHI-001`, phi = 1/2 versus 1) rests on
the same fitting machinery that produced 0.96. If the amplitude turns out to be
an artifact of an extrapolation choice, the exponent inherits the doubt.

## 7. Strongest argument that the problem is artificial, already solved, or unimportant

[J] The verdict is already correct: 0.96 is withdrawn, 0.49 supersedes it, and
no downstream claim depends on 0.96. Re-litigating a settled withdrawal to
correct a note in a `withdrawn` claim is bookkeeping, not physics, and consumes
attention that `DISP-PHI-001` has a better claim on.

## 8. What survives that criticism

[I] The criticism holds for the *verdict* and fails for the *diagnosis*. Three
independent investigators found the recorded reason contradicted by a primary
document that is tracked in git. [I] The correction is not cosmetic: the real
mechanism (an extrapolation-exponent choice that assumed the answer) is a
generative error that is still active in the exponent work, whereas the recorded
mechanism (a parameterization mislabel) is inert and already guarded by
`CB-PARAM-001`.
