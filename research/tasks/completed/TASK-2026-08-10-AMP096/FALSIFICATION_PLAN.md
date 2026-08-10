# FALSIFICATION_PLAN — TASK-2026-08-10-AMP096      (Charter Stage 4)

No large computation was proposed or run. Tier T0 throughout.

| check | done | result |
|---|---|---|
| counterexample search | yes | The endpoint relation A = lambda_c(1) under a single-power ansatz excludes A = 0.96 by a factor of two, independently of phi |
| limiting and degenerate cases | yes | zeta → 1 endpoint measured directly from `EV-DATA-BOUNDARYCSV-001`: lambda_c(1) = 0.43–0.49 |
| smallest analytically transparent model | yes | the single-power ansatz itself; it is what makes the endpoint test exponent-free |
| could the effect arise trivially? | yes | **Yes, and it did.** A = 0.53, 0.66, 0.79, 0.96 for extrapolation exponent p = 2, 1, 0.7, 0.5 on the same data. The amplitude is a function of a fitting choice. |
| strongest plausible baseline implemented | yes | the non-extrapolated audit estimator, reproduced exactly, plus an independent July dataset |
| regimes where the method SHOULD fail | yes | L ∈ {64…128} is too narrow for a 1/sqrt(L) extrapolation; per-zeta intercepts are non-monotone and one is negative |
| artifact of a definition, normalization, or the simulator? | yes | **Not the locator**: `OBS-BLPROD-001` vs `OBS-BLKMR-001` agree to 0.3–1%. It is the *estimator*, not the observable. |
| mechanism validation separated from performance | n/a | no method is being promoted |

## Smallest decisive test

If a bootstrapped, corrections-to-scaling L-extrapolation over L ≥ 5 sizes with
per-crossing error bars returns A outside 0.45–0.53, `CB-AMP-001`'s own
falsifier fires and the *superseding* claim is in trouble too. If it returns A
inside that band with a stable phi, `DISP-PHI-001` gains its first
estimator-stable input.

## What would make this test uninformative

Current L ranges. Both datasets are too narrow (`agg_caseB_combined` has 6 pairs
at large zeta; `boundary_aggregate.csv` spans only L = 64…128), and T/L is
uncontrolled on the July campaign. **Running the extrapolation on existing data
would produce a number with no discriminating power** — which is precisely the
outcome charter Stage 4 exists to predict before the compute is spent.
