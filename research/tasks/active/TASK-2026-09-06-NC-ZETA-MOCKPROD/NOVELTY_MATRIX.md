# NOVELTY_MATRIX — TASK-2026-09-06-NC-ZETA-MOCKPROD

Charter Stage 2. One row per comparator, eight columns. Labels `[E]` `[I]` `[C]`
`[J]`.

`[E]` **Nothing in this task is claimed novel.** The matrix records what this
design does and does not change relative to its three closest predecessors, so
that the difference is auditable rather than asserted.

---

| dimension | `NC-ZETA-STAGE1` (not run) | `NC-ZETA-CALIBRATION` (Gate A: Reformulate) | `NC-PLATEAU-CALIBRATION` (ran) | **this task** |
|---|---|---|---|---|
| **problem definition** | certified `N_c^req(zeta)`: smallest `N_c` whose next-rung crossing shift is inside `tau_lambda = 0.004` | same, at eight `zeta` | `N_c` ladder to 8192 at `zeta = 0.35` only | **how much do the curves and the rough crossing still move per rung**, qualitatively, at three `zeta` |
| **information assumptions** | law-agnostic brackets; `DISP-PHI-001` open | brackets left-censored by the `0.51*sqrt(zeta)` historical grid | single `zeta`, so no boundary-law assumption needed | law-agnostic brackets, **span** tested against both positions, `DISP-PHI-001` open and untouched |
| **mathematical mechanism** | TOST equivalence on a crossing shift, with an interval | TOST, applied as a point test — the defect it found in itself | crossing shift vs rung, point estimate | **no equivalence test at all.** Difference curves, RMS, SEM-normalised change, rough crossing shift, four qualitative classes |
| **guarantee** | a certification, if `R` suffices | same | none claimed | **none.** `ROUGHLY STABLE` is explicitly resolution-limited and may not be read as convergence |
| **empirical evidence** | none; not run | none; not run | 3 280 tasks planned, ladder returned to `N_c = 8192` | 6 080 committed populations at three new `zeta`; **zero exact-compatible data existed before** |
| **operational constraints** | `R_req` 51–65 at the anchor alone | 615 core-h for brackets that would miss at 6 of 8 `zeta` | 2 180 core-h at one `zeta` | **1 160 core-h committed**, `R = 16`, one 862 core-h rung held conditional |
| **computational cost** | not costed to completion | costed with the `N_c^0.1871` law | adopted the `N_c^0.1871` law | **refutes that law** from that campaign's own returned rungs; flat `rate35(L)` per `L` |
| **reusable output** | the bracket rule | `run_pack.py`, the packing discipline, `rho(zeta)` | `rate35`, the measured-memory finding | the corrected rate model, a `P9` that tests grid **span** not centre, 13 negative controls, and the first production data off `zeta = 0.35` |

## The three things this task changes

`[E]` **1. The `N_c` runtime law.** `N_c^0.1871` is refuted by measurement, not
argued down: `L = 64, N_c = 4096/8192` measured 5.05/5.11 ms against 6.57/7.48
predicted, and `L = 128, N_c = 2048` measured 22.45 against 31.7. The refit runs
on every preflight.

`[E]` **2. The grid-coverage check.** `NC-ZETA-CALIBRATION`'s `P9` tested a
stencil **centre** against a law and gave false assurance. This `P9` tests the
**span** against both open positions, requires two grid points beyond each and a
4-`tau` margin to the nearer grid end, and negative control `N8` shows it
rejecting a narrowed grid.

`[E]` **3. Production data at a `zeta` other than 0.35.** There is none in the
repository today.

## What this task does NOT change

`[E]` `DISP-PHI-001` stays open. No `lambda_c(zeta)`. No exponent. No
`N_c^req(zeta)`. No claim about `L > 64`, `N_c > 2048`, or any `zeta` outside
the three measured. `[E]` The `zeta = 0.35` anchor is not recomputed and its
`R`-limited status is not improved by anything here.
