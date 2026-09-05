# FALSIFICATION_PLAN — TASK-2026-09-06-NC-ZETA-MOCKPROD

Charter Stage 4. **PRE-SPECIFIED and frozen at `stage_3_candidates`.** This file
states what will be attempted and what would kill each target. **It contains no
results.** Outcomes go to `FALSIFICATION_RESULTS.md`, written afterwards and
never merged back. Labels `[E]` `[I]` `[C]` `[J]`.

Two classes:
- **T0** — read-only analysis, executable now, before any submission.
- **C** — needs the campaign's returned data, therefore **not attempted in this
  task**.

---

## T0 targets — attempted now

**F1 (T0). The `N_c^0.1871` rate law is correct.**
*Kill if* the rungs completed since it was fitted disagree with it by more than
25 % at any `L` this task uses. *Method:* refit `rate35` from every stored
`zeta = 0.35`, `dtau_mult = 6` population and compare against the law's
prediction. *Consequence if killed:* the cost model must not use it, and every
`--time` in the package changes.

**F2 (T0). The proposed `lambda` grids bracket both open positions of
`DISP-PHI-001`.**
*Kill if* any prediction is outside a grid, has fewer than two grid points beyond
it on either side, or sits within `4*tau_lambda` of a grid **end**. *Method:*
preflight `P9`, per arm, plus independent re-derivation of the `STAGE1` bracket
rule at all three `zeta`. *Consequence if killed:* the grid is widened, never
narrowed, and the campaign does not ship until it is.

**F3 (T0). Some exact-compatible data already exists at these `zeta`.**
*Kill the "nothing to reuse" claim if* any stored population matches a cell in
the design on all seven of `(zeta, L, T, N_c, lambda, dtau_mult,
resample_scheme)`. *Method:* whole-repository scan of stored results, plus
preflight `P14` per arm.

**F4 (T0). The preflight checks are decorative.**
*Kill the package if* any preflight check cannot be made to fail. *Method:* 13
injected-fault negative controls, each requiring rejection with a named code.
*Consequence if killed:* the failing check is rewritten or deleted; a check that
cannot fail is worse than no check, because it is read as assurance.

**F5 (T0). The `zeta` timing ratios are an artifact of one `L`.**
*Kill the adopted `rho` if* the per-`L` ratios disagree by more than a factor of
1.5 at any `zeta`. *Method:* re-derive `rho` at all four `L` in the ALGRD set
independently of the predecessor's table.

**F6 (T0). The frozen predecessor was disturbed.**
*Kill the run if* any file under `TASK-2026-09-06-NC-ZETA-STAGE1` differs from
the baseline recorded when this task opened, or if any file was added to it.
*Method:* `tools/check_predecessor.py`, also run as preflight `P15`.

**F7 (T0). Packing changes the computation.**
*Kill packing if* a packed row is not exact-compatible with an unpacked one.
*Method:* inspect `run_pack.py` for any argument, environment or ordering
difference; verify by sha256 that `run_cell.py` is byte-identical to the
predecessors' certified executor and is invoked once per row in a fresh process.

---

## Class-C targets — NOT attempted in this task

`[E]` These need the returned data. They are pre-registered now so that they
cannot be chosen after seeing it. **Their status in
`FALSIFICATION_RESULTS.md` will be `not attempted`.**

**F8 (C). `R = 16` can separate adjacent-rung movement from noise.**
*Kill the survey's product if* every rung at every `zeta` classifies
`INCONCLUSIVE`. *Consequence:* report a negative result and the `R` required;
do **not** rerun at larger `R` inside this task.

**F9 (C). The `L48-L64` crossing is interior at the new `zeta`.**
*Kill the crossing column for a `(zeta, N_c)` cell if* the sign change falls in
an end interval, or the bootstrap interval touches an end. *Consequence:*
`ENDPOINT_INDUCED`, `ABOVE_GRID` or `BELOW_GRID` is reported for that cell and
no interpolated location is quoted. The curves are still reported.

**F10 (C). The three `L` agree about the crossing.**
*Kill the "one locator" reading if* `L32-L48`, `L32-L64` and `L48-L64` give
crossings that are mutually inconsistent at a given `(zeta, N_c)`.
*Consequence:* report the disagreement; it is the diagnostic `L = 32` exists for.

**F11 (C). `dtau_mult = 6` is adequate at low `zeta`.**
*Kill it if* the `dtau_mult in {3, 6, 12}` legs at `zeta = 0.10`, `L = 64`,
`N_c = 512`, `lambda = 0.100` differ by more than 3 combined SEM.
*Consequence:* the low-`zeta` curves are flagged as discretisation-limited.
`[E]` A null here licenses **nothing global**: it is one point, one `L`, one
`N_c`, one `lambda`, and no discretisation theorem may be inferred from it.

**F12 (C). Finite-`N_c` movement is monotone in `N_c`.**
*Kill if* any `zeta` shows a rung sequence that is non-monotone beyond its SEM.
*Consequence:* the "smallest adequate `N_c`" recommendation for that `zeta`
becomes `INCONCLUSIVE`; a non-monotone ladder cannot support a threshold.
`[C]` This is expected to be a real risk: the measured `rate` ladder is itself
non-monotone in `N_c`, so there is no prior reason the drift must be monotone.

**F13 (C). The cost model predicted the campaign.**
*Kill the model if* measured core-hours exceed the pessimistic figure, or if any
task exceeds its `--time`. *Consequence:* the model is corrected before it is
used to size anything else, and the correction is recorded whichever way it errs.

---

## What no outcome of this task may license

`[E]` No `lambda_c(zeta)`. No boundary exponent. No `N_c^req(zeta)` and no fit of
one. No transfer of a `zeta = 0.35` coefficient to another `zeta`. No claim about
`L > 64` or `N_c > 2048`. No use of the word `certified`. `DISP-PHI-001` and
`DISP-WINDOW-001` stay open.
