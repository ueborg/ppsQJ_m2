# PARKING_LOT — TASK-2026-09-06-NC-ZETA-MOCKPROD

Things noticed and **deliberately not pursued**, with what would open each.
Nothing here was investigated. Labels `[E]` `[I]` `[C]` `[J]`.

---

1. **Does the finite-`N_c` drift follow the `O(1/N)` particle-filter bias
   structure?** `[C]` A genuinely mechanistic question, and the one that would
   turn this replication into a contribution. `[E]` Not pursued: brief §14
   forbids turning this into a theory investigation, and answering it needs a
   change-of-measure argument, not more populations. **Opens if** the survey
   returns a clean monotone drift at two or more `zeta`.

2. **`results/boundary_aggregate.csv` has `zeta = 0.1, 0.2, 0.7` rows and no
   `N_c`, `T`, `dtau_mult` or resampling column.** `[E]` Found by the numerics
   investigator; the lead's own scan covered only `research/tasks/**/results/`
   and missed it. `[E]` Registered as `EV-DATA-BOUNDARYCSV-001`, reproducibility
   `unknown_recoverable`. **Opens as a metadata-recovery task**, never as a reuse
   shortcut — C5 was killed for exactly that.

3. **One local timing probe at `zeta = 0.10, L = 64, N_c = 512, lambda = 0.040`.**
   `[E]` The investigator's recommended next check. It would replace this
   package's weakest input (`f_lam`, a 4x extrapolation) with a measurement, and
   it costs minutes. `[E]` Not run: `/research` is read-only and any local pilot
   needs prior human approval (`RESOURCE_POLICY.md` §3). Carried to
   `RECOMMENDATION.md` as a Gate-A option.

4. **No Ruche `MaxRSS` exists anywhere in this repository.** `[E]` One `sacct`
   line on one completed task would be the first, and would settle memory sizing
   for the whole programme. In `RUCHE_RUNBOOK.md` §3. **Opens on the day.**

5. **The small-batch penalty at `N_c = 128, 256`.** `[E]` The lead applies 1.25
   and 1.15 upward; the investigator would carry the flat plateau value down,
   noting that at `L = 64` the measured `N_c = 64` and `256` rates are *below*
   the `L = 64` constant. `[E]` Unresolved, recorded in `COST_MODEL.md` §3, and
   it changes no `--time` in the package. **Opens** when this campaign returns
   `wall_s` at those rungs, which will settle it for free.

6. **`R` versus `N_c` as the binding budget.** `[E]` At the `zeta = 0.35` anchor
   `R_req` was 51–65 against 48 run. `[I]` If this survey returns `INCONCLUSIVE`
   broadly, the same is true at the new `zeta` and the follow-up is an
   `R`-campaign, not an `N_c`-campaign. `[E]` Pre-registered as F8; **not** a
   rerun inside this task.

7. **Whether `zeta = 0.70` needs a grid extension upward.** `[E]` Its grid
   brackets `phi` in `[0.08, 1.13]`; a boundary exponent above 1.13 would put the
   crossing off the grid. `[E]` No canonical claim proposes `phi > 1`. **Opens
   only if** this campaign reports `ABOVE_GRID` there — which is a
   pre-registered, detectable outcome, not a silent failure.

8. **`L = 96` and `L = 128` at the new `zeta`.** `[E]` Out of scope by
   instruction. `[J]` Worth noting that `L = 128, N_c = 4096` is at the edge of
   what `cpu_long`'s 168 h `MaxTime` can run as one job, and `N_c = 8192` at
   `L = 128` is not runnable at all under this architecture — a hard limit that
   bears on any later high-`zeta` production plan.

9. **`OBS-ACTIVITY-001` is still `needs_audit`.** `[E]` Not touched; this task
   uses only `OBS-CMI-001`.
