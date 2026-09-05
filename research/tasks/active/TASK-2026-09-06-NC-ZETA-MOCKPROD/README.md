# TASK-2026-09-06-NC-ZETA-MOCKPROD

Mock-production survey of finite-`N_c` effects across `zeta`.

**Terminal state: `READY_FOR_HUMAN_SUBMISSION`.** No agent submitted anything
and no agent may — `research/RESOURCE_POLICY.md` §4, unconditionally, at every
stage and gate. `research/state/**` was not written. **The predecessor
`TASK-2026-09-06-NC-ZETA-STAGE1` is untouched**, verified byte for byte by
`tools/check_predecessor.py` and by preflight `P15` on every arm.

Labels `[E]` `[I]` `[C]` `[J]`.

---

## The question

> **As `N_c` increases at fixed `zeta`, how much do the `CMI(lambda)` curves and
> their rough cross-`L` crossing locations still move?**

At `zeta in {0.10, 0.20, 0.70}`, `L in {32, 48, 64}` with `T = L`, over
`N_c in {128, 256, 512, 1024, 2048}`, nine `lambda` per `zeta`, `R = 16` matched
independent populations per cell. `zeta = 0.35` is a **reference row** and is not
recomputed.

`[E]` **This is a survey. It certifies nothing.** The deliverable is a
qualitative per-rung status in `{CLEARLY TOO SMALL, STILL CHANGING, ROUGHLY
STABLE, INCONCLUSIVE}` and a "smallest `N_c` that looks adequate" row with a
caveat. `ROUGHLY STABLE` means the curves overlap within the resolution of an
`R = 16` survey and **never** means convergence.

## Read this first, if you read one thing

`[E]` **The red team killed all five candidates**, C1 on its **analysis half
only**; the measurement half survived every attack and no population changed.
Its four repairs are applied. `[E]` Applying them exposed a limit that had been
invisible: at `R = 16` the minimum detectable effect at `L = 64` is **7.9–13.3 %
of CMI**, against measured top-rung drifts at `zeta = 0.35` of **0.1–2 %**. So
the `1024 -> 2048` question is answerable at `L = 32` and `48` and **not at
`L = 64`**, and that is not fixable by tuning `R`. `RECOMMENDATION.md` §1.


`[E]` **The campaign is 1 160.3 committed core-hours, plus one arm worth 861.6
core-hours held behind an interlock.** That single rung — `zeta = 0.70`,
`N_c = 2048` — is **42.6 % of the full design**, and the interlock's `hold`
branch fires on two of four outcomes, including the one most likely to be
misread as needing more compute. `CONDITIONAL_SUBMISSION.md`.

## Three things established at zero new compute

1. `[E]` **`rate ~ N_c^0.1871` is refuted.** The law adopted by
   `TASK-2026-09-03-NC-PLATEAU-CALIBRATION` is contradicted by rungs that
   campaign itself has since returned: `L=64, N_c=4096` measured 5.05 ms against
   6.57 predicted; `N_c=8192` 5.11 against 7.48; `L=128, N_c=2048` 22.45 against
   31.7. At `L in {32,48,64}` the measured rate is flat in `N_c` to ±9 %.
   Re-checked on every preflight run.
2. `[E]` **There is nothing to reuse.** 5 185 stored populations scanned; **zero**
   exact-compatible cells exist at `zeta in {0.10, 0.20, 0.70}`. The `zeta = 0.7`
   rows that do exist are architecture benchmarks with no `resample_scheme` and
   no `dtau_mult`.
3. `[E]` **All three `lambda` grids are adopted unchanged and are strictly wider
   than the independently re-derived law-agnostic bracket rule on both sides**,
   at all three `zeta`. Both open positions of `DISP-PHI-001` are strictly
   interior with `>= 2` grid points beyond each and a worst margin of
   6.9 `tau_lambda` to the nearer grid end.

## Seven errors this task found in its own work

`[E]` Charter §4.4 makes these first-class outputs, so they are on the front
page rather than in a footnote. Full detail: `VALIDATION.md` §4.

1. `[E]` **The `lambda`-rate exponent was chosen in the wrong direction.** The
   draft adopted `-0.35` under a comment claiming it was conservative; it is the
   least conservative estimate available. Caught by the numerics investigator;
   corrected to `-0.50`.
2. `[E]` **`P2` was a tautology** — it "checked" the discretisation identity by
   comparing `K` with `K`. The negative control corrupted `T` and the check
   agreed with the corruption. Replaced by the checkable invariant `T = L`.
3. `[E]` **The frozen classification thresholds could not return their own
   null.** On synthetic data with an injected drift of exactly zero, the rule
   said `STILL_CHANGING`. `max|z|` over ~18 comparisons is 2.5–3.2 under the
   null and the threshold was 2. Corrected to a multiplicity-corrected
   `z_crit = 3.451`, recorded as `ANALYSIS_SPEC.yaml` `amendment_1`, found
   **before any real datum existed**.
4. `[E]` **The conditional arm could not have run at all.** It sits one
   directory deeper than the rest and the `submit.slurm` template hard-coded
   `../shared/run_pack.py`, which does not resolve there — so the 861.6
   core-hour arm would have died immediately on every array task. Nineteen
   preflight checks passed on it while it was unrunnable. New check `P17`
   resolves the path from the arm; new control `N14` breaks it.
5. `[E]` **The headline recommendation row had no rule.** "Smallest `N_c` that
   looks adequate" was nowhere in the frozen spec, so the script invented
   first-stable-wins; with a real monotone drift injected it printed `128` while
   its own rung table said `STILL_CHANGING`. Found by the red team **by running
   the code**.
6. `[E]` **The `INCONCLUSIVE` gate fired on the wrong side of an "or"** — a
   minimum where the spec says "either", on one rung of the pair. An `L = 64`
   curve at six times the noise threshold read `ROUGHLY_STABLE`.
7. `[E]` **The analysis loader would have ingested scratch data.** The red team's
   fabricated populations, written to `scratch/` as instructed, made `P14` report
   27 false duplicates — and the analysis used the same glob, so they would have
   entered the real curves as measurements. `scratch/` is now excluded
   everywhere and smoke case `S9` asserts it.

## Validation

```
bash shared/run_preflight.sh                          ALL ARMS PASS (21/21, 19/19 control)
.venv/bin/python3 tools/negative_controls.py          14 of 14 injected faults rejected
.venv/bin/python3 tools/smoke_test.py                 13 of 13 synthetic cases
.venv/bin/python3 tools/check_predecessor.py          22 files, none modified, none added
.venv/bin/python3 tools/cost_model.py                 literals within 0.5 % of raw data
.venv/bin/python3 tools/inventory_existing.py         0 exact-compatible existing populations
.venv/bin/python3 analysis/mockprod_analysis.py       runs clean on an empty corpus
```

`[E]` One `validate_task.py` T2 error remains and is a **false positive caused
by this task's own amendment prose**; it is explained in `VALIDATION.md` §4b
rather than patched, because the ledger has one supported writer.

## Files

| file | what it is |
|---|---|
| `CHARTER.md` | the question, what this task is not, kill criterion. **frozen** |
| `PROBLEM_MEMO.md` | Stage 1, including the strongest case *against* the task. **frozen** |
| `SOURCE_REGISTER.md` | Stage 0 scope. **frozen** |
| `CANDIDATES.md` | five design options, eleven fields each. **frozen**, one heading-format amendment |
| `FALSIFICATION_PLAN.md` | F1–F13, class T0 versus class C. **frozen**, contains no results |
| `ANALYSIS_SPEC.yaml` | the frozen analysis rules, plus `amendment_1` |
| `LAMBDA_GRID_DECISION.md` | why the grids are adopted unchanged, and the residual exposure at `zeta = 0.70` |
| `COST_MODEL.md` | the rate model, the refutation, and where it is conjecture |
| `REUSE_AND_DEDUP_AUDIT.md` | why there is nothing to reuse |
| `SEED_LEDGER.md` | 6 512 seeds from 37 000 000, disjoint from everything |
| `RUCHE_RUNBOOK.md`, `HUMAN_SUBMISSION.md`, `CONDITIONAL_SUBMISSION.md` | the package and its gates |
| `VALIDATION.md` | every check, and the four failures found during construction |
| `RECOMMENDATION.md` | the decision gate |
| `RESEARCH_MEMO.md` | Stage 9 synthesis |
| `REDTEAM.yaml` | Stage 8, all nine attacks per candidate |
| `agent_reports/numerics.md` | the investigator's frozen first pass |

## What this task may never conclude

`[E]` No `lambda_c(zeta)`. No boundary exponent; `DISP-PHI-001` and
`DISP-WINDOW-001` stay open. No `N_c^req(zeta)` and no fit of one. No transfer
of a `zeta = 0.35` coefficient. Nothing about `L > 64` or `N_c > 2048`. **No
`N_c` here is converged and nothing here is certified.**
