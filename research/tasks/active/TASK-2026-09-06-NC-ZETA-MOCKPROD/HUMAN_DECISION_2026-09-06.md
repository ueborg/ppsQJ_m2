# HUMAN_DECISION — 2026-09-06

Researcher decision on the mock-production package. Recorded per
`CONDITIONAL_SUBMISSION.md` ("whichever branch is taken, record it") and
charter §4.4. Labels `[E]` `[I]` `[C]` `[J]`.

**Task state remains `READY_FOR_HUMAN_SUBMISSION`. No agent submitted anything.**

---

## What the researcher decided

1. `[E]` **The `L = 64` top-rung power limitation is ACCEPTED.** The limit
   stated at the top of `RECOMMENDATION.md` stands as a known and accepted
   scope bound, not as an open defect.
2. `[E]` **The three `zeta` waves are NOT staged.** The advisory gates in
   `HUMAN_SUBMISSION.md` rows 6–14, and the §2 wave ordering in
   `RUCHE_RUNBOOK.md`, are **waived by the researcher**. All committed arms go
   in one submission session.
3. `[E]` **All 15 committed / unconditional arms are approved for immediate
   manual submission by the researcher.**
4. `[E]` **`conditional/M_z070_nc2048` stays CONDITIONAL and BLOCKED.** It is
   not in this submission. Its release condition in `CONDITIONAL_SUBMISSION.md`
   is unchanged and still binds: release only if the `512 -> 1024` rung at
   `zeta = 0.70` classifies `CLEARLY_TOO_SMALL` or `STILL_CHANGING`.
   Held spend: **861.6 core-hours, 42.6 % of the full design.**
5. `[E]` **The cost-model uncertainty is accepted as-is.** The optional local
   `f_lam` timing probe is **not required** beforehand.
6. `[E]` **No scientific design change.** Lambda grids, `R = 16`, the `N_c`
   ladder, the seed ledger and `ANALYSIS_SPEC.yaml` are untouched by this
   decision. Nothing in the frozen set was modified.

## What the waiver in item 2 costs, stated plainly

`[E]` The advisory gates existed for two reasons, and the researcher's decision
overrides both knowingly:

- `[E]` **Wave-1-first was a packaging tripwire.** Going cheapest-first meant a
  packaging fault would surface on a 4 core-hour arm rather than a 431
  core-hour one. Releasing everything at once removes that tripwire.
  `[J]` Mitigation: the recommended order is still cheapest-first, so a fault is
  still *observable* on the cheap arms first even though the expensive arms are
  already queued behind it. Cancelling a queued array is cheap; the arms are
  idempotent and may be requeued freely.
- `[E]` **The `R = 16` resolution question.** If every rung in waves 1–2 returns
  `INCONCLUSIVE`, the correct response is to raise `R`, not to spend wave 3's
  837 core-hours. That branch is now spent in advance. `[C]` This is a real and
  accepted risk, not a resolved one: it remains possible that the full 1 160.3
  core-hours returns `INCONCLUSIVE` at every rung and the survey's answer is
  "`R` binds, not `N_c`". `[J]` That would still be a valid negative result and
  must be reported as one.
- `[E]` **`RUCHE_RUNBOOK.md` §2b — the `rho(0.70)` timing readback — is ALSO
  waived by item 5**, since it gated `M_z070_nc1024` on reading wall times back
  from `M_z070_nc128`. `[C]` `rho(zeta)` still has no production-regime
  validation anywhere in this repository and `rho(0.70)` is understated by 7–9 %
  (`COST_MODEL.md` §4). The `--time` margins absorb this by design (`>= 1.6 x`
  pessimistic), but the readback that would have *measured* it is not being
  taken beforehand. `[J]` `sacct` after the fact still settles it, and
  `FALSIFICATION_PLAN.md` F13 still applies whichever way the model errs.

`[E]` None of the above changes any number in the package. They are the
pre-registered reasons the gates existed, recorded here so the decision is
legible later.

## Validation re-run at decision time

`[E]` Re-run immediately before hand-off, on the current working tree:

| check | result |
|---|---|
| `bash shared/run_preflight.sh` | **ALL ARMS PASS** — 21/21 on each of the 14 production arms, 19/19 on `E_dtau_z010`, 21/21 on the held conditional arm; 0 failures |
| `tools/negative_controls.py` | 14 of 14 injected faults rejected with the required code |
| `tools/smoke_test.py` | 13 of 13 synthetic cases classified as constructed |
| `tools/check_predecessor.py` | predecessor isolation OK, 22 files verified, none modified or added |
| `analysis/mockprod_analysis.py` | runs clean on zero returned results; reports `MISSING` throughout and imputes nothing |
| `research/tools/validate_state.py` | 0 errors (1 pre-existing unrelated warning on `CB-MIPT-001`) |
| `research/tools/validate_task.py` | **1 error, `T2`** — see below |

`[E]` `P10` refits `rate35` and `rho` from raw stored results on every preflight
run and passed to within 0.5 % of the literals in `tools/cost_model.py`: the
cost-model literals have **not** drifted since the package was built.

### The one open validator error, and why it was not "fixed"

`[E]` `validate_task.py` reports:

> `ERROR T2 [TASK_MANIFEST.yaml] still contains template placeholders`

`[E]` This is a **false positive and provably so**. `T2` greps for four literal
placeholder strings. The only hit is inside an *amendment reason* at
`TASK_MANIFEST.yaml:91`, which quotes the very placeholder string that was
removed from `NOVELTY_GATE.md` on 2026-09-05, in order to record that it was
removed. The audit trail describes the fix and is therefore matched by the check
that motivated the fix.

`[J]` It was left alone deliberately. Editing it means either mangling a
verbatim quotation in an amendment record, or adding an amendment whose own
reason text would trip the same check. The record's fidelity is worth more than
a green line, and this error has **no bearing whatsoever** on submission
correctness — nothing in the `T2` path touches an arm, a manifest row, a seed, a
pack, a scheduler parameter or the analysis. `[E]` It is recorded here rather
than silently cleared.

## Scale of what was approved

`[E]` 15 arms, **6 080 populations**, **2 557 array tasks**,
**1 160.3 core-hours** (1 624.4 pessimistic at `x1.40`).
`[E]` Held back: 1 arm, 432 populations, 432 tasks, 861.6 core-hours.
`[E]` Full design would have been 2 021.9 core-hours; this release is
**57.4 %** of it.

## Submission commands

`[E]` The exact scheduler command lines were delivered to the researcher in the
session response, **not written into this package**. `RUCHE_RUNBOOK.md` §2
leaves `<your submit command>` as a placeholder by design and preflight `P12`
enforces that no scheduler call exists in any executable file here. That
invariant is preserved: this file adds none.

`[E]` **`research/RESOURCE_POLICY.md` §4 is unaffected by this decision.** The
researcher's approval authorises the researcher. It does not authorise any agent,
at any stage, gate or approval level, and none acted.
