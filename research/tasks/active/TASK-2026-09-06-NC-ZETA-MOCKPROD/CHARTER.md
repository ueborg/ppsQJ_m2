# CHARTER — TASK-2026-09-06-NC-ZETA-MOCKPROD

Mock-production survey of finite-`N_c` effects across `zeta`. Labels `[E]` `[I]`
`[C]` `[J]` per Research Charter §2.

---

## 1. The question

> **As `N_c` increases at fixed `zeta`, how much do the `CMI(lambda)` curves and
> their rough cross-`L` crossing locations still move?**

Asked separately at `zeta in {0.10, 0.20, 0.70}`, at `L in {32, 48, 64}` with
`T = L`, over the `N_c` ladder `{128, 256, 512, 1024, 2048}`, with `zeta = 0.35`
carried as an already-measured reference row.

`[J]` This is a **practical production-planning survey**, not a certification.
The deliverable is a qualitative per-rung status and a "smallest `N_c` that
looks adequate" recommendation with its caveat.

## 2. What this task deliberately is NOT

`[E]` It is **not** `TASK-2026-09-06-NC-ZETA-STAGE1`, whose frozen design
answers a stricter question — certified `N_c^req(zeta)` under a TOST equivalence
test at `tau_lambda = 0.004`. That task is **untouched** by this one: no file in
it is read for design authority, edited, or re-run. `PREDECESSOR_ISOLATION.md`
records the checks.

Specifically excluded, by instruction and by design:

| excluded | why |
|---|---|
| TOST equivalence at `tau_lambda = 0.004` | this is a survey; no equivalence test is run and none is reported |
| a precision phase boundary | `L <= 64` locators are not `lambda_c(zeta)` |
| any fit of `N_c^req(zeta)` | three `zeta` cannot support a law and none is proposed |
| the word "certified" | reserved for a certification task that this is not |
| `N_c > 2048`, `L in {96, 128}` | out of scope |
| a new algorithm | the certified production path is used byte-identically |
| a broad theory or literature investigation | `research/RESOURCE_POLICY.md` §5.5, and the brief's §14 |

## 3. Hypotheses, stated so they can fail

`[C]` **H1.** At every `zeta` in the set there exists a rung `N*` in
`{128, ..., 2048}` above which the `L = 48` vs `L = 64` `CMI` difference curve
stops moving by more than this survey's own statistical resolution.

`[C]` **H2.** `N*` is not the same at all three `zeta`.

`[J]` Neither is required to be true for the task to succeed. `H1` failing at a
`zeta` — the curves still moving at 2048 — is the **most decision-relevant**
outcome available, because it tells the researcher that production at that
`zeta` cannot be sized from this ladder at all.

## 4. Kill criterion

`[E]` The task's own product is killed, and reported as killed, if the analysis
returns `INCONCLUSIVE` for **every** rung at **every** `zeta` — i.e. `R = 16`
independent populations cannot separate adjacent-rung movement from noise
anywhere. In that event the correct output is a negative result plus the
`R` required to do better, **not** a rerun at larger `R` inside this task.

## 5. Standing rules this task binds itself to

- `[E]` Uncertainty comes from **independent populations only**. `ESS`, `GESS`,
  `VIF` and founder counts are diagnostics — `DEC-MASTER-METRIC-001`.
- `[E]` Finite-`N_c` movement is **drift**, never bias: the `N_c -> infinity`
  target is unknown.
- `[E]` No smoothing, no interpolation replacing a measurement, no imposed
  monotonicity, no value-based exclusion. Under-covered cells are flagged.
- `[E]` `dtau_mult != 6` rows are a discretisation control and are **never**
  pooled with the production corpus.
- `[E]` `R` is matched across every cell that is compared.
- `[E]` No agent submits an HPC job. `research/RESOURCE_POLICY.md` §4.
- `[E]` `research/state/**` is not written.

## 6. Terminal state

`READY_FOR_HUMAN_SUBMISSION`, with the researcher's manual commands in
`RUCHE_RUNBOOK.md` and the per-arm gates in `HUMAN_SUBMISSION.md`.
