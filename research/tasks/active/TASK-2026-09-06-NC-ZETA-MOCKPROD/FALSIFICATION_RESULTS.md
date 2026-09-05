# FALSIFICATION_RESULTS — TASK-2026-09-06-NC-ZETA-MOCKPROD

Outcomes of `FALSIFICATION_PLAN.md`. **Written after the plan froze and never
merged back into it.** Labels `[E]` `[I]` `[C]` `[J]`.

`[E]` **Class-T0 targets were attempted. Class-C targets were NOT**, because
they need the campaign's returned data and this task stops at
`READY_FOR_HUMAN_SUBMISSION`. Their status below is `not attempted`, which is
the pre-registered outcome, not an omission.

---

## T0 targets — attempted

### F1. "The `N_c^0.1871` rate law is correct." — **KILLED**

`[E]` Refit from every stored `zeta = 0.35`, `dtau_mult = 6` population. The law
was fitted on three `L = 128` rungs; the rungs returned since disagree with it
well beyond the 25 % kill threshold:

| cell | law predicts | measured | error |
|---|---:|---:|---:|
| `L=64, N_c=4096` | 6.57 ms | 5.05 | +30 % |
| `L=64, N_c=8192` | 7.48 ms | 5.11 | +46 % |
| `L=128, N_c=2048` | 31.7 ms | 22.45 | +41 % |

`[E]` **Consequence, as pre-registered:** the cost model does not use it. Every
`--time` in this package derives from a flat per-`L` constant instead, and
`cost_model.refit()` re-checks the refutation on every preflight run.
`[I]` The law erred **upward**, so no predecessor job was under-timed by it; the
cost was over-planning, not failure.

### F2. "The grids bracket both open positions of `DISP-PHI-001`." — **SURVIVES**

`[E]` All six predictions strictly interior, `>= 2` grid points beyond each on
each side, worst margin to a grid **end** 6.9 `tau_lambda`, and every grid
strictly wider than the independently re-derived bracket rule on both sides at
all three `zeta`. Checked per arm by `P9`; negative control `N8` shows `P9`
rejecting a narrowed grid.
`[C]` **The residual exposure is recorded, not eliminated**: the `zeta = 0.70`
grid brackets `phi in [0.08, 1.13]`, so an exponent above 1.13 would put the
crossing off the grid. No canonical claim proposes one, and the pre-registered
`ABOVE_GRID` class detects it.

### F3. "Some exact-compatible data already exists." — **KILLED**

`[E]` 5 185 stored populations, 136 distinct cells, **zero** exact-compatible.
`[E]` The investigator additionally found `results/boundary_aggregate.csv`
carrying `zeta = 0.1/0.2/0.7` rows, **outside** the directory the lead's scan
covered. It has no `N_c`, `T`, `dtau_mult` or resampling column and a different
`lambda` grid; not poolable. `[J]` The conclusion did not change but the audit
would have been incomplete, and that is worth recording as a near-miss.

### F4. "The preflight checks are decorative." — **PARTIALLY CONFIRMED, then fixed**

`[E]` **`P2` was decorative.** It recomputed `K` from a row's own four numbers
and compared it with `K` from the same four numbers. Negative control `N13`
corrupted `T` and **the check agreed with the corruption**. Replaced by the
checkable design invariant `T = L`; `N13` now fires.
`[E]` **And a second gap: nothing checked that the runner path resolved.**
Nineteen checks passed on `conditional/M_z070_nc2048` while its hard-coded
`../shared/run_pack.py` pointed at a directory that does not exist from there,
so that 861.6 core-hour arm would have died on every array task. Added `P17`,
which resolves the path from the arm, and `N14`, which breaks it.

`[E]` All 14 controls now fire with the expected code. `[J]` The target found
two real defects, which is what a falsification target is for -- and both were
absences rather than errors, which is the harder kind to see.

### F5. "The `zeta` timing ratios are an artifact of one `L`." — **SURVIVES, scoped**

`[E]` Per-`L` ratios: 0.342–0.449 at `zeta = 0.10`, 0.587–0.639 at 0.20,
2.085–2.372 at 0.70. Maximum spread 1.31, below the 1.5 kill threshold.
`[E]` The maximum over `L` is adopted, which errs upward; the investigator would
adopt `L = 64`. Disagreement recorded in `COST_MODEL.md` §4, not resolved.
`[C]` The dataset remains local, at `N_c <= 500`, with tiny `n_steps`, and with
`lambda` co-varying as `0.51*sqrt(zeta)` throughout. Directional, not exact.

### F6. "The frozen predecessor was disturbed." — **KILLED**

`[E]` 22 files verified byte for byte against the baseline recorded when this
task opened; none modified, none added. Re-run as `P15` on every arm.

### F7. "Packing changes the computation." — **KILLED**

`[E]` `run_pack.py` is byte-identical to the predecessors' and invokes the
byte-identical certified `run_cell.py` once per row **in a fresh process**. No
argument, environment variable, seed or ordering differs. `P11` verifies all
three files by sha256 and `run_cell.py` refuses to start on altered bundle bytes.

---

## Class-C targets — NOT ATTEMPTED

`[E]` F8 (`R = 16` resolves anything), F9 (crossing interiority at the new
`zeta`), F10 (the three `L` agree), F11 (`dtau_mult = 6` adequate at low
`zeta`), F12 (drift monotone in `N_c`), F13 (the cost model predicted the
campaign) — **all `not attempted`**. Each needs the returned data.

`[E]` They were pre-registered before the package was built so that they cannot
be chosen after seeing results. `[J]` F8 is the one to watch: if it fires, the
survey's product is killed and the correct response is an `R` campaign, not a
larger `N_c`.

---

## Two failures found outside the plan

`[E]` Neither was a pre-registered target; both are recorded because §4.4
forbids dropping a negative result on the grounds that nobody asked for it.

`[E]` **The frozen classification thresholds could not return their own null.**
`tools/smoke_test.py`, on synthetic data with an injected drift of exactly zero,
returned `STILL_CHANGING`. `max|z|` is a maximum over ~18 comparisons and sits
at 2.5–3.2 under the null; the threshold was 2. Corrected to
`z_crit = 3.451`, recorded as `ANALYSIS_SPEC.yaml` `amendment_1`. **Found before
any real datum existed.**

`[E]` **The headline recommendation row had no rule, and the `INCONCLUSIVE`
gate fired on the wrong side of an "or".** Both found by the red team by
executing the shipped script on injected data; both would have produced a
confidently wrong headline from a perfectly good campaign; both are load-bearing
on an 861.6 core-hour interlock. Repairs R1 and R2, `ANALYSIS_SPEC.yaml`
`amendment_2`, smoke cases `S10`–`S13`.

`[E]` **`z_crit` was under-corrected threefold.** The normal Šidák point 3.451
has a true null exceedance of 3.05 %, not 1 %, because `z` divides by SEMs
estimated from `R = 16`. Reproduced by the lead (3.05 % against the reviewer's
3.04 %) before changing anything. `z_crit` is now simulated at the actual `R`;
the value is 3.88.

`[E]` **The task's declared kill criterion cannot fire.** `CHARTER.md` §4 kills
the product if every rung at every `zeta` is `INCONCLUSIVE`, which needs median
SEM/|CMI| above 0.05; the corpus says `<= 0.030` at every `N_c >= 256` at
`R = 16`. `[I]` The criterion is inoperative as written. `[J]` That is good news
about `R = 16` at the low rungs and bad news for a criterion that was supposed to
be able to kill the task — it is recorded rather than quietly re-tuned, and
`CHARTER.md` is frozen so it is not edited.

`[E]` **The analysis loader would have ingested scratch data.** The red team's
fabricated populations under `scratch/` made `P14` report 27 false duplicates,
and `analysis/mockprod_analysis.py` used the same glob — so those files would
have entered the real `CMI(lambda)` curves as measurements. Excluded everywhere;
smoke case `S9` asserts it. `[J]` Found by accident, by another agent doing
exactly what it was told to do in the directory it was told to use.
