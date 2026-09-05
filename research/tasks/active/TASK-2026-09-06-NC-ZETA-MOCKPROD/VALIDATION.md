# VALIDATION — TASK-2026-09-06-NC-ZETA-MOCKPROD

What was checked, what failed during construction, and what remains unchecked.
Labels `[E]` `[I]` `[C]` `[J]`.

---

## 1. The suite

| command | expected | status |
|---|---|---|
| `bash shared/run_preflight.sh` | `ALL ARMS PASS PREFLIGHT.` — 21/21 per production arm, 19/19 for the control | **PASS** |
| `.venv/bin/python3 tools/negative_controls.py` | 14 of 14 injected faults rejected, each with the expected code | **PASS** |
| `.venv/bin/python3 tools/smoke_test.py` | 13 of 13 synthetic cases classified as constructed | **PASS** |
| `.venv/bin/python3 tools/check_predecessor.py` | predecessor isolation OK, 22 files | **PASS** |
| `.venv/bin/python3 tools/cost_model.py` | all literals within 0.5 % of the raw data | **PASS** |
| `.venv/bin/python3 tools/inventory_existing.py` | 0 exact-compatible existing populations | **PASS** |
| `.venv/bin/python3 analysis/mockprod_analysis.py` | runs clean on an empty corpus, reports every cell MISSING | **PASS** |
| `.venv/bin/python3 research/tools/validate_task.py <TASK_DIR>` | see §5 | |
| `.venv/bin/python3 research/tools/validate_redteam.py REDTEAM.yaml` | see §5 | |

`[E]` The preflight is not a build-time artifact: it **refits the cost-model
literals from raw stored data on every run** and fails if they have drifted.

## 2. The 21 preflight checks

`P1` matched `R` and cell count · `P2` `T = L` and the `K` the packer costed
with · `P3` array range matches `packs.csv` · `P3b` packs tile the manifest
exactly once · `P4` `--time >= 1.6 x` pessimistic slowest pack · `P4b` every
`est_sec` reproduces from the cost model · `P5` `--mem >= 1.35 x` modelled peak
· `P6` smallest partition that fits, `cpu_short` never · `P7` seed disjointness
against every manifest in the repository · `P8` no `zeta = 0.35` row · `P9`
grid identity **and span coverage of both open positions of `DISP-PHI-001`** ·
`P9b` the `phi` range each grid brackets · `P10` cost literals refit · `P11`
bundle integrity by sha256 · `P12` no executable scheduler call anywhere ·
`P13` discretisation discipline · `P14` no duplicate of a stored population ·
`P15` the frozen predecessor is untouched · `P16` no certification language
where it would be a claim · `P17` the runner path in `submit.slurm` resolves
from **this** arm.

## 3. The 14 negative controls

`[E]` Each injects one fault into a **copy** of a real arm and requires
rejection with a named code. Nothing under the task directory is modified.

`N1` short `--time` → `P4` · `N2` `--mem` with no suffix, which Slurm reads as
**megabytes** → `P5` · `N3` `cpu_short` → `P6` · `N4` array range mismatch →
`P3` · `N5` seed collision with `MOCK-PRODUCTION`'s block → `P7` · `N6`
unmatched `R` → `P1` · `N7` a `zeta = 0.35` row → `P8` · **`N8` a narrowed
`lambda` grid → `P9`** · `N9` `dtau_mult = 12` in a production arm → `P13` ·
`N10` `est_sec` drift → `P4b` · `N11` a dropped pack → `P3b` · `N12` a row
duplicating a stored population → `P14` · `N13` `T != L` → `P2` · **`N14` a
runner path that does not resolve → `P17`**.

`[J]` `N8` is the one that matters most: it reproduces the failure that stopped
`TASK-2026-09-05-NC-ZETA-CALIBRATION` at Gate A, and shows this package's `P9`
catching it.

## 4. Four things that failed during construction, and what changed

`[E]` **4.1 The `lambda`-rate exponent was chosen in the wrong direction.** The
draft `COST_MODEL.md` adopted `A_LAM = -0.35` under a comment claiming it was
the conservative choice. It is not: this task's grids sit *below* the reference
line, so the correction grows as the exponent becomes more negative, making
`-0.35` the *least* conservative estimate available. Caught by
`agent_reports/numerics.md` Q2. **Changed to `-0.50`**, which roughly doubles
the low-`lambda` correction at `zeta = 0.10` and moves `--time` in the safe
direction. Cost rose from 2 019 to the shipped figures.

`[E]` **4.2 `P2` was a tautology.** It "verified the discretisation identity" by
recomputing `K` from a row's own `(L, lambda, T, dtau_mult)` and comparing it to
`K` computed from the same four numbers. Negative control `N13` corrupted `T`
and the check **agreed with the corruption**. `P2` now asserts the design
invariant `T = L`, which is checkable, and `N13` now fires.

`[E]` **4.3 The frozen qualitative thresholds could not return their own null.**
`tools/smoke_test.py`, on synthetic data with an injected drift of **exactly
zero**, classified the rung `STILL_CHANGING`. `max|z|` is a maximum over ~18
comparisons and sits at 2.5–3.2 under the null, so the frozen threshold of 2 was
below the noise floor of its own statistic and would have reported almost every
rung as still moving whatever the data did. Corrected to a multiplicity-corrected
`z_crit = 3.451` at `n = 18`, recorded as `ANALYSIS_SPEC.yaml` `amendment_1` via
`task_phase.py amend`, which records the superseded hash. `[E]` **Found before
any real datum existed**, on data with a known answer.

`[E]` **4.4 The analysis loader would have ingested scratch data.** The red
team's own fabricated populations, written to `scratch/` as instructed, made
`P14` report 27 false duplicates — and `analysis/mockprod_analysis.py` used the
same glob, so those files **would have entered the real `CMI(lambda)` curves as
measurements**. `scratch/` is now excluded in the preflight, the inventory tool,
the cost-model refit and the analysis loader, and smoke case `S9` asserts a
scratch population is invisible to the loader. `[J]` This one was found by
accident, by another agent doing legitimate work in the directory it was told to
use. It is the strongest argument in this file for running the suite after every
change rather than once at the end.

### `[E]` **4.5 The conditional arm's runner path did not resolve.**

`[E]` `conditional/M_z070_nc2048` lives one directory deeper than every other
arm, and the `submit.slurm` template hard-coded `../shared/run_pack.py`. From
that arm the path points at `conditional/shared/`, which does not exist, so the
**861.6 core-hour arm would have died immediately on every array task** with
"No such file or directory". `[I]` It is the same class of failure that killed
the first Ruche job of `TASK-2026-09-01-SMCRUCHE-PACKFIX`: a path that is
correct in one layout and silently wrong in another.

`[E]` The path is now computed per arm (`os.path.relpath`), giving `../shared`
for the fifteen committed arms and `../../shared` for the conditional one. New
preflight check **`P17`** verifies the path resolves **from that arm** and that
`run_cell.py` sits beside it; new negative control **`N14`** breaks the path and
requires rejection.

`[J]` Nineteen checks passed on that arm while it was unrunnable. That is the
argument for checks that resolve a path rather than inspect one.

### `[E]` **4.6 The red team killed the analysis half, and was right.**

`[E]` The independent reviewer executed the shipped script on injected data at
the noise level measured in the real `zeta = 0.35` corpus, and found two defects
that 21 preflight checks and 9 smoke cases had not:

- `[E]` **The headline row had no rule.** "Smallest `N_c` that LOOKS adequate"
  was specified in the brief in words and **nowhere in the frozen
  `ANALYSIS_SPEC.yaml`**, so the implementation invented first-stable-wins and
  never revised the pick when a higher rung moved. With a real monotone
  3–4 %/rung drift injected it printed `0.10 | 128 | low` while its own rung
  table read `256 -> 512 STILL_CHANGING`. `[I]` Directional, not random:
  relative SEM falls with `N_c`, so the lowest rung is always the most likely to
  read stable for want of power, and the bug converted that into the cheapest
  recommendation.
- `[E]` **The `INCONCLUSIVE` gate fired on the wrong side of an "or".** The spec
  says "at `L=48` **or** `L=64` exceeds 0.05"; the code took the **minimum**,
  and only on the higher rung's curve. An `L = 64` curve at six times the
  threshold classified `ROUGHLY_STABLE`. That is precisely the "a broken rung
  reads as stable" path the `INCONCLUSIVE`-first ordering exists to close.

`[E]` Both are load-bearing: `CONDITIONAL_SUBMISSION.md` releases or holds
**861.6 core-hours** on `classify_rung`'s verdict.

`[E]` Two further findings, both reproduced by the lead before anything changed:
**`z_crit` was under-corrected 3x** (the normal Šidák point 3.451 has a true null
exceedance of 3.05 %, not 1 %, because `z` divides by SEMs estimated from
`R = 16`; simulated 99th percentile 3.88), and the **declared kill criterion
cannot fire** (it needs median SEM/|CMI| above 0.05 and the corpus says
`<= 0.030` at every `N_c >= 256`).

`[E]` Repairs R1–R4 applied; `ANALYSIS_SPEC.yaml` `amendment_2`. New smoke cases
`S10`–`S13` cover the recommendation chain, the `INCONCLUSIVE` break and a
single unusable `L`. `S10` **failed on first run** against the repaired code and
exposed a second bug in the fix — the chain scan stopped at the first moving
rung, which is wrong because drift is not required to be monotone.

`[J]` The smoke suite was written by the same author as the analysis and did not
find either defect. The red team did, by **running** the code rather than
reading it. That is the strongest argument in this file for an independent
reviewer who executes.

## 4b. Two known validator artefacts, explained rather than hidden

`[E]` **`validate_task.py` T2 fires on `TASK_MANIFEST.yaml`, and it is a false
positive caused by this task's own prose.** The check looks for template
placeholder strings; line 83 of the ledger is an amendment *reason* that quotes
the literal `'<statement>'` while explaining that that literal had been removed
from `NOVELTY_GATE.md`. `[E]` The ledger is append-only and
`research/tools/task_phase.py` is its only supported writer, so the wording is
**not** hand-edited out. `[J]` If the researcher wants a clean validator run, the
one-line fix is to reword that reason in the ledger; it changes no record of
what happened. Recorded here rather than silently patched.

`[E]` **`NOVELTY_GATE.md` needed two amendment passes**, because the lead did not
run `validate_task.py` before the first. Both are recorded with reasons and an
authoriser. `[J]` Routine work should not need an `amend` at all; running the
validator earlier would have avoided both, and that is the process lesson.

## 5. Task-level validators

`[E]` Run and recorded in §1 of `README.md`. `validate_task.py` checks the
mandated stage artifacts, the phase ledger, frozen-artifact hashes (`M5`), the
falsification plan/results separation (`F1`, `F2`), the analysis-spec estimator
fields (`N1`–`N6`), the task-evidence bookkeeping (`E1`–`E3`) and the novelty
gate (`G1`–`G3`). `validate_redteam.py` enforces that all nine mandated attacks
carry a verdict per candidate and that `lead_summary_seen` is false.

## 6. What is NOT validated, and cannot be here

- `[C]` **Memory on the cluster.** Every `--mem` descends from local macOS
  `ru_maxrss`. **No Ruche `MaxRSS` exists anywhere in this repository.** One
  `sacct` line would fix it for the whole programme; `RUCHE_RUNBOOK.md` §3.
- `[C]` **`f_lam` below its calibrated window.** A ~4x extrapolation at
  `zeta = 0.10, lambda = 0.040`. Named as the package's weakest input in
  `COST_MODEL.md` §5. The probe that would settle it is a Gate-A option, not run.
- `[C]` **Whether `R = 16` resolves anything.** Pre-registered as F8 and
  answerable only by the returned data. The `INCONCLUSIVE` class exists for it.
- `[E]` **External prior art.** No search was performed anywhere in this task, by
  instruction. External novelty is `UNRESOLVED`, not favourable.
- `[E]` **The classifier's intent.** The smoke test shows it behaves as its
  author intended on data its author constructed. It cannot show the intention
  was right — and §4.6 records that it did not: the red team found two defects
  the suite missed. `S10`–`S13` now cover them, but that is a patch on a known
  hole, not evidence the hole is closed.
- `[C]` **The survey's power at `L = 64`.** MDE is 7.9–13.3 % of CMI there
  against measured top-rung drifts of 0.1–2 %. Measured, reported in every
  verdict, and **not fixable by tuning `R`**. `RECOMMENDATION.md` §1.
- `[E]` **`rho(0.70)`** has no production-regime validation anywhere and is
  understated by 7–9 %. `RUCHE_RUNBOOK.md` §2b gates the two large `zeta = 0.70`
  arms on a timing readback.
