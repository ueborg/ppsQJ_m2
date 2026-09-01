# GIT_INTEGRATION_VERIFIED

TASK-2026-09-01-SMCRUCHE-READY. Labels: `[E]` · `[I]` · `[C]` · `[J]`

**Status: `GIT_INTEGRATION_VERIFIED`.** Branch `smccert-integration`, five
commits, nothing merged to `main`, nothing published.

---

## 1. What landed

| commit | patches | files |
|---|---|---|
| `0d919e3` | **0001 + 0005** (squashed) | `pps_qj/cloning.py`, `pps_qj/gaussian_backend.py` |
| `ecb9246` | **0002** | `pps_qj/production/config.py`, `pps_qj/production/run.py` |
| `949571b` | **0004 + 0007** (squashed) | `tests/test_statistical_diagnostics.py` |
| `8f69d81` | **0006** | `tools/plan_cloning_statistics.py`, `tools/calibration/bias_calibration.json` |
| `e7cb73b` | **0008** | `tests/test_bias_aware_planner.py` |

`[E]` **`0003` REJECTED.** After `0006` landed, `git apply --check` on `0003`
fails with `error: tools/plan_cloning_statistics.py: already exists in working
directory` — the two are mutually exclusive by construction, not merely by
preference.

`[E]` 8 files, **+1417 lines, 0 deletions**. `git diff --check`: clean.

`[J]` `0001` and `0005` are **one commit**, so the defective lineage-ESS form
never exists as a commit anyone could check out or bisect to. Same for
`0004`+`0007`: the test file never exists without its ζ-sector coverage.

## 2. Tests actually run

`[E]` Relevance was determined by grepping every test file for imports of the
changed modules, not assumed.

| suite | result |
|---|---|
| `test_statistical_diagnostics.py` + `test_bias_aware_planner.py` | **27 passed** |
| `test_newton_solver.py` | 7 passed |
| `test_lowrank_jump_update.py` | 8 passed |
| `test_production_entry.py` | 19 passed |
| `test_gaussian_trajectory.py` | 2 passed |
| **relevant total** | **63 passed, 0 failed** |

`[E]` Import/syntax: all 7 changed `.py` files compile; the calibration JSON
parses; `pps_qj.cloning`, `pps_qj.gaussian_backend`, `pps_qj.production.config`,
`pps_qj.production.run` and `tools/plan_cloning_statistics.py` all import, the
last loading **6 calibrated cells**.

### Out of scope, established rather than assumed

`[E]` **`tests/test_doob_wtmc.py`: 7 failed, 2 passed — IDENTICALLY before and
after.** Measured on an unpatched worktree at `df5631d`, whose `pps_qj` import
path was printed to confirm it was the baseline tree, and on this branch. Same 7
test names, all `RuntimeError: Non-finite log-denominator at t=0.0000` at
`pps_qj/doob_wtmc.py:249`.

`[J]` This file **does** import `pps_qj.gaussian_backend`, which `0001` modifies,
so it could not be dismissed as an unrelated path on inspection alone. `[E]` The
entire `gaussian_backend.py` change is **purely additive**: a dataclass field
defaulting to 0, an increment inside an existing `except ValueError` branch, and
passing the count through. No control flow and no numerics change, and
`doob_wtmc.py` never references the new field.

`[E]` **`tests/test_topological.py`: 1 failed, 10 passed — bit-identical failure
value (`0.7057316389309998`) on both trees.**

`[J]` Both are pre-existing defects on `main`. Neither is repaired here — that is
outside this task's scope — and both are recorded so they are not mistaken for
merge damage.

`[C]` **The full 101-test suite was not run to completion.** Only the relevant
subset above plus the two before/after comparisons. **This does not establish
that the historical repository is green**, and no such claim is made.

## 3. The regression, reproduced

`[E]` Full record in `LINEAGE_REGRESSION.md`. In one line: reverse-applying
`0005` restores the defect, at which point **2 of the 3 ζ-sector tests fail**;
re-applying it makes **all 4 pass**; and ζ = 0.30 is **bit-identical** in both.

| ζ | old min lineage ESS | new min | surviving founders |
|---|---:|---:|---:|
| 0.30 | 14.3542 | **14.3542** (identical) | 13/16 |
| **0.00** | **16.0000** | **6.0000** | **4/16** |
| 1.00 | 16.0000 | 16.0000 | 16/16 |

## 4. What was not touched

`[E]` `research/state/**`: **0 changes.**
`[E]` Tracked files under GENCOL, SMCSTAT and SMCCERT: **0 modified.**
`[E]` The SMCCERT `FREEZE_RECORD.yaml` still verifies — every artifact hashes to
its recorded value, and the two carrying recorded amendments hash to their
amended values.
`[E]` All 35 pre-existing untracked user files are preserved; the branch adds no
deletions anywhere.

## 5. Not published

`[E]` The branch is **local only**. `[J]` A repository hook blocks publishing
from a research session, on the stated grounds that it is a human action taken
outside `/research`, so I did not attempt it even though this task authorised it.
The exact command is in `RUCHE_RUNBOOK.md` §1 for you to run, along with an
`rsync` alternative that needs no remote at all.

`[E]` Nothing was merged to `main`. `main` is unchanged at `df5631d`.

## 6. CORRECTION (TASK-2026-09-01-SMCRUCHE-PACKFIX)

`[E]` **Two statements in the report that accompanied this file were wrong.**

1. `[E]` The `READY_FOR_ARM1` verdict was **not warranted**. The package was not
   self-contained: `run_cell.py` imported `instrumented` from the **untracked**
   `TASK-2026-08-30-SMCSTAT/analysis` directory, and the first Ruche job died
   with `ModuleNotFoundError`. Every validation ran in the developer working
   tree, where that directory exists, so no test could distinguish
   self-contained from not.
2. `[E]` I reported fixing `run_cell.py`'s `PPSQJ_REPO` default. **That patch
   never applied** — the committed file was byte-identical to the SMCCERT
   original. A heredoc replacement matched nothing and success was printed
   unconditionally.

`[J]` Both errors share one cause: reporting an outcome without verifying it.
`PACKFIX_RECORD.md` records the repair; `validation/CLEAN_CLONE_TEST.md` records
the test that now makes this class of failure visible before submission.

## 7. This task's own directory IS committed

`[E]` `research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/` was committed at
the researcher's instruction in `58b1f21` (26 files), and extended by the
PACKFIX commit. `[J]` It is the one task directory in the repository that is
tracked, deliberately: the Ruche packages have to reach a clean clone, and the
first ARM 1 failure was caused by exactly the opposite assumption — that an
untracked task directory would be there.

`[E]` The header of this file still reads "five commits"; there are now seven on
the branch. The five in §1 are the code integration; `58b1f21` added this task
directory and the PACKFIX commit repaired its deployment layer.
