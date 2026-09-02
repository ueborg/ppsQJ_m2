# Validation record

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief §15.

**Every result below was re-read from disk after the file was generated.** The
predecessor readiness task twice reported state it had not actually re-read, and
its first `READY` verdict was false because testing happened in a dirty working
tree with hidden untracked dependencies. Nothing here is inferred from "the
build step should have done X".

---

## 1. Repository validators

| check | result |
|---|---|
| `research/tools/validate_state.py` | **exit 0** — 0 errors, 1 warning (pre-existing) |
| `research/tools/validate_resource_policy.py` | **exit 0** — 0 errors, 0 warnings |
| `research/tools/test_model_routing.py` | **exit 0** |
| `research/tools/test_workflow_regressions.py` | **exit 0** — 25/25 passed |
| `.claude/hooks/test_guard_research.py` | **exit 0** — 62/62 passed |
| `research/tools/validate_task.py <this task>` | **exit 0** — 14 legacy findings, 1 warning, reported not failed |

`validate_task.py` is written for a full `/research` task (Stages 0–9, red team,
novelty gate, twelve slop warnings). This is a numerical preparation child task
and does not carry those artifacts, exactly as
`TASK-2026-09-01-SMCRUCHE-READY` does not: run against the predecessor it
reports the same class of legacy findings and also exits 0. There is no
regression here, and the missing artifacts are a task-class mismatch rather than
a skipped stage.

## 2. Test suite

`.venv/bin/python3 -m pytest tests/<file> -q`, run per file:

| file | result |
|---|---|
| `test_backward_pass_sector.py`, `test_bias_aware_planner.py` | passed (22 passed in this batch alongside `test_doob_wtmc.py`) |
| `test_exact_backend.py` | **6 passed** |
| `test_gaussian_trajectory.py` | **2 passed** |
| `test_lowrank_jump_update.py` | **8 passed** |
| `test_newton_solver.py` | **7 passed** |
| `test_production_entry.py` | **19 passed** |
| `test_statistical_diagnostics.py` | **15 passed** |
| `test_topological.py` | 10 passed, **1 failed** |
| `test_doob_wtmc.py` | 2 passed, **7 failed** |
| `test_exact_benchmark.py` | long-running (> 10 min); not completed within this session |

**89 passed, 8 failed.**

### The 8 failures are pre-existing and are not on this campaign's path

They were reproduced **identically** from a `git archive` of commit `26bcf06`,
the commit immediately before this task, in which none of this task's files
exist:

```
FAILED tests/test_doob_wtmc.py::test_2_zeta_to_zero_concentrates_no_click_sector
FAILED tests/test_doob_wtmc.py::test_3_single_mode_exact_formulas
FAILED tests/test_doob_wtmc.py::test_4_commuting_case_backward_pass_and_rates
FAILED tests/test_doob_wtmc.py::test_5_partition_function_and_moment_consistency
FAILED tests/test_doob_wtmc.py::test_6_click_count_distribution_matches_weighted_born
FAILED tests/test_doob_wtmc.py::test_7_entanglement_entropy_decreases_with_zeta
FAILED tests/test_doob_wtmc.py::test_9_procedure_c_differs_from_qs_but_doob_matches_ab
    RuntimeError: Non-finite log-denominator at t=0.0000: log_denom=-inf
    pps_qj/doob_wtmc.py:249

FAILED tests/test_topological.py::test_backward_pass_ZT_nontrivial_at_zeta_less_than_1
    AssertionError: Expected: 1.0 ± 1.0e-06     tests/test_topological.py:190
```

Both concern the **Doob-transformed backward pass**, which this campaign does
not use. The module import closure of `support/instrumented.py` was enumerated
at runtime and is:

```
pps_qj, pps_qj.cloning, pps_qj.gaussian_backend, pps_qj.overlaps,
pps_qj.types, pps_qj.parallel, pps_qj.parallel.grid_pps,
pps_qj.parallel.worker_clone_pps
```

`pps_qj.doob_wtmc` is **not** among them. So these failures neither were caused
by this task nor can affect its results — but they are a standing defect in the
repository and are **flagged here rather than passed over**. They are outside
this task's scope to fix; fixing them inside a sampling-budget task would
violate §8.

## 3. Static checks on the package

| check | result |
|---|---|
| `py_compile` on all 22 `.py` files in the task | **all OK** |
| `bash -n` on all 12 `.sh` files | **all OK** |
| `bash -n` on all 5 `submit.slurm` | **all OK** |
| `git diff --check` on the staged tree | **clean** |

`git diff --check` initially reported every manifest row as trailing whitespace,
because Python's `csv` module writes CRLF by default (the predecessor's
manifests are CRLF for that reason). Fixed by passing `lineterminator="\n"` in
`tools/build_arms.py` and rewriting `frozen_inputs/predecessor_populations.csv`
the same way; re-checked from disk afterwards and confirmed clean, with zero
CRLF in all five manifests.

## 4. Manifest and array-range audit

Verified **independently of the preflight**, by re-parsing each `manifest.csv`
and each `submit.slurm` from disk:

| arm | rows | `--array` | expected | partition | `--time` | `--mem` |
|---|---:|---|---|---|---|---|
| `armA512` | 48 | `0-47%64` | `0-47` **OK** | cpu_long | 12:00:00 | 3G |
| `armA1024` | 32 | `0-31%64` | `0-31` **OK** | cpu_long | 24:00:00 | 5G |
| `armB` | 288 | `0-287%64` | `0-287` **OK** | cpu_med | 03:00:00 | 2G |
| `armC` | 96 | `0-95%64` | `0-95` **OK** | cpu_long | 12:00:00 | 3G |
| `armA2048_optional` | 16 | `0-15%64` | `0-15` **OK** | cpu_long | 48:00:00 | 9G |

## 5. Seed audit

```
allocated: 480, distinct: 480, range 30100000-30500015          OK
predecessor seeds: 2116, max 20384063; overlap = 0              OK
```

Disjointness is structural, not merely observed: the lowest new seed exceeds the
highest existing seed by 9.7 million. See `SEED_LEDGER.md`.

## 6. No scheduler call anywhere executable

Every `.py` and `.sh` file in the task was scanned line by line for anything in
command position matching `sbatch`, `srun`, `salloc` or `qsub`. **Zero hits.**
`preflight.py` additionally re-asserts this about `run_preflight.sh` at runtime
and refuses to pass otherwise. `submit.slurm` is data for the human to submit;
`RUCHE_RUNBOOK.md` is prose.

## 7. Preflights — positive controls

All five arms, in the working tree: **exit 0**, 18 `OK` checks each, no `FAIL`,
no `WARN`. `analysis-spec sha256` reads
`e79e34365475824200667aeb3a8f250563705c453a309a0daa3ff397622d1f14` in all five.

## 8. Preflights — negative controls

Ten faults injected into a scratch copy. **Every one makes the preflight exit
non-zero**, and the pristine package still exits 0 afterwards:

| # | injected fault | exit | reported |
|---:|---|---:|---|
| 1 | `--array=0-46` against a 48-row manifest | 1 | `--array 0-46%64 does not match the manifest's 48 rows` |
| 2 | two bytes appended to `support/instrumented.py` | 1 | `does not match its recorded sha256` |
| 3 | `support/instrumented.py` removed (the predecessor's real failure) | 1 | `bundled instrumented.py missing` + `run_cell.py's imports do not resolve` |
| 4 | `--partition` line deleted | 1 | `declares NO --partition; the scheduler would pick cpu_short and kill the job` |
| 5 | `cpu_med` against a 12 h request | 1 | `--time=12:00:00 exceeds partition cpu_med MaxTime of 4 h` |
| 6 | `--time=05:00:00` against a 7.04 h pessimistic task | 1 | `is below the pessimistic slowest task` |
| 7 | `--mem=1G` against a 1202 MB peak | 1 | `is below 1.5x the estimated 1202 MB peak` |
| 8 | predecessor seed `20192000` injected | 1 | `seeds in the fresh block … FAIL` |
| 9 | `lambda = 0.3050`, off the frozen stencil | 1 | `lambda on frozen stencil: [0.3032, 0.305]` |
| 10 | `dtau_mult = 12.0`, the corpus convention | 1 | `CERTIFIED 6.0; the historical corpus used 12.0 and is NOT poolable` |

Fault 9 also tripped `R equal across lambdas`, which is the intended
belt-and-braces behaviour.

## 9. Clean tracked-only clone test

This is the check the predecessor's false `READY` verdict lacked.

```bash
git archive HEAD | tar -x -C <empty dir>
```

Confirmed **absent** from that tree, by direct `[ -e ]` test:

```
absent (good): research/tasks/active/TASK-2026-08-30-SMCSTAT/analysis
absent (good): research/tasks/active/TASK-2026-08-30-SMCSTAT/scratch
absent (good): research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/arm1/results
absent (good): research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/arm2/results
```

With `PPSQJ_REPO` deliberately **unset**, all five preflights exit 0 in that
tree, and the runtime block resolves everything inside the archive:

```
OK  bundled instrumented.py   <clean>/…/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/support/instrumented.py
OK  bundle manifest           <clean>/…/support/BUNDLE_MANIFEST.json
OK  pps_qj package            <clean>/pps_qj/__init__.py
OK  bundle sha256             0a33c4034cda70ea...  (matches manifest)
OK  import instrumented+pps_qj+numpy
OK  --array / --partition / --time / --mem
OK  run_preflight.sh has no scheduler call
```

## 10. Tiny local smoke, run inside the clean tracked-only tree

Three rows at `L = 16, T = 8, N_c = 8` — 0.28 s of CPU in total, far inside the
`RESOURCE_POLICY.md` §3 local budget, and not a scientific pilot. It exercises
the full runtime chain and nothing else.

```
[env] instrumented <clean>/…/support/instrumented.py
[env] pps_qj       <clean>/pps_qj/__init__.py
[ok] idx=0 L=16 N_c=8 wall=0.02s -> …/SMOKE_00000.json
[ok] idx=1 …  [ok] idx=2 …
```

**Idempotence confirmed:** re-running index 0 printed `[cached] …` and did not
recompute. A resubmitted array will therefore top up a partial run rather than
redo it.

`analyse_arm.py` then ran on the smoke output and produced the per-cell block
correctly.

**One real defect was found by this smoke and fixed:** the split-half
reproducibility diagnostic returned `NaN ± NaN` at `R = 3`, because each half
needs ≥ 2 populations for its own variance to exist. Every production cell here
has `R ≥ 32`, so it was unreachable in practice — but a silent `NaN` reads as a
diagnostic result rather than an unmet precondition. `shared/analyse_arm.py` now
prints `not computed (R = n < 4)`, and `analysis/combined_analysis.py`'s S1 now
**fails** rather than silently passing in that case. Both were re-compiled and
the arms regenerated afterwards.

## 11. The combined analysis runs end to end

Executed against the frozen predecessor snapshot with **zero** new results —
exactly the state it will be in before the campaign returns. It runs to
completion, degrades correctly (`F1 -> NOT EVALUATED (rung missing)`; both
stencils reported incomplete), and independently reproduces ARM2's published
numbers:

```
   N_c    R   mean CMI       SEM    variance      VIF   N_eff
    64   64    0.51957   0.02494  3.9811e-02    71.79    0.89
   128   64    0.42059   0.02354  3.5474e-02   146.53    0.87
   256   64    0.29932   0.01679  1.8049e-02   177.48    1.44

            pair     Delta      SEM                95% CI  |D|<tau_step?
         64->128  -0.09898  0.03384    [-0.1667, -0.0330]             no
        128->256  -0.12127  0.02875    [-0.1786, -0.0653]             no

   gamma (64+128+256) = +0.571 CI=[+0.108,+1.019]
   B = +17.920 CI=[+12.884, +22.923]   (descriptive)
```

`gamma = +0.571` and `B = +17.920` match `ARM2_FINAL_ANALYSIS.txt` exactly, from
an independent implementation and an independently rebuilt input file. The
`COMBINED_RESULTS.json` this dry run produced was **deleted**, so no
result-shaped artifact predating the data is left in the task.

## 12. Frozen-input fidelity

`frozen_inputs/predecessor_populations.csv` (528 rows,
`971d272a1aa3b0f4861975475490f4dc…`) reproduces the published ARM1 and ARM2
per-cell statistics digit for digit — the table is in `INPUTS_LEDGER.md` §
"Verification that the snapshot is faithful". Re-verified from disk after the
line-ending rewrite.

## 13. Scope invariants

| invariant | check |
|---|---|
| no predecessor archive modified | `git status --porcelain` shows zero modified tracked files anywhere |
| `research/state/**` untouched | not written; guard rule G1 would have denied it |
| production path unchanged | `PRODUCTION_PATH_UNCHANGED.md`; bundle sha256 `0a33c403…` identical to the predecessor's |
| `main` untouched | committed on `smccert-integration` |
| nothing submitted | guard rule G4 fired twice during this session and blocked the Bash calls; no scheduler command was ever executed |
| commit confined to the task | all 73 files under `research/tasks/active/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/`; nothing outside |

---

## Outstanding, and honestly stated

- **`tests/test_exact_benchmark.py` did not finish** inside this session's time
  budget. It is a long-running benchmark and was not observed to complete. It is
  not implicated by anything here, but it was not verified either.
- **The 8 pre-existing `doob_wtmc` / `topological` failures remain.** Not caused
  by this task, not on its execution path, not fixed here.
- **Ruche partition limits are hard-coded from the predecessor's record**
  (`cpu_short` 1 h, `cpu_med` 4 h, `cpu_long` 7 d) and were **not** re-verified
  against the live cluster, because this session has no cluster access and must
  not obtain any. `RUCHE_RUNBOOK.md` §2 gives the `sinfo` / `scontrol` commands
  to confirm them before submitting.
- **The L = 64 rate has never been measured on Ruche.** Two independent
  derivations agree to 1.3 %; that is not a measurement. The runbook's §4
  single-task check exists to catch it.
