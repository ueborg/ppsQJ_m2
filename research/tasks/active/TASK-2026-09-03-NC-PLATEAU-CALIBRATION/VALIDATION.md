# VALIDATION — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Every check run on this package, with its result — **including the one that
fails and is not repaired.**

Reproduce all of it from the repository root:

```bash
.venv/bin/python3 research/tasks/active/TASK-2026-09-03-NC-PLATEAU-CALIBRATION/tools/run_all_checks.py
```

Nothing in this file, or in anything it runs, can submit an HPC job. One of the
checks is that nothing in the package *can*.

---

## Summary

**25 of 26 automated checks pass.** The one that does not is Charter Stage 8
(§11 below) and it is an unrepaired gap, not a false alarm.

---

## 1. Syntax and import

| check | result |
|---|---|
| `bash -n` on all 74 shell and job scripts (24 `submit.slurm`, 24 `run_preflight.sh`, 24 `analyse_results.sh`, 2 in `shared/`) | **PASS** |
| `py_compile` on all 81 Python files | **PASS** |
| `import numpy + instrumented + pps_qj` **in a subprocess**, exactly as `run_cell.py` imports them (numpy 2.4.3) | **PASS** |

`[J]` The import check runs in a subprocess so a broken import cannot poison the
checking process. It is the check that would have caught the
`ModuleNotFoundError` that killed a predecessor campaign's first Ruche job.

## 2. The design regenerates identically

| check | result |
|---|---|
| `tools/build_arms.py` is **idempotent**: 34 manifests and job scripts byte-identical after regeneration | **PASS** |
| `tools/build_conditional.py` runs | **PASS** |
| `K = ceil(2 lambda (L-1) T / dtau_mult)` against the `n_steps` every completed run recorded for itself | **PASS — exact in all 1 896 cases** |

`[E]` The last one matters more than it looks: it verifies the discretisation
formula against 1 896 independent instances of the sampler's own arithmetic,
across every `(L, lambda)` in the corpus. Campaign E's `K = 816/408/204` comes
from this formula, not from the design document that proposed it.

## 3. Exact compatibility — the check the reuse ledger rests on

```
tools/reproduce_check.py 2                                            PASS
```

`[E]` Two completed predecessor populations
(`MOCK-LOWLAMBDA-EXTENSION/lowlamL32`, seeds 32002014 and 32002021) re-executed
through **this task's modified** `shared/run_cell.py`:

| quantity | result |
|---|---|
| `per_clone_CMI`, 1 024 values | **BIT-IDENTICAL**, both cells |
| `n_steps`, `n_nonfinite`, `n_distinct_anc_final`, `brentq_fallbacks` | **exactly equal**, both cells |
| `cmi_weighted_mean`, `cmi_within_var`, `gess_final`, `ess_cum_final`, `ess_frac_mean` | agree to **≤ 1.7e-14 relative** |

`[E]` **A finding the check was not aimed at.** The residual is x86-versus-arm64
summation order in numpy's pairwise reductions — the stored values were reduced
on Ruche, these on arm64. `[I]` **A stored AGGREGATE in this corpus is not
bit-reproducible on a different architecture even though the trajectory is.**
Nothing in the repository said so, and anyone re-deriving a published mean on a
different machine needs to know it.

`[E]` The check's criterion was tightened to match — exact on the trajectory and
on integers, `1e-12` relative on derived reductions, which is three orders
tighter than anything that could hide a real change and three orders looser than
the architectures actually differ by. `[J]` A criterion loosened after seeing a
result is a move this project rightly distrusts; the reason is written at the
line that changed and recorded in `POST_FREEZE_EVENTS.md`.

## 4. The runtime end to end

```
tools/smoke_test.py                                                   PASS
```

`[E]` Three toy populations (`L = 12`, `N_c = 8`, `dtau_mult ∈ {3,6,12}`) run
through the real `run_cell.py`, checking:

- every field the **predecessor** wrote is still written, under the same key;
- every **new** instrumentation field is written (19 of them);
- per-clone arrays have length `N_c`; every per-window history has length `K`;
- `logw_carry_var_final` equals `Var(log final_weights)` **recomputed
  independently** of the writer;
- `final_weights` are normalised;
- a completed row is **not recomputed** — re-running an array is idempotent and
  tops up a partial one rather than redoing it;
- `dtau_mult` moves `K` **exactly**, at all three values;
- `delta_tau` as recorded is the **actual** step `T/K`, not the nominal
  `dtau_mult/(2 lambda (L-1))` it came from (the `ceil` rounds it down) —
  recording the nominal value would trap anyone reconstructing the schedule from
  a result file;
- `analyse_arm.py` reports **three** cells, one per discretisation: it does not
  pool them;
- the frozen analysis runs to completion on **zero** new results and reports
  empty sections as empty.

## 5. Preflights

| check | result |
|---|---|
| all **17** immediate arms exit 0 with `PREFLIGHT PASSED` | **PASS** |
| all **7** conditional arms report `BLOCKED` and exit **3** | **PASS** |

`[E]` Each preflight checks its manifest **against what the builder actually
produces**, row by row — not against a second copy of the design. That is the
change from the predecessor's preflight, which restated its grid as literals and
could therefore drift from its own builder while both looked right.

## 6. Negative controls — the preflight fails when it should

```
tools/negative_controls.py                            16/16 FIRED, PASS
```

| control | injected fault |
|---|---|
| N01 | an off-grid `lambda` |
| N02 | a duplicate seed |
| N03 | `zeta = 0.30` |
| N04 | `dtau_mult = 12` in a production arm |
| N05 | `T != L` |
| N06 | `resample_scheme = multinomial` |
| N07 | a **predecessor's** seed |
| N08 | one manifest row missing |
| N09 | `--array` not matching the manifest |
| N10 | `--time=00:05:00` |
| N11 | `--mem=100M` |
| N12 | `--mem=200` — the **unit trap** |
| N13 | `--partition=cpu_short` |
| N14 | `--cpus-per-task=4` |
| N15 | a drifted `run_cell.py` |
| N16 | an injected scheduler call in `analyse_results.sh` |

`[J]` **A limitation of this result, stated rather than glossed.** N01–N08 all
fire through the *same* check — "manifest == frozen design" — because the
design-identity comparison is the first and strongest gate and catches them
before the field-level checks run. The field-level checks (`zeta`, `T == L`,
`dtau_mult`, resampler, seeds) are a second line of defence that would fire only
if the **builder itself** were changed. That is the correct architecture, and it
does mean eight of the sixteen controls exercise one mechanism.

### The Slurm `--mem` unit table, re-verified

`[E]` `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` asked for this to be checked
again after the predecessor's parser bug.

| `--mem` | parsed | note |
|---|---:|---|
| `2G`, `2g` | 2.000000 GiB | |
| `512M` | 0.500000 | |
| `600M` | 0.585938 | the value the older parser **raised** on |
| `2048` | **2.000000** | **no suffix means MEGABYTES**; the older parser read 2 048 GiB and failed **OPEN** |
| `1024K` | 0.000977 | |
| `1T` | 1024.0 | |
| `` , `nonsense` | 0.0 | fails **closed** |

## 7. Nothing in this package can submit

| check | result |
|---|---|
| no executable file carries a scheduler or remote-launch verb (155 files scanned) | **PASS** |
| every conditional job script carries the `GATE_RELEASED_*` interlock | **PASS** |

`[E]` The three files that *name* the forbidden verbs — `preflight.py`,
`negative_controls.py`, `run_all_checks.py` — are excluded from the scan because
naming them is their job; in all three the verbs are assembled from string
fragments so no source line is itself a command.

`[E]` `.claude/hooks/guard_research.py` (rule **G4**) fired **twice** during this
run, both times on read-only commands whose *string content* contained a
forbidden verb: a `grep` searching for scheduler references, and a heredoc whose
documentation text listed the transfer commands. `[J]` Both denials were correct
behaviour from a hook that does not try to infer intent, and should not.

## 8. Duplicate compute, reuse and seeds

```
tools/dedup_scan.py                                                   PASS
```

| check | result |
|---|---|
| all 19 reuse-ledger entries match the populations **on disk**, exactly | **PASS** |
| no cell is pushed past its target `R` by recomputation (84 cell decisions) | **PASS** |
| 3 280 immediate + 464 conditional seeds vs **4 408** already allocated elsewhere | **no overlap** |
| repository seed ceiling 32 203 023 < this campaign's floor 33 000 000 | **structurally disjoint** |
| no physical cell is built by two different arms | **PASS** |

`[E]` The seed scan reads every `manifest.csv` **and** every result JSON in the
repository, including arms that were built and never run — which is exactly
where a collision would otherwise hide.

`[E]` `REUSE_LEDGER.csv` is written by this scan, so it cannot drift from the
data it describes.

## 9. Cost and memory

| check | result |
|---|---|
| every rate carries its rung, its `n`, and the median/p90/max it came from | **by construction** |
| `--time` ≥ 1.6× the **pessimistic** slowest task, on every arm | **PASS** (preflight) |
| partition rule recomputed, not asserted; `cpu_short` never used | **PASS** (preflight) |
| `--mem` ≥ 1.2× the measured model, on every arm | **PASS** (preflight) |

### A defect found in the inherited model, and a second one found in the fix

`[E]` **Defect 1.** The `--mem` model every predecessor package used
(`128 + 2 N_c per_clone`) was never checked against a running process, and
`TASK-2026-09-01-SMCRUCHE-READY` describes its output as "the measured 732 MB
peak" — which is exactly what the formula returns for `L = 96, N_c = 512`. No
`MaxRSS` from any Ruche job appears anywhere in this repository. Direct
`ru_maxrss` at 15 cells shows it **under**-predicts at seven of them;
`L = 64, N_c = 2048` measures 1 694 MB against a predicted 1 202 MB, and that
arm shipped `--mem=2G`.

`[E]` **Defect 2, found while fixing defect 1.** Repeated probes of the *same*
cell are **not reproducible**, and at every cell probed twice the **second probe
came in higher**:

| cell | probes, MB | spread | note |
|---|---|---:|---|
| `L = 128`, `N_c = 2048` | 3 482.5 · 3 521.7 · **6 275.9** | **1.80** | three probes; the first two agree to 1.1 % and are still 1.80× below the third |
| `L = 64`, `N_c = 8192` | 3 547.0 · **4 593.8** | 1.30 | |
| `L = 64`, `N_c = 4096` | 2 032.6 · **2 747.1** | 1.35 | |
| `L = 96`, `N_c = 1024` | 2 006.3 · **2 139.8** | 1.07 | |
| `L = 96`, `N_c = 2048` | 2 200.8 | — | probed once; a second run's output line was truncated by a concurrent write and its peak is unrecoverable, so it is **not** counted |

`[E]` **Every cell probed more than once varies.** The `L = 128` triple carries
the sharper lesson: two probes agreeing to 1.1 % were still 1.80× below a third,
so **agreement between two probes is not evidence of a bound either**.

`[I]` Nothing about the sampler changed between any of them. `ru_maxrss` is a
high-water mark and depends on when the allocator happens to release the
transient copies selection makes.

`[I]` **A single probe of a cell is not a bound.** Treating one as a bound is how
a 31-hour job dies at hour 20 with OOM. `[E]` The model therefore takes the
**maximum over probes**, keeps the old formula as a **floor**, and applies a
1.35× margin on top of both. `D_L128_nc2048` consequently requests **9G**, not
the 6G a single lower probe would have justified, and `A_L64_nc8192` **7G**
rather than 6G.

`[J]` This is the check that most nearly did not happen. The first round of
probes was taken once per cell, the numbers looked clean, and the package would
have shipped 6G against a cell subsequently observed to reach 6.3 GB. Running
the probe again is the whole of what caught it.

`[E]` **A record error this audit also caught, in this same table.** An earlier
version of `tools/cost_model.py` listed `L = 96, N_c = 2048` as
`[2200.8, 2200.8]` — two agreeing probes. Only **one** complete probe of that
cell exists; a second run's output line was truncated by a concurrent write and
its peak is unrecoverable. The duplicate was a transcription error on my part,
and it was doing real work in the argument: it was the one cell that appeared to
*reproduce*. It is corrected to a single probe, and the `L = 128` cell is
corrected from two probes to the three that exist. **Neither correction changes
any `--mem` request** — both maxima are unchanged — but the table is evidence
about reproducibility, and a fabricated agreement in it is exactly the kind of
thing it exists to warn against.

`[E]` **Honest limitation:** these are macOS `ru_maxrss` and the cluster is
Linux. `RUCHE_RUNBOOK.md` §7 asks for one `sacct … MaxRSS` line on a completed
`D` task — it would be the first such measurement of this sampler on the cluster
in existence.

## 10. Repository hygiene

| check | result |
|---|---|
| `git diff --check` | **clean** |
| `research/state/**` untouched | **PASS — no modification** |
| no predecessor task directory modified (26 scanned, by mtime) | **PASS** |

## 11. Engine validators — and the one unrepaired failure

| validator | result |
|---|---|
| `validate_state.py` | **PASS** (0 errors, 1 pre-existing warning) |
| `validate_task.py` | **1 error** — see below |
| `validate_resource_policy.py` | **PASS** (0, 0) |
| `test_model_routing.py` | **PASS** |
| `test_workflow_regressions.py` | **PASS** — 25/25 |
| `test_guard_research.py` | **PASS** — 62/62 |
| **`validate_redteam.py`** | **REFUSES THE REPORT — correctly** |

### The failure, and why it is not fixed

```
ERROR R3  reviewer saw the lead summary;
          Stage 8 requires review independent of the affirmative reasoning
ERROR T4  [REDTEAM.yaml] fails validate_redteam.py
```

`[E]` `REDTEAM.yaml` declares `lead_summary_seen: true`, because the red-team
pass was run **by the lead, against the lead's own design, with full sight of
its reasoning.** No independent investigator and no independent red-team agent
ran at any point in this task.

`[E]` **The flag was not set to `false`.** Doing so would make both validators
green by misdescribing how the review was produced. `[J]` **Charter Stage 8 is
not satisfied by this run**, `INDEPENDENCE_LEDGER.yaml` records the same thing
without softening it, and `RECOMMENDATION.md` states that every "survives"
verdict in that file should be treated as **unreviewed** at the human gate.

`[E]` What the self-red-team nevertheless did: it killed two design decisions,
at a combined ~386 core-hours (`R = 24 → 48` in campaign A; campaign B2 rebuilt
from three `lambda` to seven). `[J]` That is more than nothing and it is not a
substitute. A self-red-team is a checklist, not a check.

`[E]` `INDEPENDENCE_LEDGER.yaml` names the three passes that would repair it,
the most valuable being an independent numerics pass that tries to **locate** the
four-rung `L = 96` ladder behind the predecessor's `chi2 = 10.54`.

## 12. What was NOT validated

`[E]` Stated plainly, because an unlisted gap reads as a covered one.

- **The cluster.** No Ruche run, no queue behaviour, no real `MaxRSS`, no
  confirmation that `%64` grants per-array concurrency. The runbook's §3
  accounting check exists because none of that could be verified here.
- **`N_c` above the largest measured rung.** Every rate at `L = 64` above 2 048,
  `L = 96` above 512 and `L = 128` above 1 024 is an extrapolation with the
  measured exponent `G = 0.1871`, applied deliberately conservatively.
  `COST_MODEL.md` §3 marks these `[C]`.
- **The three modelled memory cells** — `B_L64_cross_nc1024`, the `B2_L48_*`
  arms, and campaign E's small-`N_c` cells. All are ≤ 1 515 MB modelled against
  1–2G requests.
- **Linux allocator behaviour**, per §9.
- **Any physics.** Y1–Y8 of `FALSIFICATION_PLAN.md` are `not yet attempted` and
  `FALSIFICATION_RESULTS.md` says so rather than writing outcomes for data that
  do not exist.
