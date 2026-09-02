# Validation record

TASK-2026-09-02-MOCK-PRODUCTION, brief §16.

**Every result below was re-read from disk after the file was generated.**
Nothing here is inferred from "the build step should have done X". The
predecessor programme twice reported state it had not actually re-read, and one
`READY` verdict was false because testing happened in a dirty working tree with
hidden untracked dependencies.

**Revalidated in full after the matched-R amendment** (`MATCHED_R_AMENDMENT.md`).
Every figure, table and verdict below is from the post-amendment run. §0 records
what the amendment changed and what it provably did not.

---

## 0. The matched-R amendment — revalidation

The amendment is an **analysis** rule. The first thing checked was that it
touches no compute.

**[E] The manifests are byte-identical across it.** All seven `manifest.csv`
files were hashed, `tools/build_arms.py` was re-run, and they were hashed again:
`diff` is empty. The lambda grid, `zeta`, `T = L`, `dtau_mult`, the `L` values,
the `N_c` values, the `R` values, every seed, every array range, every partition
and the bundled sampler are all unchanged, and the seed audit, dedup scan and
all fourteen negative controls were re-run against them.

| new check (brief §6) | result |
|---|---|
| block selection depends only on seed/order, never CMI | **PASS** — same cell built with 5 different CMI assignments (random, sorted ascending, sorted descending, adversarial ±99, all identical) gives byte-identical block membership; also invariant to file-read order, with the paired CMI values travelling with their seeds |
| `R = 24` primary counts at every applicable cell | **PASS** — the synthetic end-to-end run prints an `R` column on every curve row; the set of distinct values across all six curves × 13 lambdas is exactly `{24}` |
| `R = 24` blocks are disjoint | **PASS** — the four blocks of an `R = 96` cell give 96 distinct seeds and cover the cell exactly; the two blocks of an `R = 48` cell are disjoint and cover it |
| `R = 48` splits exactly into 24 + 24 | **PASS** — `n_blocks = 2`, block sizes `[24, 24]`, no third block |
| `R = 96` splits exactly into 24+24+24+24 | **PASS** — `n_blocks = 4`, block sizes `[24, 24, 24, 24]`, no fifth block |
| a short cell cannot masquerade as matched | **PASS** — `R = 23` yields `n_blocks = 0` and `cell_block(c, 0) is None` |
| block A is the 24 lowest seeds, blocks ascend | **PASS** |
| block statistics are the block's, not inherited from the parent | **PASS** — block A mean differs from the parent mean; block SEM uses `R = 24` |
| every arm's `R` is an exact multiple of the block size | **PASS** — read from the seven real manifests: 24, 24, 24, 24, 48, 48, 48 |
| reused ARM-B cells hold `R = 96` with unique seeds per cell | **PASS** — read from `frozen_inputs/armB_populations.csv` |

`tools/test_matched_r.py`, **30 checks, 0 failures**, and it passes identically
inside the clean tracked-only tree.

**[E] The block boundaries are contiguous seed ranges**, as the seed allocation
guarantees. From the real data, at `lambda = 0.2932`:
`A: 30300000–30300023  B: …024–047  C: …048–071  D: …072–095`.

### The amendment changes a verdict, which is why it was needed

**[E]** On the synthetic end-to-end dataset, M3 returns **INCONCLUSIVE**
(5 raw sign changes at `N_c = 1024` against 7 at `N_c = 128`) under the original
unequal-`R` analysis and **KILLED** (5 against 5) at matched `R = 24`. Two of the
seven "extra" sign changes at `N_c = 128` were bought by its having `R = 48`.

**[E]** The same run shows roughness at `L = 64, N_c = 1024` falling from
**18.024** at full `R` to **8.773** at matched `R = 24` — the three reused
`R = 96` points had error bars half their neighbours', which the second
difference read as excess structure in the *curve*.

Neither number is a scientific result; both are demonstrations on fabricated
data that the confound the amendment removes was real rather than theoretical.

---

## 1. Repository validators

| check | result |
|---|---|
| `research/tools/validate_state.py` | **exit 0** — 0 errors, 1 warning (pre-existing `W3` on `CB-MIPT-001`) |
| `research/tools/validate_resource_policy.py` | **exit 0** — 0 errors, 0 warnings |
| `research/tools/test_model_routing.py` | **exit 0** — 109 passed |
| `research/tools/test_workflow_regressions.py` | **exit 0** — 25/25 passed |
| `.claude/hooks/test_guard_research.py` | **exit 0** — 62/62 passed |
| `research/tools/validate_task.py <this task>` | **exit 0** — 14 legacy findings, 1 warning, reported not failed |

`validate_task.py` is written for a full `/research` task (Stages 0–9, red team,
novelty gate, twelve slop warnings). This is a numerical preparation child task
and does not carry those artifacts — exactly as
`TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA` and `TASK-2026-09-01-SMCRUCHE-READY` do
not. The 14 findings are the same class those tasks report and it exits 0. **A
task-class mismatch, not a skipped stage.**

The guard hook fired **five times** during this session and blocked those Bash
calls: twice on `G4` — a scheduler name reaching command position inside a
here-document while *writing* a SLURM script, and again inside a `grep` pattern
while *auditing* for scheduler calls — and three times on `G3`, on recursive
forced deletes of scratch directories. **No scheduler command was ever
executed.** The two `G4` hits are false positives on legitimate preparation
work; both were worked around by using a file-write tool and a Python scanner
rather than by weakening anything.

## 2. Test suite

Run per file. **After the amendment the campaign-scoped subset was re-run** —
the amendment brief asks not to re-run the full repository suite indefinitely
and to record the standing baseline failures separately, which is what §2b does.

| file, re-run after the amendment | result |
|---|---|
| `test_production_entry.py` | **19 passed** |
| `test_statistical_diagnostics.py` | **15 passed** |
| `test_gaussian_trajectory.py` | **2 passed** |
| `test_lowrank_jump_update.py` | **8 passed** |

These four are the ones that touch the campaign's execution path. The full
per-file run below is the pre-amendment baseline and is unchanged by an
analysis-only edit:

| file | result |
|---|---|
| `test_exact_backend.py` | **6 passed** |
| `test_gaussian_trajectory.py` | **2 passed** |
| `test_lowrank_jump_update.py` | **8 passed** |
| `test_newton_solver.py` | **7 passed** |
| `test_production_entry.py` | **19 passed** |
| `test_statistical_diagnostics.py` | **15 passed** |
| `test_backward_pass_sector.py` | **8 passed** |
| `test_bias_aware_planner.py` | **12 passed** |
| `test_topological.py` | 10 passed, **1 failed** |
| `test_doob_wtmc.py` | 2 passed, **7 failed** |

**89 passed, 8 failed** — identical to the count the predecessor recorded
against commit `26bcf06`.

## 2b. Standing baseline failures, recorded separately

### The 8 failures are pre-existing and are not on this campaign's path

They are the same eight the predecessor reproduced from a `git archive` of the
commit before its own work existed:

```
tests/test_doob_wtmc.py  (7)   RuntimeError: Non-finite log-denominator at t=0
tests/test_topological.py::test_backward_pass_ZT_nontrivial_at_zeta_less_than_1
```

Both concern the **Doob-transformed backward pass**, which this campaign does
not use. The import closure of `support/instrumented.py` was enumerated at
runtime in this session and is:

```
instrumented, pps_qj, pps_qj.cloning, pps_qj.gaussian_backend,
pps_qj.overlaps, pps_qj.parallel, pps_qj.parallel.grid_pps,
pps_qj.parallel.worker_clone_pps, pps_qj.types
```

`pps_qj.doob_wtmc` is **not** among them, and neither is any topological module.
So these failures neither were caused by this task nor can affect its results —
but they remain a standing defect in the repository and are **flagged here
rather than passed over**. Fixing them inside a sampling-budget task is out of
scope.

`tests/test_exact_benchmark.py` was **not run**: the predecessor recorded it as
exceeding 10 minutes and it was not attempted here either. It is not implicated
by anything in this package, and it was not verified.

## 3. Static checks on the package

| check | result |
|---|---|
| `py_compile` on all 30 `.py` files in the task | **all OK** |
| `bash -n` on all 16 `.sh` files and 7 `submit.slurm` | **all OK** |
| `import mock_production_analysis` (the amended analysis) | **OK** — `BLOCK=24`, `PRIMARY_BLOCK=0` |
| `git diff --check` on the staged tree | **clean** |
| CRLF audit on all 7 manifests and both frozen inputs | **0 CR bytes** in every file |

`tools/build_arms.py` writes manifests with `lineterminator="\n"`; Python's
`csv` module defaults to CRLF, which makes `git diff --check` report every
manifest row as trailing whitespace. Verified by re-reading the bytes from disk,
not by trusting the flag.

## 4. Manifest and array audit, independent of the preflight

Verified by a **separate script** that re-parses every `manifest.csv` and
`submit.slurm` from disk and imports nothing from the task, so a bug shared with
`preflight.py` cannot hide:

| arm | rows | `--array` | expected | partition | `--time` | `--mem` | λ | R |
|---|---:|---|---|---|---|---|---:|---:|
| `mockL32` | 312 | `0-311%64` | `0-311` **OK** | cpu_short | 01:00:00 | 1G | 13 | 24 |
| `mockL48` | 312 | `0-311%64` | `0-311` **OK** | cpu_med | 02:00:00 | 1G | 13 | 24 |
| `mockL64` | 240 | `0-239%64` | `0-239` **OK** | cpu_med | 03:00:00 | 2G | 10 | 24 |
| `mockL64nc2048` | 72 | `0-71%64` | `0-71` **OK** | cpu_med | 04:00:00 | 3G | 3 | 24 |
| `mockNC128L32` | 624 | `0-623%64` | `0-623` **OK** | cpu_short | 01:00:00 | 1G | 13 | 48 |
| `mockNC128L48` | 624 | `0-623%64` | `0-623` **OK** | cpu_short | 01:00:00 | 1G | 13 | 48 |
| `mockNC128L64` | 624 | `0-623%64` | `0-623` **OK** | cpu_short | 01:00:00 | 1G | 13 | 48 |

Also checked independently, per arm: every lambda on the frozen 13-point grid;
`zeta = 0.35`; `dtau_mult = 6.0`; `T == L`; equal `R` across lambdas; no
duplicate seeds; every seed inside `[31e6, 32e6)`; and — for `mockL64` — that
none of the three ARM-B lambdas appears.

## 5. Seed audit

```
allocated: 2808, distinct: 2808, range 31000000-31612047        OK
existing seeds scanned: 2596, max 30500015, overlap = 0         OK
```

Disjointness is structural, not merely observed: the lowest new seed exceeds the
highest existing seed by 499,985. `tools/existing_seeds.json` is the union of
the predecessor's ledger (2,116) with its own 480 allocations — so this campaign
checks against a ledger that includes the predecessor's fresh block, which the
predecessor's own ledger by construction could not. See `SEED_LEDGER.md`.

## 6. Preflights — positive and negative controls

**Positive.** All seven arms in the working tree: **exit 0**, no `FAIL`, no
`WARN`. `analysis-spec sha256` reads
`a1613a3716b2b33b7d601a5606026bae0f1a57b0e620dd5c8c2c748d731a1f13` in all seven.

**Negative.** Fourteen faults injected one at a time into a scratch copy.
**Every one makes the preflight exit non-zero**, and the pristine package still
exits 0 afterwards:

| # | injected fault | exit | reported |
|---:|---|---:|---|
| 1 | `--array=0-238` against a 240-row manifest | 1 | `--array 0-238%64 does not match the manifest's 240 rows` |
| 2 | two bytes appended to `support/instrumented.py` | 1 | `bundle sha256 … DOES NOT MATCH manifest` |
| 3 | `support/instrumented.py` removed | 1 | `bundled instrumented.py` missing + imports do not resolve |
| 4 | `--partition` line deleted | 1 | `--partition MISSING` |
| 5 | `cpu_short` against a 3 h request | 1 | `MaxTime 1 h vs requested 3 h` |
| 6 | `--time=00:30:00` against a 0.92 h pessimistic task | 1 | `is below the pessimistic slowest task` |
| 7 | `--mem=1G` against a 1202 MB peak (`mockL64nc2048`) | 1 | `is below 1.5x the estimated peak` |
| 8 | predecessor seed `30300000` injected | 1 | `seeds in the fresh block … FAIL` |
| 9 | `lambda = 0.3050`, off the frozen grid | 1 | `OFF-GRID: [0.305]` |
| 10 | `dtau_mult = 12.0`, the corpus convention | 1 | `CERTIFIED 6.0; the historical corpus used 12.0 and is NOT poolable` |
| 11 | `N_c = 512`, off the frozen set | 1 | `manifest contains (L, N_c) outside the frozen sets: [(64, 512)]` |
| 12 | `T = 32` against `L = 64` | 1 | `T == L … L=64:T=32` |
| 13 | an ARM-B lambda re-added to `mockL64` | 1 | `no ARM-B duplication … DUPLICATES ARM B at [0.3032]` |
| 14 | one row deleted, making `R` unequal | 1 | `R equal across lambdas … R = [23, 24]` |

### Two defects this exercise found, and what was done

**[E] Fault 11 crashed with a `KeyError` instead of reporting.** An `(L, N_c)`
off the frozen sets reached the cost loop, which indexes `BASE_MS` and
`NC_FACTOR`, and the preflight aborted with a traceback. The exit code was still
non-zero — it fails closed — but **a traceback is not a report**, and a human
reading it would not learn that the manifest had been hand-edited. Fixed by
moving the membership check ahead of the cost loop and emitting an explicit
message. Re-verified: fault 11 now reports
`manifest contains (L, N_c) outside the frozen sets: [(64, 512)]`.

**[E] The first run of this exercise was invalid** and is recorded rather than
quietly re-run. The scratch copy sits outside the repository, so `pps_qj` did
not resolve and *every* fault "failed" for that reason instead of its own — and
the pristine control failed too, which is what exposed it. Fixed by passing
`PPSQJ_REPO` explicitly to the subprocess. A negative-control suite in which the
control itself fails proves nothing, and the first table it produced has been
discarded.

**One fault was mis-specified and is worth recording.** `--mem=1G` on `mockL64`
does **not** fail, and should not: the 665 MB peak needs 0.97 G, so 1 G is
compliant with the 1.5× rule. The arm requests 2 G for margin. The control was
retargeted to `mockL64nc2048`, where 1 G against a 1202 MB peak is a genuine
violation. Noted because the original table would have looked like a preflight
bug and is not one.

**A parsing detail, fail-closed.** `preflight.py`'s `_gb()` strips only a `G`
suffix, so a request written in megabytes (`--mem=512M`) parses as 0.0 and fails
the memory check. That is the safe direction, but it means the failure message
would read as "far too small" rather than "unparseable". Left as is; all seven
arms request whole gigabytes.

## 7. End-to-end synthetic run of the analysis

The package's preflights validate the *submission*. Nothing in them validates
the *analysis*, so a separate exercise fabricated a plausible result JSON for
**all 2,808 manifest rows** from a closed-form surrogate — one with a deliberate
`L`-crossing near `lambda = 0.27` and a deliberate `N_c` offset — in a scratch
copy, and ran the whole chain.

**[E] Result: every branch executed.** All seven `analyse_results.sh` produced
per-cell blocks; the combined analysis produced all six curves, all six crossing
analyses, the three-`L` `N_c` comparison, the `Delta_N` shape check, all seven
criteria and all four figures; and the criteria returned **SUPPORTED, KILLED,
INCONCLUSIVE and NOT EVALUATED verdicts across the set** — so none of them is a
formality that passes by construction.

**Re-run in full after the matched-R amendment**, with the new sections A0
(block inventory), E2 (ARM-B block sensitivity A/B/C/D) and E3
(`R = 12` historical-precision checks) all executing. Post-amendment the
surrogate gives M1 INCONCLUSIVE, M2 KILLED, **M3 KILLED** (it was INCONCLUSIVE
before the amendment — see §0), and M4/M5/M6/M7 SUPPORTED. The `R` column on
every printed curve row reads 24 throughout, which is the matched-R property
demonstrated on real output rather than only in the unit test.

### The defect it found

**[E] The `Delta_N` bootstrap null was wrong, and it was wrong in the direction
that hides the effect M5 exists to detect.** The null was built by resampling
the observed populations, which carries the observed lambda-dependence into the
"null" and inflates `p`. On a surrogate constructed with a real tilt it returned
`chi2 = 6.106` on 2 dof with `p = 0.572` — against a parametric expectation of
`p ≈ 0.047` — and therefore verdict **C (unresolved)** where the truth was **B
(lambda-dependent)**.

Fixed by shifting each `N_c = 2048` cell's populations by `-(D_j - cbar)` before
resampling, so the null is true by construction while the empirical spread is
preserved. Re-verified on the same data: `p = 0.045`, verdict **B**. The slope
bound remains bootstrapped around the observed data, since that is a confidence
statement rather than a null test.

This is the defect the whole exercise was worth running for, and it would have
been invisible until the real data returned an unresolved verdict that nobody
could distinguish from a genuinely small tilt.

**A second, smaller correction:** M5 originally returned **KILLED** when the
`N_c = 2048` arm had simply not run yet. An unsubmitted arm must never read as a
scientific verdict; it now returns **NOT EVALUATED**, and KILLED is reserved for
an arm that returned data that cannot be used.

**Nothing from this exercise was left in the task.** It ran entirely in the
session scratchpad. The dry-run `MOCK_PRODUCTION_RESULTS.json` and the four
placeholder figures produced by the zero-data run were **deleted**, so no
result-shaped artifact predates the data.

## 8. The analysis degrades correctly with no new results

Run in the working tree against the frozen ARM-B snapshot alone — exactly the
state it is in before the campaign returns:

```
populations loaded: 288 (0 new, 288 frozen ARM-B)
[L=64,N_c=1024]  INCOMPLETE — 3/13 grid points present
[L=32,N_c=1024]  INCOMPLETE — 0/13 grid points present
...
exact common (N_c=1024, N_c=128) cells: 0 of 39
M1..M4, M7 -> NOT EVALUATED    M5 -> NOT EVALUATED    M6 -> SUPPORTED
```

It runs to completion, exits 0, names what is missing at every step, and emits
Figure C **empty with the reason printed on it** rather than filling it by
interpolating the historical corpus.

## 9. Clean tracked-only checkout test

This is the check whose absence made the predecessor's first `READY` verdict
false.

```bash
git write-tree && git archive <tree> | tar -x -C <empty dir>
```

With `PPSQJ_REPO` deliberately **unset**, all seven preflights exit 0 in that
tree and the runtime block resolves everything inside the archive:

```
OK  bundled instrumented.py   <clean>/…/TASK-2026-09-02-MOCK-PRODUCTION/support/instrumented.py
OK  bundle manifest           <clean>/…/support/BUNDLE_MANIFEST.json
OK  pps_qj package            <clean>/pps_qj/__init__.py
OK  bundle sha256             0a33c4034cda70ea...  (matches manifest)
OK  import instrumented+pps_qj+numpy
```

Confirmed **absent** from that tree by direct test — the untracked directories
that killed the first ARM 1 job:

```
absent (good): research/tasks/active/TASK-2026-08-30-SMCSTAT/analysis
absent (good): research/tasks/active/TASK-2026-08-30-SMCSTAT/scratch
```

**[E] A correction to an earlier reading of this test.** The directory
`TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/armB/results` appears in the clean tree, but
`git ls-files` shows it holds **exactly one tracked file, `.gitkeep`**. All 288
ARM-B result JSONs are **untracked** and exist only in this working tree.

That makes `frozen_inputs/armB_populations.csv` **load-bearing, not a
convenience**: a clean checkout has no other route to the reused populations, and
a package that read them from the sibling task would fail on Ruche in exactly the
way the first ARM 1 job did (`ModuleNotFoundError` on an untracked sibling path).
Verified directly: the frozen CSV is present in the archive, the analysis loads
288 populations from it inside the clean tree, and the block inventory prints the
four ARM-B seed blocks there.

The bundled sampler and both frozen inputs are all present in the archive.

## 10. Tiny local smoke, run inside the clean tracked-only tree

Three rows at `L = 16, T = 8, N_c = 8` — **0.10 s of CPU in total**, far inside
the `RESOURCE_POLICY.md` §3 local budget, and not a scientific pilot. It
exercises the full runtime chain through the real sampler and nothing else.

```
[env] instrumented <clean>/…/TASK-2026-09-02-MOCK-PRODUCTION/support/instrumented.py
[env] pps_qj       <clean>/pps_qj/__init__.py
[ok] idx=0 L=16 N_c=8 wall=0.04s   [ok] idx=1 … 0.03s   [ok] idx=2 … 0.03s
```

**Idempotence confirmed:** re-running index 0 printed `[cached] …` and did not
recompute. A resubmitted array tops up a partial run rather than redoing it.

## 11. Cost-model back-test against completed Ruche runs

Predicted / observed ≥ 1 means conservative:

| cell | source | predicted | observed | ratio |
|---|---|---:|---:|---:|
| L=64, N_c=1024, λ=0.2932 | ARM B, 96 runs | 1962 s | 1893 s | 1.036 |
| L=64, N_c=1024, λ=0.3032 | ARM B, 96 runs | 2026 s | 2008 s | 1.009 |
| L=64, N_c=1024, λ=0.3132 | ARM B, 96 runs | 2091 s | 2015 s | 1.038 |
| L=128, N_c=1024, λ=0.3032 | ARM A1024, 32 runs | 46 936 s | 46 937 s | 1.000 |

An earlier draft of the model, keyed on working-set size rather than on
`(L, N_c)`, under-predicted the two L=128 rungs by 8 % and 15 % and was
**discarded** rather than tuned. The current form reproduces every measured cell
at or above the observation.

## 12. Frozen-input fidelity

`frozen_inputs/armB_populations.csv` (288 rows, sha256 `9e37733e…`) reproduces
the predecessor's published ARM-B per-cell block **digit for digit** — mean, SEM
and variance at all three lambdas (`INPUTS_LEDGER.md`). Re-verified from disk
after the file was written, by recomputing rather than by copying.

## 13. Duplicate-compute scan

`tools/dedup_scan.py`, run over all eight sources, finds exactly **300** rows on
any `(L, lambda)` cell of this campaign's grid at `L in {32, 48, 64}`: the 288
reused ARM-B rows, and 12 historical-corpus rows at `L=64, lambda=0.3032,
N_c=128, dtau_mult=12` which are not poolable. **Every other cell in the
campaign is new compute**, and no cell is computed twice.

## 14. Scheduler-call scan

Every `.py` and `.sh` file in the task — 46 files after the amendment — scanned line by line for a
scheduler name in **command position** (start of line, or after `; & | ( ` $(`
or a wrapper). **Zero hits.** The seven `submit.slurm` files are excluded by
design: they are data for the human to submit and are never executed by anything
in the package. `preflight.py` additionally re-asserts this about
`run_preflight.sh` at runtime and refuses to pass otherwise.

## 15. Scope invariants

| invariant | check |
|---|---|
| no predecessor archive modified | `git status --porcelain` shows zero modified tracked files outside this task |
| `research/state/**` untouched | not written; guard rule `G1` would have denied it |
| production path unchanged | `PRODUCTION_PATH_UNCHANGED.md`; bundle sha256 `0a33c403…` identical to the predecessor's |
| `main` untouched | work is on `smccert-integration` |
| nothing submitted | guard rule `G4` fired twice and blocked those Bash calls; no scheduler command was ever executed |
| no result-shaped artifact predates the data | the zero-data dry run's `MOCK_PRODUCTION_RESULTS.json` and four figures were deleted |
| no `L = 80` package exists | confirmed by directory listing; `L80_RUNTIME_GATE.md` records the rejection |

---

## Outstanding, and honestly stated

- **`L = 32` and `L = 48` have never run on Ruche.** Their rates are downward
  extrapolations from a measured `L = 64`, deliberately conservative and
  cross-checked two ways, but not measurements. `RUCHE_RUNBOOK.md` §4 adds a
  single-task `L = 32` check specifically to catch this before 379 core-hours
  are committed.
- **`NC_FACTOR[2048] = 1.20` is an extrapolation** of one measured doubling at a
  different `L`. `COST_MODEL.md` §2 gives the bracket.
- **Ruche partition limits were not re-verified** against the live cluster; this
  session has no cluster access and must not obtain any. The runbook gives the
  commands.
- **`tests/test_exact_benchmark.py` was not run.**
- **The 8 pre-existing `doob_wtmc` / `topological` failures remain.** Not caused
  by this task, not on its execution path, not fixed here.
- **The concurrency regime is assumed from one observation.** Seven arrays at
  %64 each has not been tested; the runbook gives the check and the fallback.
- **The ≤ 3 h requirement is met on the predicted figure (2.32 h) and marginal
  on the ×1.40 pessimistic band (3.25 h).** Both are stated in `COST_MODEL.md`
  §7 rather than only the favourable one.
