# Validation

Every check run on this package, with its result. Commands are reproducible from
the repository root with `.venv/bin/python3`.

Nothing was submitted. No check in this file can submit.

---

## 1. Syntax and import

```
bash -n  on all 11 shell files (3 submit.slurm, 3 run_preflight.sh,
         3 analyse_results.sh, 2 in shared/)                          PASS
py_compile on every .py in the package                                PASS
import instrumented + pps_qj + numpy, in a subprocess, exactly as
         run_cell.py imports them (numpy 2.4.3)                       PASS
```

The import check runs in a subprocess so a broken import cannot poison the
checking process. It is the check that would have caught the
`ModuleNotFoundError` that killed the first ARM 1 Ruche job in a predecessor
task.

**Python-version note.** One f-string in `shared/preflight.py` initially used
implicit string concatenation inside the replacement field — legal only on
Python 3.12+ (PEP 701). It compiled locally on 3.13 and would have been a
`SyntaxError` on an older Ruche interpreter, at import time, in every array
task. Rewritten as `%`-formatting before any other check was run.

---

## 2. The manifests

```
tools/build_arms.py                                                   PASS
```

| arm | rows | lambdas | R | seeds | `--array` | partition |
|---|---:|---|---:|---|---|---|
| `lowlamL32` | 96 | 0.1932 0.2032 0.2132 0.2232 | 24 | 32,000,000–32,003,023 | `0-95%64` | `cpu_med` |
| `lowlamL48` | 96 | 0.1932 0.2032 0.2132 0.2232 | 24 | 32,100,000–32,103,023 | `0-95%64` | `cpu_med` |
| `lowlamL64` | 96 | 0.1932 0.2032 0.2132 0.2232 | 24 | 32,200,000–32,203,023 | `0-95%64` | `cpu_med` |

288 rows, 288 seeds, all distinct.

The builder asserts `GRID[4:] == OLD_GRID` at import — the extended grid must
contain the predecessor's grid **in floating point**, not to four decimal
places. A grid that agreed to four decimals but not bitwise would silently split
every reused cell in two.

---

## 3. The reuse is faithful — **bit for bit**

```
tools/freeze_predecessor.py                                           PASS
```

1,152 rows, 39 cells, `L ∈ {32,48,64}`, 13 lambdas, all `status = ok`, all
`N_c = 1024`, all `dtau_mult = 6.0`, all `zeta = 0.35`, all `T == L`.

**[E]** Block-A means and SEMs recomputed from the frozen CSV, differenced
against `TASK-2026-09-02-MOCK-PRODUCTION/MOCK_PRODUCTION_RESULTS.json → curves`:

| L | max abs deviation, mean | max abs deviation, SEM |
|---:|---:|---:|
| 32 | `0.000e+00` | `0.000e+00` |
| 48 | `0.000e+00` | `0.000e+00` |
| 64 | `0.000e+00` | `0.000e+00` |

Exactly zero over all 39 means and all 39 SEMs. The CSV stores floats via
`repr()`, so they round-trip exactly.

**Exclusions asserted rather than assumed**, printed by the freeze script:

```
mockL64nc2048    excluded -- 72 results present and DELIBERATELY NOT READ
mockNC128L32     excluded -- cancelled, zero results returned
mockNC128L48     excluded -- cancelled, zero results returned
mockNC128L64     excluded -- cancelled, zero results returned
```

---

## 4. Cost model against its own data

```
tools/cost_model.py                                                   PASS
```

Refit from the frozen snapshot versus the literals in the module, compared
**where the model is used** (a slope/intercept trade can leave both coefficients
far off while the prediction is fine, and only the prediction is load-bearing):

| L | refit | n | resid sd | max prediction drift |
|---:|---|---:|---:|---:|
| 32 | `0.815551·n + 68.43` | 312 | 7.1 s | **0.000 %** |
| 48 | `1.588743·n + 286.09` | 312 | 24.3 s | **0.000 %** |
| 64 | `2.723572·n + 850.23` | 528 | 99.9 s | **0.000 %** |

The preflight repeats this refit on every run and fails above 0.5 % drift.

---

## 5. Preflight — positive

```
for arm in lowlamL32 lowlamL48 lowlamL64: run_preflight.sh    all exit 0
```

Every design check, every cost-model check and every runtime check `OK`,
including:

- `no predecessor duplication  the 13 already-measured lambdas are absent`
- `all four new lambdas present  ['0.1932','0.2032','0.2132','0.2232']`
- `R matches the reused half  R = [24]`
- `seeds in the fresh block  … block [32000000, 33000000)`
- `no overlap with predecessors  5404 predecessor seeds scanned, 0 collisions`
- `bundle sha256  0a33c4034cda70ea…  (matches manifest)`
- `sampler == predecessor's  byte-identical`
- `--partition  cpu_med`
- `--time vs pessimistic  margin 6.0× / 3.3× / 3.0×`
- `--mem vs 1.5× peak  1G vs 262 MB / 1G vs 430 MB / 2G vs 665 MB`
- `run_preflight.sh has no scheduler call`

---

## 6. Preflight — negative controls

```
tools/negative_controls.py                                            PASS
```

`N00`, the unmodified staged copy, **passes** — without which every failure
below would be uninterpretable. Then 19 single-fault injections, each required
to exit non-zero **and to name the fault**:

| | fault | caught |
|---|---|---|
| N01 | duplicates an already-measured lambda | yes |
| N02 | drops one of the four new lambdas | yes |
| N03 | lambda off the frozen 17-point grid | yes |
| N04 | reuses a predecessor seed | yes |
| N05 | duplicates a seed within the arm | yes |
| N06 | unequal `R` across lambdas | yes |
| N07 | `dtau_mult = 12.0`, the non-poolable corpus value | yes |
| N08 | `N_c = 128` instead of 1024 | yes |
| N09 | an `L` with no measured cost model | yes |
| N10 | partition `cpu_short` | yes |
| N11 | no `--partition` at all | yes |
| N12 | `--time` below the pessimistic slowest task | yes |
| N13 | `--time` above the partition `MaxTime` | yes |
| N14 | `--mem` below 1.5 × the estimated peak | yes |
| N15 | `--array` range not matching the manifest | yes |
| N16 | an unrecorded edit to the certified sampler | yes |
| N17 | the frozen predecessor data removed | yes |
| N18 | cost-model literal drifted from the frozen data | yes |
| N19 | a scheduler call added to `run_preflight.sh` | yes |

### 6a. A defect this testing found: the `--mem` size parser

N14 was first written as `--mem=2G → --mem=1G` and **was not caught**. That was
correct behaviour — `1 GiB` genuinely clears the `0.97 GiB` requirement at
`L = 64` — but chasing it exposed a real defect in the inherited size parser,
`float(str(v).rstrip("Gg"))`, which is wrong in two ways:

- **`--mem=600M`** is an ordinary Slurm request. `rstrip("Gg")` leaves `"600M"`,
  `float()` raises, and the `except` branch returned `0.0` — so the arm failed
  the memory check *for a parse error* while *reporting* an under-request. It
  failed closed, which is why nothing ever broke, but the reason it printed was
  not the reason it failed.
- **`--mem=2048`** means 2048 **megabytes** to Slurm, not 2048 gigabytes. The old
  parser read it as 2048 GiB and would have waved through an arm asking for a
  third of what it needs. **That one fails open.**

Rewritten to understand `K`/`M`/`G`/`T` and Slurm's megabyte default, and unit
checked:

```
'2G' -> 2.0    '1G' -> 1.0     '600M' -> 0.5859   '2048' -> 2.0
'4T' -> 4096   '1024K' -> 0.001   'bogus' -> 0.0   '3g' -> 3.0
```

N14 now injects `--mem=600M`, which is both a realistic request and genuinely
insufficient, and is caught.

---

## 7. The frozen analysis, end to end

```
tools/smoke_test.py                                                   PASS
```

Three synthetic scenarios, each built by choosing the cross-`L` **differences**
directly so the scenario controls what it claims to. All three drive the
analysis through the complete path — 17-point curves, join tests, crossing
protocol, X1–X7 and all three figures — and the pre-registered classification is
required to come out **differently** in each:

| scenario | steered pair `L32–L48` | expected | got |
|---|---|---|---|
| `interior` | sign change two intervals in | `INTERIOR` | `INTERIOR` |
| `below_grid` | negative throughout the new region | `NONE` or `BELOW_GRID` | `NONE` |
| `edge` | sign change forced into the **first** interval | `STILL_BOUNDARY` | `STILL_BOUNDARY` |

The `edge` case is the one that matters: it is exactly the failure mode this
task exists to detect, and an analysis that called it `INTERIOR` would be
worthless. It is shown failing.

**Limitation, stated rather than hidden.** The two pairs involving `L = 64` are
not free to be given arbitrary behaviour, because 13 of their 17 points are the
real reused data and the predecessor's `I64 − I48` already changes sign between
`0.2332` and `0.2432`. No choice of new points can remove that — turning it into
an interior crossing is precisely the point of the extension. So the scenarios
steer `D(32→48)`, assert on that pair, and for the `L = 64` pairs assert only
that the class comes from the frozen four-item vocabulary.

**Zero-data run.** `analysis/lowlambda_analysis.py` with no new results runs to
completion and degrades explicitly: `39 of 51 cells`, every curve
`INCOMPLETE — 13/17`, `X1`–`X5` and `X7` `NOT EVALUATED`, `X6` `SUPPORTED`, all
three figures written with an explicit "no complete curve yet" annotation. It
never analyses a shorter grid silently.

### 7a. A defect the smoke test found: the outcome classifier

The first implementation routed "no raw crossing, bootstrap mass not at the
lower end" into `STILL_BOUNDARY` via a fall-through `else`. That reports a
boundary artefact where there is no locator to be an artefact of. Corrected so
`STILL_BOUNDARY` is reachable **only** when a raw crossing exists, which is what
`analysis_spec.yaml` said all along — the spec was right and the code was wrong.

---

## 8. Clean tracked-only checkout

The strongest self-containment test available: build a tree from the git index
(`git write-tree`), extract **only** the tracked-and-staged content of this task
plus `pps_qj`, into a directory outside the repository, and run everything
there. **The predecessor archive is absent from that tree.**

```
$ ls <clean>/research/tasks/active/
TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION          <- and nothing else
```

| check in the clean checkout | result |
|---|---|
| `lowlamL32` / `lowlamL48` / `lowlamL64` preflight | **all exit 0** |
| `frozen predecessor data` resolves inside the archive | `OK` |
| `sampler == predecessor's` | degrades to `NOTE`, does not fail |
| `import instrumented + pps_qj + numpy` | `OK` |
| `analysis/lowlambda_analysis.py` | loads **all 1,152** reused populations |
| `tools/smoke_test.py` | **PASS**, all three scenarios |
| `tools/dedup_scan.py` | **AUDIT PASSED** |

This is what the `frozen_inputs/` snapshot is for. `.gitignore`'s bare
`results/` rule leaves all 864 of the predecessor's returned JSONs untracked, so
a clean checkout has no other route to them, and a package that reached sideways
for them would fail on Ruche exactly as a predecessor's first job did.

---

## 9. Duplicate, seed and manifest audit

```
tools/dedup_scan.py                                                   PASS
```

- **D1** — 17 other manifests scanned across `research/tasks/active/`, 92
  distinct pre-existing physical cells, **0 duplicates**. The 12 new cells exist
  nowhere else.
- **D2** — 288 seeds, 288 distinct, range 32,000,000–32,203,023; ledger of 5,404
  prior seeds with maximum 31,612,047; **overlap 0**; structural floor
  32,000,000 > 31,612,047, so disjointness does not depend on the scan.
  `tools/allocated_seeds.json` matches the written manifests.
- **D3** — three 96-row manifests, four lambdas × `R = 24`, `--array=0-95%64`
  matching, `cpu_med` on all three.
- **D4** — one design either side of the join: both halves `zeta = 0.35`,
  `N_c = 1024`, `dtau_mult = 6.0`, systematic, `T == L`. The only cells at
  `R ≠ 24` are the three reused `L = 64` centres at `R = 96`, cut to block A in
  seed order.

---

## 10. Repository hygiene

```
git diff --cached --check                          no whitespace errors
git diff --cached --name-only | grep -v <this task>       (nothing)
git status --porcelain -- <predecessor> | grep -v '^??'   (nothing)
```

Nothing outside this task directory is staged. No tracked file under the
predecessor is modified, staged or deleted. `research/state/**` was not written.
`main` was not touched. Nothing was pushed.

Predecessor whole-tree content digest, recorded in
`PREDECESSOR_UNMODIFIED.md`: `661eed3feeda9ffbe6f7187ea657dfcef4b2a2e4b4bf9c1ee03922d220eeef19`.

Zero-data placeholder figures and `LOWLAMBDA_RESULTS.json` produced while
testing were **deleted before staging**. Committing figures drawn from an
incomplete grid would put artefacts in the repository that look like results.
They are regenerated by the single command in `RUCHE_RUNBOOK.md` §9 once the
data returns.

---

## 11. Not validated here, because it cannot be

- **That the four new cells behave.** That is `X1`, and it needs the data.
- **That the join is continuous.** That is `X3` / `J1`–`J3`, and it needs the
  data. The package can only guarantee that the question is asked correctly and
  that no fit is applied across the join.
- **That extending the grid brackets the crossing.** That is `X4`, the entire
  point, and it is the one thing this package deliberately does not presuppose.
  `FALSIFICATION_PLAN.md` Y4, Y5 and Y8 pre-register the ways it may not.
- **Ruche queue behaviour.** `SCHEDULER_DECISION.md` rests on the preceding
  campaign's observed `QOSMaxJobsPerUserLimit` serialisation of `cpu_short`.
  `RUCHE_RUNBOOK.md` §2 gives the check to re-confirm it on the day.
- **Runtime on the actual nodes.** The cost model is fitted to that cluster's
  own recorded `wall_s`, extrapolated at most 0.83× below its fitted `n_steps`
  floor, but a node is a node. `X7` checks it after the fact.
