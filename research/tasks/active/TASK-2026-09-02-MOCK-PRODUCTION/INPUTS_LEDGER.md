# Inputs ledger — every file this task depends on, with its hash

TASK-2026-09-02-MOCK-PRODUCTION.

## Tracked inputs carried inside the package

| file | bytes | sha256 |
|---|---:|---|
| `support/instrumented.py` | 11,396 | `0a33c4034cda70ea635cf715ee0b160d9f29e75ceacde0de89628ff2c533032d` |
| `support/BUNDLE_MANIFEST.json` | 2,304 | `153c68d81b755e3f9af413587c63529e72cca684a3d660f3c4439ddb5fe97768` |
| `frozen_inputs/armB_populations.csv` | 76,669 | `9e37733e26e61c42f32f3cc548a806a55beb40bb1aa8b8ee3286c3784ec8328e` |
| `frozen_inputs/historical_corpus_zeta035.csv` | 469,104 | `f7598a121c99506021634fb4a6024ab28fb06550e82cb5332b22b666719e811a` |
| `analysis_spec.yaml` | 32,087 | `a1613a3716b2b33b7d601a5606026bae0f1a57b0e620dd5c8c2c748d731a1f13` |

`shared/preflight.py` prints the `analysis_spec.yaml` hash in every arm. If it
does not read `a1613a37…`, the frozen spec has been edited and the run should
stop.

## The bundled sampler

`support/instrumented.py` is **byte-identical** to
`TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/support/instrumented.py`, which is itself
byte-identical to `TASK-2026-09-01-SMCRUCHE-READY/support/instrumented.py`, which
was copied verbatim from the untracked
`TASK-2026-08-30-SMCSTAT/analysis/instrumented.py` that
`TASK-2026-08-30-SMCSTAT` validated bitwise against the production path.

**[E]** Same sha256 `0a33c403…` at every hop. That identity is what licenses
reusing ARM-B's populations as if they were this campaign's own: they were
produced by this exact file. See `PRODUCTION_PATH_UNCHANGED.md`.

`run_cell.py` re-checks the hash at runtime and refuses to start on a mismatch;
`preflight.py` checks it before submission.

## `frozen_inputs/armB_populations.csv` — and the proof it is faithful

288 rows, one per completed ARM-B population, built by re-parsing
`TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/armB/results/*.json` and keeping the
per-population summary fields (the per-clone arrays are not carried; nothing in
this task's analysis uses them).

**[E]** Recomputing the per-cell statistics from this snapshot reproduces the
predecessor's published ARM-B block **digit for digit**:

| lambda | R | mean CMI | published | SEM | published | variance | published |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2932 | 96 | 0.40307 | 0.40307 | 0.00373 | 0.00373 | 1.3384e-03 | 1.3384e-03 |
| 0.3032 | 96 | 0.35316 | 0.35316 | 0.00291 | 0.00291 | 8.1232e-04 | 8.1232e-04 |
| 0.3132 | 96 | 0.31612 | 0.31612 | 0.00265 | 0.00265 | 6.7500e-04 | 6.7500e-04 |

(published figures from
`TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/COMBINED_ANALYSIS.txt` §C.)

All 288 rows carry `status = ok`, zero non-finite clones and zero brentq
fallbacks.

**[E] This snapshot is load-bearing, not a convenience.** `git ls-files` shows
that `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/armB/results` has **exactly one tracked
file, `.gitkeep`** — all 288 source JSONs are untracked and exist only in the
developer working tree. A clean checkout therefore has **no other route** to the
reused populations, and a package that reached sideways for them would fail on
Ruche in exactly the way the first ARM 1 job did. The clean tracked-only checkout
test in `VALIDATION.md` §9 confirms the analysis loads all 288 from this file
inside the archive.

Under the matched-R amendment these 288 populations supply the four disjoint
`R = 24` blocks A/B/C/D at each of the three reused lambdas; block A is the
primary dataset and the seed ranges are recorded in `MATCHED_R_AMENDMENT.md`.

**The predecessor's archive was read and not modified.**

## `frozen_inputs/historical_corpus_zeta035.csv` — DESCRIPTIVE ONLY

1,200 rows: the `zeta = 0.35` slice of
`/Users/catlover1337/Downloads/pps_all_realizations.csv` (20,355 rows, sha256
`7066bac78198f5a93fa5688a5490540ca4897ca4826bb36be87b809dcaf04c27`), resolved
via `research/state/DATA_ROOTS.yaml` + the machine-local
`research/data_roots.local.yaml`. It is frozen into the package so the analysis
runs from a tracked checkout without reaching outside the repository.

**[E]** `L in {64, 80, 96, 112, 128}` — **no L = 32, no L = 48** — all at
`N_c = 128`, `R = 12` per cell, `dtau_mult = 12.0` throughout.

**It carries no quantitative weight anywhere in this task.** It is used to
locate where the cross-`L` ordering reverses (`LAMBDA_GRID_DECISION.md`, sign
information only) and drawn in Figure B panel 2, dashed, labelled
`DESCRIPTIVE ONLY` inside the axes. `REUSE_AND_DEDUP_AUDIT.md` §3 gives the four
independent reasons it is not poolable.

## Runtime dependencies outside the package

| what | where | note |
|---|---|---|
| `pps_qj` | the repository root | tracked; `run_cell.py` puts the repo root on `sys.path` itself and does not need `PPSQJ_REPO` set |
| `numpy` | the validated interpreter | 2.4.3 locally; whatever the Ruche prefix carries |
| `PyYAML` | optional | used only to pretty-print three preflight fields; the frozen analysis imports no yaml and the preflight has a dependency-free fallback |
| `matplotlib` | optional, Mac side only | figures only; the analysis prints `figures SKIPPED` and completes without it |

**There is no dependency on any other task directory at runtime.** That was the
defect that killed the first ARM 1 Ruche job (`ModuleNotFoundError` on an
untracked sibling path) and it is checked explicitly by the clean tracked-only
checkout test in `VALIDATION.md` §9.

## Verified-input restatement

Everything above was **re-read from disk after it was written**, not inferred
from the fact that a build step ran. The hashes in this file were produced by
hashing the files as they now stand in the working tree, and the ARM-B fidelity
table was produced by recomputing the statistics from the frozen CSV, not by
copying them from the predecessor.
