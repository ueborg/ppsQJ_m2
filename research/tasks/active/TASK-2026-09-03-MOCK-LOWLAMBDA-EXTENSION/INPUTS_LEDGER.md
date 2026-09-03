# Inputs ledger — every file this task depends on, with its hash

`TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION`.

## Tracked inputs carried inside the package

| file | bytes | sha256 |
|---|---:|---|
| `support/instrumented.py` | 11,396 | `0a33c4034cda70ea635cf715ee0b160d9f29e75ceacde0de89628ff2c533032d` |
| `support/BUNDLE_MANIFEST.json` | 2,464 | `33bdfdf02ef4a85018d66d1672d86f6ee363c6a380e9f3c729b4fbf028266356` |
| `frozen_inputs/predecessor_nc1024_populations.csv` | 317,956 | `dd1ebac39ed24340807b094b568909a73d870877c8755e25d5062bd7a7e4babf` |
| `analysis_spec.yaml` | 13,560 | `ee80b386f50d35aa3483a07c373b8455c18cffd8d9a9954466b93036aca9cd18` |

`shared/preflight.py` prints the `analysis_spec.yaml` hash in every arm. If it
does not read `ee80b386…`, the frozen spec has been edited and the run should
stop.

---

## The bundled sampler, and why it is load-bearing here

`support/instrumented.py` is **byte-identical** to
`TASK-2026-09-02-MOCK-PRODUCTION/support/instrumented.py` — copied with `cp(1)`
and verified with `cmp(1)` — which is byte-identical to
`TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/support/instrumented.py`, which is
byte-identical to `TASK-2026-09-01-SMCRUCHE-READY/support/instrumented.py`,
which was copied verbatim from the untracked
`TASK-2026-08-30-SMCSTAT/analysis/instrumented.py` that `TASK-2026-08-30-SMCSTAT`
validated bitwise against the production path.

**[E]** Same sha256 `0a33c403…` at every hop.

In the predecessor that identity licensed *reusing* ARM-B's populations. Here it
does more work than that. This task plots four new lambdas on the same axes as
thirteen old ones, computes increments and second differences across the join
between them, and runs a single crossing analysis over all seventeen. That is
only legitimate if the two halves are **the same measurement**, not two
measurements that happen to agree. Identical sampler bytes are the evidence.

`run_cell.py` re-checks the hash at runtime and refuses to start on a mismatch.
`preflight.py` checks it before submission, **and additionally compares it
against the predecessor's copy when that archive is present**, reporting
`sampler == predecessor's: byte-identical`. Negative control N16 confirms an
unrecorded edit to the bundle is caught.

---

## `frozen_inputs/predecessor_nc1024_populations.csv` — and the proof it is faithful

**1,152 rows**, one per reused population, built by `tools/freeze_predecessor.py`
from the predecessor's completed campaign:

| source | cells | rows |
|---|---:|---:|
| `mockL32/results/*.json` — 13 lambdas × R=24 | 13 | 312 |
| `mockL48/results/*.json` — 13 lambdas × R=24 | 13 | 312 |
| `mockL64/results/*.json` — 10 lambdas × R=24 | 10 | 240 |
| `frozen_inputs/armB_populations.csv` — 3 lambdas × R=96 | 3 | 288 |
| | **39** | **1,152** |

The three `L = 64` centre lambdas `0.2932 / 0.3032 / 0.3132` are absent from
`mockL64` by the predecessor's own design; they live in its frozen ARM-B
snapshot at `R = 96` and are carried through here **at full `R = 96`** so that
this task can apply the predecessor's own matched-`R` block rule (block A = the
first 24 in seed order) rather than inheriting a pre-cut subset it cannot audit.

**[E] The snapshot reproduces the predecessor's published curves EXACTLY.**
Recomputing block-A means and SEMs at all 39 cells from this CSV and differencing
against `TASK-2026-09-02-MOCK-PRODUCTION/MOCK_PRODUCTION_RESULTS.json → curves`:

| L | max abs deviation in mean | max abs deviation in SEM |
|---:|---:|---:|
| 32 | `0.000e+00` | `0.000e+00` |
| 48 | `0.000e+00` | `0.000e+00` |
| 64 | `0.000e+00` | `0.000e+00` |

Bit-for-bit, not "agrees to five figures". The CSV stores the floats via
`repr()`, so they round-trip exactly.

All 1,152 rows carry `status = ok`, `zeta = 0.35`, `N_c = 1024`,
`dtau_mult = 6.0`, `resample_scheme = systematic` and `T == L`. Seed range
`30,300,000 – 31,212,023`.

**[E] This snapshot is load-bearing, not a convenience.** `.gitignore` carries a
bare `results/` rule, so **every one of the predecessor's 864 returned JSONs is
untracked**: `git ls-files` finds only `.gitkeep` under each `*/results/`. A
clean checkout of this repository has **no other route** to the reused
populations, and a package that reached sideways for them would fail on Ruche in
exactly the way the first ARM 1 job did. The clean tracked-only checkout test in
`VALIDATION.md` §8 confirms the analysis loads all 1,152 from this file inside
the archive.

### What was deliberately NOT taken

`tools/freeze_predecessor.py` asserts these rather than assuming them, and prints
the assertions:

| excluded | status in the predecessor archive |
|---|---|
| `mockNC128L32` | **cancelled — zero results returned** |
| `mockNC128L48` | **cancelled — zero results returned** |
| `mockNC128L64` | **cancelled — zero results returned** |
| `mockL64nc2048` | 72 results present and **deliberately not read** |
| `frozen_inputs/historical_corpus_zeta035.csv` | `dtau_mult = 12.0`, not poolable, **not read at all** |

The `N_c = 2048` refusal is the one that needed asserting rather than assuming:
those results exist on disk, and a glob one directory wider would have swept
them into the `L = 64` curve at a different population size without any file
recording that it had happened.

---

## Runtime dependencies outside the package

| what | where | note |
|---|---|---|
| `pps_qj` | the repository root | tracked; `run_cell.py` puts the repo root on `sys.path` itself and does not need `PPSQJ_REPO` set |
| `numpy` | the validated interpreter | 2.4.3 locally; whatever the Ruche prefix carries |
| `PyYAML` | optional | used only to pretty-print three preflight fields; the frozen analysis imports no yaml and the preflight has a dependency-free fallback |
| `matplotlib` | optional, Mac side only | figures only; the analysis prints `figures SKIPPED` and completes without it |

**There is no dependency on any other task directory at runtime.** The
predecessor archive is read exactly once, by `tools/freeze_predecessor.py`, on
the developer's machine, to build the CSV above. Nothing that runs on Ruche and
nothing in `analysis/lowlambda_analysis.py` reaches outside this package except
for the tracked `pps_qj`.

The one place the predecessor archive is touched at check time is the
preflight's optional `sampler == predecessor's` comparison, which degrades to a
`NOTE` when the archive is absent and never fails for its absence — verified by
the negative-control staging tree, which does not contain it.

---

## Verified-input restatement

Everything above was **re-read from disk after it was written**, not inferred
from the fact that a build step ran. The hashes were produced by hashing the
files as they now stand in the working tree. The fidelity table was produced by
recomputing the statistics from the frozen CSV and differencing against the
predecessor's own results JSON, not by copying numbers across.
