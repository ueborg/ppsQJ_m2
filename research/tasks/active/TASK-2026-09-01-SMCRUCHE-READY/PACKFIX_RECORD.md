# PACKFIX_RECORD — the readiness verdict was wrong, and why

TASK-2026-09-01-SMCRUCHE-PACKFIX. Labels: `[E]` · `[I]` · `[C]` · `[J]`

---

## 1. What happened

`[E]` The first ARM 1 Ruche job failed immediately:

```
ModuleNotFoundError: No module named 'instrumented'
```

`[E]` `arm1/run_cell.py` did:

```python
sys.path.insert(0, os.path.join(
    REPO, "research/tasks/active/TASK-2026-08-30-SMCSTAT/analysis"))
import instrumented as I
```

`[E]` `git ls-files` on that directory returns **0 tracked files**. It exists in
the developer working tree and in **no** clone.

`[J]` **My `READY_FOR_ARM1` verdict was wrong.** I validated the package in the
working tree, where the untracked parent task happened to be present, and
declared a package self-contained on the strength of a test that could not
distinguish self-contained from not.

## 2. A second, related error in the same report

`[E]` I also reported: *"`run_cell.py`'s default `PPSQJ_REPO` resolved to
`<repo>/research` (wrong) — now it fails loudly."* **That patch never applied.**
The committed `arm1/run_cell.py` was byte-identical to the SMCCERT original; a
heredoc replacement silently matched nothing and I printed "patched" without
checking.

`[J]` Both errors have one root cause: **I reported an outcome I had not
verified.** Every patch in this task now asserts its own anchor and re-reads the
file afterwards, and refuses rather than claiming success.

## 3. The fix

### 3a. Bundle the exact frozen instrumentation

`[E]` Transitive closure computed by `ast.parse`, walking `Import`/`ImportFrom`
and following anything resolving to a sibling module in that untracked directory.
**Result: no sibling imports at all.** `instrumented.py` needs only `numpy`,
`dataclasses`, `time` and the **tracked** `pps_qj` package.

`[E]` So the closure is **one file**, bundled byte-for-byte:

| | |
|---|---|
| original | `research/tasks/active/TASK-2026-08-30-SMCSTAT/analysis/instrumented.py` |
| bundled | `research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/support/instrumented.py` |
| sha256 (source) | `0a33c4034cda70ea635cf715ee0b160d9f29e75ceacde0de89628ff2c533032d` |
| sha256 (bundled) | `0a33c4034cda70ea635cf715ee0b160d9f29e75ceacde0de89628ff2c533032d` |
| size | 11,396 bytes, byte-identical |

`[E]` **The numerical implementation was not rewritten, reformatted or
re-implemented.** This is the exact file that produced every frozen local SMCCERT
result and that SMCSTAT validated bitwise against the production path.

`[E]` `run_cell.py` re-hashes the bundle on **every run** against
`support/BUNDLE_MANIFEST.json` and exits on a mismatch, so the instrumentation
cannot be swapped without changing a recorded hash.

`[E]` `PPSQJ_REPO` is no longer required; the repository root is derived from the
package's own location, with the variable retained as an override.

### 3b. Partition

`[E]` The first attempt declared **no** `--partition`, so Ruche defaulted to
`cpu_short` (MaxTime 1 h) against a 4 h request.

| arm | frozen `--time` | partition | MaxTime | why |
|---|---|---|---|---|
| ARM 1 | `04:00:00` | **`cpu_med`** | 4 h | smallest that accommodates it |
| ARM 2 | `08:00:00` | **`cpu_long`** | 7 d | `cpu_med` caps at 4 h |

`[J]` **No wall request was changed to suit a partition.** The frozen `--time`
values are untouched and the partition was chosen to fit them.

### 3c. Python environment

`[E]` `submit.slurm` resolves the interpreter explicitly from `PPSQJ_PYTHON`,
defaulting to `/gpfs/workdir/ercetinut/envs/pps_qj/bin/python`, refuses to start
if it is not executable, prepends its directory to `PATH`, and prints the
resolved executable and numpy version before doing any work. **No conda is
assumed** — there is none on Ruche.

### 3d. PyYAML

`[E]` **Not required.** `analyse_ruche.py` — the frozen analysis, and the only
thing that computes a verdict — contains **zero** yaml imports and was verified
to run to completion with `yaml` hard-blocked at the import hook.

`[E]` `preflight.py` used it only to pretty-print the question and decision rule.
That now falls back to a dependency-free block scanner, so those still print
without PyYAML, and the preflight reports which path it took. **Nothing is
installed from inside a job.**

## 4. The test that would have caught it

`[E]` A clean tracked-only environment built with `git archive HEAD | tar -x`,
plus the new package files. It contains **754 tracked files** and, under
`research/tasks/active/`, **only** `TASK-2026-09-01-SMCRUCHE-READY` — the SMCSTAT
directory is absent, exactly as on Ruche.

`[E]` **Control, in that environment:** the OLD import form fails with
`ModuleNotFoundError: No module named 'instrumented'` — the reported error,
reproduced.
`[E]` **New form:** both arms import, resolving `instrumented` from the bundled
`support/` and `pps_qj` from the clean tree.

`[E]` **The preflight now fails (exit 1)** on a missing bundle, a SHA mismatch,
unresolvable imports, a missing `--partition`, or a wall request the partition
cannot hold. Verified by negative control on each:

| forced condition | preflight |
|---|---|
| `support/` hidden | **exit 1**, `ModuleNotFoundError` named |
| `--partition=cpu_short` on a 4 h request | **exit 1**, "exceeds partition cpu_short MaxTime of 1 h" |
| `--partition` line deleted | **exit 1**, "the scheduler will pick a default (cpu_short…) and kill the job" |
| everything restored | **exit 0**, `PREFLIGHT PASSED` |

## 4b. A third instance of the same failure mode, caught before commit

`[J]` While preparing this very fix I made the same class of error again, and it
is worth recording rather than quietly correcting.

`[E]` `build_arms.py` copied **both** `run_cell.py` and `analyse_ruche.py` from
the frozen SMCCERT package on every run. Re-running it as part of "final
verification" **silently reverted the bundled-import fix** in both arms. The
patch had applied, been asserted, compiled, and passed the clean-clone smoke
test — and was then overwritten by a later verification step.

`[E]` It was caught by reading `git diff --cached --name-status` and noticing
that `arm1/run_cell.py` and `arm2/run_cell.py` were **absent from the staged
list** when they should have been modified. Had I trusted the earlier
"PATCHED AND VERIFIED" output, the unfixed file would have been committed and the
second Ruche attempt would have failed identically to the first.

`[E]` **Repaired at the tooling level**, not just in the file: `build_arms.py` no
longer touches `run_cell.py` at all, and instead **refuses to run** if the
bundled-import block is missing:

```
run_cell.py has lost the PACKFIX bundled-import block (missing 'SUPPORT = os.path.abspath').
  It would import `instrumented` from an UNTRACKED directory and fail on any
  clean clone. Restore it before continuing.
```

`[E]` Verified: after re-applying the fix, `build_arms.py` runs clean and the
block survives. `[E]` And the committed `run_cell.py` is **byte-identical** to
the copy that produced the clean-clone smoke result, so that evidence applies to
what is actually committed.

`[J]` Three instances now, all the same shape: **state assumed rather than
re-read**. The durable defences added here are the preflight's exit-1 checks, the
`build_arms.py` refusal, and the run-time SHA256 gate in `run_cell.py` — all of
which fail loudly rather than producing a plausible-looking pass.

## 5. What did NOT change

`[E]` No scientific parameter: not `L`, `T`, ζ, λ, `dtau_mult`, the `N_c` ladder,
`R`, any seed, any manifest row, any fit rule, any decision threshold, or the
observable. `[E]` `build_arms.py` re-verifies the manifests against the frozen
SMCCERT source row by row on every run. `[E]` `research/state/**`: unchanged.
`[E]` No frozen artifact of GENCOL, SMCSTAT or SMCCERT was modified.
