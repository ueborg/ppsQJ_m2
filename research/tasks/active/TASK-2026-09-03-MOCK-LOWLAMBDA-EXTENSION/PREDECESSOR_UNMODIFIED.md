# The completed predecessor archive was read and not modified

The brief's first constraint: *do not modify the completed predecessor
archive.* This file is the evidence, not the assurance.

## 1. What the archive is

`research/tasks/active/TASK-2026-09-02-MOCK-PRODUCTION/` — 1,040 files
(excluding `.DS_Store` and `__pycache__`), of which 98 are tracked and 864 are
returned result JSONs that `.gitignore`'s bare `results/` rule leaves untracked.

## 2. Tracked files: git says nothing changed

```
$ git status --porcelain -- research/tasks/active/TASK-2026-09-02-MOCK-PRODUCTION | grep -v '^??'
(no output)
```

No tracked file under the predecessor is modified, staged, renamed or deleted.
Only `??` untracked entries remain, and those are the result JSONs and analysis
outputs the campaign itself produced — they were untracked before this task
began and are untracked now.

## 3. Whole-tree content hash, recorded here

Over all 1,040 files, `find | sort | xargs shasum -a 256 | shasum -a 256`:

```
661eed3feeda9ffbe6f7187ea657dfcef4b2a2e4b4bf9c1ee03922d220eeef19
```

Re-running that command must reproduce this digest. If it does not, something
touched the archive and every reuse claim in this task needs re-checking.

## 4. Structural reasons it could not have been modified

Evidence beats assertion, but the structure is worth stating too:

- **`tools/freeze_predecessor.py` only reads.** Its sole write is
  `frozen_inputs/predecessor_nc1024_populations.csv`, inside *this* task. It
  opens predecessor paths with the default mode and never with `w`, `a` or
  `os.remove`.
- **`analysis/lowlambda_analysis.py` never opens a predecessor path at all.**
  It reads the frozen CSV and this task's own `lowlam*/results/`. Its
  `--task-root` argument is what lets the smoke test point it at a staging copy
  instead of anywhere real.
- **`shared/preflight.py` reads one predecessor file**, `support/instrumented.py`,
  to hash it for the `sampler == predecessor's` comparison, and degrades to a
  `NOTE` when the archive is absent.
- **`tools/dedup_scan.py` reads predecessor manifests** and every other manifest
  under `research/tasks/active/`. Read-only.
- **`tools/smoke_test.py` and `tools/negative_controls.py` write only inside a
  staging directory** given on the command line, outside the repository.

## 5. Also not modified

- **`research/state/**`** — not written. No proposal, no claim, no observable,
  no decision. This is a numerical child task; its outputs live in the execution
  plane (`research/tasks/active/…`) and are never authoritative.
- **`main`** — not touched. The commit is on `smccert-integration`, and nothing
  was pushed.
- **Any other task directory** — `tools/dedup_scan.py` reads 17 manifests across
  the active tree and writes to none of them.

## 6. What this task added, and where

Everything this task created is inside
`research/tasks/active/TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/`. Nothing
outside it was created, edited or deleted.
