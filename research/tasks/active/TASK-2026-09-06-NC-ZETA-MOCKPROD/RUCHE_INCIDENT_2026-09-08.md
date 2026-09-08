# RUCHE INCIDENT — 2026-09-08 — first submission of TASK-2026-09-06-NC-ZETA-MOCKPROD

Operational note. **No scientific content.** Nothing here is evidence for or
against any claim, and no observable was measured.

## What was submitted

The committed arms of `TASK-2026-09-06-NC-ZETA-MOCKPROD`, submitted by hand by
the researcher on Ruche.

| | |
|---|---|
| Job IDs | **1694328 – 1694621** |
| Arrays | the committed production arms and the discretisation control |
| Result JSONs produced | **0** |
| Populations completed | **0** |
| Disposition | **all arrays cancelled** |

## What happened

Every array task failed **before the sampler was reached**, with
`FileNotFoundError` on `shared/manifest.csv`.

`shared/run_pack.py` set `ARM = os.getcwd()` — correct — but resolved the
executor as `RUN_CELL = os.path.join(HERE, "run_cell.py")`, i.e.
`shared/run_cell.py`. `run_cell.py` takes **no manifest argument and no output
argument**: it reads `manifest.csv` from the directory *its own file* sits in,
and writes to `results/` beside it. Which manifest a row runs is therefore
decided by where the **executor file** is, not by the working directory. Run out
of `shared/`, it looked for `shared/manifest.csv`, which does not exist and never
did.

The failure was total and immediate — 100 % of rows, before any simulation.

## What is retained

**No scientific data from this submission is retained**, because none was
produced. There is nothing to quarantine, nothing to exclude from an analysis
and nothing to re-check: zero result JSONs were written, so no partial,
suspect or half-length population exists anywhere in the repository or on the
cluster. The seed block allocated to these arms was never consumed and is
re-used unchanged by the repaired package.

Cluster logs for the cancelled arrays remain under each arm's `logs/` on Ruche
and are operational records only.

## The repair

Packaging only, on 2026-09-08. **`shared/run_cell.py` is unchanged, byte for
byte** (`sha256 571f7ff42baccf2905f04fed2ade24f13fd1316edb47ac9cb8d7491e75a1f0a8`).
No sampler parameter, argument, default, seed, grid point, manifest row or cost
figure was touched.

- Every production, control and conditional arm now carries its **own
  byte-identical copy** of `run_cell.py`, written verbatim by
  `tools/build_arms.py`.
- `shared/run_pack.py` invokes `ARM/run_cell.py`, verifies at job start that
  those bytes equal `shared/run_cell.py`, and **refuses to fall back** to the
  shared copy if the arm-local one is absent.
- `conditional/M_z070_nc2048` sits one directory deeper than the other arms, so
  its parent carries a byte-identical copy of `support/` — `run_cell.py`
  resolves its bundle as `HERE/../support`. That arm would have failed its own
  startup gate for a second, independent reason once the executor moved into it;
  it no longer does.
- Every `submit.slurm` derives `PPSQJ_REPO` from its own depth instead of
  relying on `run_cell.py`'s five-levels-up default, and checks it before the
  array does any work.

## What now proves it

- Preflight **P18** — the arm-local executor exists and is the frozen bytes.
- Preflight **P19** — `run_pack.py` is *executed* from the arm and asked which
  file it would run; the answer must be the arm-local one. Measured, not read
  out of the source: the fault was a resolution, not a spelling.
- Preflight **P20** — the arm-local executor is run for real from an unrelated
  working directory with an index past the end of the manifest. Reaching
  `IndexError` proves it found this arm's `manifest.csv`, its bundle and
  `pps_qj`; the working directory is then checked to be empty, which proves the
  default `results/` landed in the arm.
- Negative controls **N15** (no arm-local executor — the exact broken layout)
  and **N16** (arm-local executor drifted from the frozen bytes) must both be
  rejected by P18.
- `tools/negative_controls.py` additionally **reproduces this failure end to
  end** before showing the repair: the pre-fix wrapper resolving an executor
  outside the arm, that executor dying with `FileNotFoundError` on a
  `manifest.csv` outside the arm having written nothing, the shipped wrapper
  resolving the arm-local executor, and the shipped wrapper refusing to run at
  all when the arm-local executor is missing.

## Standing rule, unchanged

`research/RESOURCE_POLICY.md` §4 — **agents never submit HPC jobs**, at any
stage, gate or approval level. This repair was prepared locally; the researcher
submits by hand. Nothing in this package contains a submission command.
