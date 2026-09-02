# RUCHE_RUNBOOK — exact commands, for the human to type

TASK-2026-09-02-MOCK-PRODUCTION.

**No agent submitted anything, at any stage.** `research/RESOURCE_POLICY.md` §4
forbids it unconditionally. Every submission command below is for you to type.
The preflight scripts contain no scheduler call, cannot submit, and
`preflight.py` asserts that fact about `run_preflight.sh` before passing.

---

## 0. Set these once per shell

```bash
export PPSQJ_REPO=$HOME/ppsQJ_m2                      # adjust to your Ruche path
export TASKD=$PPSQJ_REPO/research/tasks/active/TASK-2026-09-02-MOCK-PRODUCTION
export PPSQJ_PYTHON=$WORKDIR/envs/pps_qj/bin/python   # the validated interpreter
export PATH="$(dirname "$PPSQJ_PYTHON"):$PATH"
export ARMS="mockL32 mockL48 mockL64 mockL64nc2048 mockNC128L32 mockNC128L48 mockNC128L64"
```

`PPSQJ_REPO` is optional — `run_cell.py` derives the repository root from the
package's own location — but setting it removes one variable if your layout
differs.

---

## 1. Push from the laptop, pull on Ruche

A research session never pushes: `.claude/hooks/guard_research.py` rule **G2**
denies `git push`. So the push is yours.

```bash
# ON YOUR LAPTOP
cd /Users/catlover1337/Documents/ppsQJ_m2
git log --oneline -3
git status                                # expect a clean tree
git push -u origin smccert-integration
```

```bash
# ON RUCHE
ssh <you>@ruche.mesocentre.universite-paris-saclay.fr
cd $HOME/ppsQJ_m2
git fetch origin
git checkout smccert-integration
git pull
git log --oneline -3
git status                                # expect a clean tree

# the bundled instrumentation MUST be present and MUST hash to this
sha256sum $TASKD/support/instrumented.py
#   expect 0a33c4034cda70ea635cf715ee0b160d9f29e75ceacde0de89628ff2c533032d
#   (identical to the predecessor's, which is what licenses reusing ARM B)
```

If you would rather not push, copy the task directory across — but note the
package needs the **repository**, not just the task folder, because
`run_cell.py` imports the tracked `pps_qj`:

```bash
# ON YOUR LAPTOP
rsync -av --exclude 'results/' --exclude '__pycache__/' \
    /Users/catlover1337/Documents/ppsQJ_m2/research/tasks/active/TASK-2026-09-02-MOCK-PRODUCTION/ \
    <you>@ruche:$HOME/ppsQJ_m2/research/tasks/active/TASK-2026-09-02-MOCK-PRODUCTION/
```

---

## 2. Check the environment, the partitions and the concurrency regime

```bash
"$PPSQJ_PYTHON" -c "import sys, numpy; print(sys.executable, numpy.__version__)"
"$PPSQJ_PYTHON" -c "import pps_qj; print(pps_qj.__file__)"
```

There is no conda on Ruche; this is a plain prefix on the work filesystem. If
`pps_qj` does not import, do nothing — `run_cell.py` puts the repository root on
`sys.path` itself. **PyYAML is not required.** **Never install packages from
inside a batch job.**

The package hard-codes Ruche's partition limits as `cpu_short` 1 h, `cpu_med`
4 h, `cpu_long` 7 days. **These were not re-verified against the live cluster —
this session has no cluster access and must not obtain any.** Confirm them:

```bash
sinfo -o "%20P %10l %10L %6D %t" | head -20
scontrol show partition cpu_short | grep -i maxtime
scontrol show partition cpu_med   | grep -i maxtime
```

If any MaxTime has changed, update `RUCHE_PARTITIONS` in `shared/preflight.py`,
re-run `tools/build_arms.py`, and re-run every preflight before submitting.

**The concurrency question, which decides whether this is a 3-hour campaign.**
On 2026-09-02 the predecessor's three arrays ran with 64, 48 and 32 tasks
concurrent **at the same time**, so that allocation granted 64 slots *per array*
rather than 64 in total. This campaign queues seven arrays and has not tested
that. Check:

```bash
sacctmgr show assoc where user=$USER format=Account,Partition,MaxJobs,MaxSubmit,GrpTRES
squeue -u $USER
```

- **Per-array slots** (the observed regime): submit all seven together, elapsed
  ≈ **2.32 h**, set by `mockL64`.
- **Shared slots**: elapsed is throughput-bound at `378.8 / 64 * 1.15 = 6.8 h`.
  In that case submit the four `mock*` main arms first (297.5 core-h → 5.3 h)
  and the three `mockNC128*` companion arms afterwards; they are 81 core-hours
  and nothing depends on them being simultaneous.

---

## 3. Preflight every arm. All seven must exit 0.

```bash
for A in $ARMS; do
  echo "===== $A"
  ( cd $TASKD/$A && PYTHON="$PPSQJ_PYTHON" bash run_preflight.sh ) || echo "FAILED: $A"
done
```

Confirm, per arm:

| arm | rows | array | partition | --time | --mem | core-h | slowest | elapsed |
|---|---:|---|---|---|---|---:|---:|---:|
| `mockL32` | 312 | `0-311%64` | cpu_short | 01:00:00 | 1G | 12.1 | 0.05 h | 0.22 h |
| `mockL48` | 312 | `0-311%64` | cpu_med | 02:00:00 | 1G | 58.8 | 0.23 h | 1.06 h |
| `mockL64` | 240 | `0-239%64` | cpu_med | 03:00:00 | 2G | 129.3 | 0.66 h | **2.32 h** |
| `mockL64nc2048` | 72 | `0-71%64` | cpu_med | 04:00:00 | 3G | 97.3 | 1.39 h | 1.75 h |
| `mockNC128L32` | 624 | `0-623%64` | cpu_short | 01:00:00 | 1G | 4.1 | 0.01 h | 0.07 h |
| `mockNC128L48` | 624 | `0-623%64` | cpu_short | 01:00:00 | 1G | 19.9 | 0.04 h | 0.36 h |
| `mockNC128L64` | 624 | `0-623%64` | cpu_short | 01:00:00 | 1G | 57.3 | 0.11 h | 1.03 h |

and that in every arm the **FROZEN DESIGN** and **RUNTIME SELF-CONTAINMENT**
blocks are all `OK`, including `bundle sha256 … matches manifest`,
`import instrumented+pps_qj+numpy`, `no overlap with predecessors … 0
collisions`, and — in `mockL64` only — `no ARM-B duplication`.

`analysis-spec sha256` must read
`a1613a3716b2b33b7d601a5606026bae0f1a57b0e620dd5c8c2c748d731a1f13`
in all seven. If it does not, the frozen spec has been edited: **stop.**

---

## 4. Run TWO tasks by hand before queueing 2,808

Worth the 45 minutes. It validates the interpreter, the imports, the bundle gate
and — most importantly — the two rates the cost model is least sure of.
`run_cell.py` is idempotent, so the results are kept and the arrays will skip
them.

```bash
# the campaign's long pole, at a MEASURED rate. Expect ~2300 s.
cd $TASKD/mockL64
srun --partition=cpu_med --time=01:00:00 --mem=2G --cpus-per-task=1 \
     "$PPSQJ_PYTHON" run_cell.py 0 "$PWD/results"

# the one L that has never run on Ruche at all. Expect ~140 s.
cd $TASKD/mockL32
srun --partition=cpu_short --time=00:30:00 --mem=1G --cpus-per-task=1 \
     "$PPSQJ_PYTHON" run_cell.py 0 "$PWD/results"
```

Index 0 of each is the **cheapest** lambda (0.2332), so expect a little under
the arm's slowest-task figure.

**If either wall time exceeds 1.4× the prediction, stop and re-read
`COST_MODEL.md` before committing 379 core-hours.** The `L = 32` check is the
one that matters most: it is a downward extrapolation, and it is the only rate
in the package with no measurement behind it at all.

---

## 5. Submit. YOU type these. Queue `mockL64` first — it is the long pole.

```bash
cd $TASKD/mockL64        && sbatch submit.slurm
cd $TASKD/mockL64nc2048  && sbatch submit.slurm
cd $TASKD/mockL48        && sbatch submit.slurm
cd $TASKD/mockL32        && sbatch submit.slurm
cd $TASKD/mockNC128L64   && sbatch submit.slurm
cd $TASKD/mockNC128L48   && sbatch submit.slurm
cd $TASKD/mockNC128L32   && sbatch submit.slurm
```

If §2 said your slots are **shared in total**, submit only the first four now
and throttle them so they do not starve each other:

```bash
scontrol update jobid=<JOBID> arraytaskthrottle=16
```

**Do NOT submit `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/armA2048_optional`
alongside this.** It is a separate recommended submission with a ~31 h single
task — read `L128_NC2048_HANDOFF.md` first, because its cost is now known to be
55 % higher than that package's own preflight reports.

---

## 6. Monitor and check completeness

```bash
squeue -u $USER -o "%.10i %.16j %.10P %.8T %.10M %.10l %.6D %R"
sacct -X -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode | head -30

# how many array tasks failed
sacct -j <JOBID> --format=JobID,State,ExitCode -n | grep -cv COMPLETED

# results on disk vs manifest rows expected
for A in $ARMS; do
  n=$(find $TASKD/$A/results -name '*.json' | wc -l)
  m=$(( $(wc -l < $TASKD/$A/manifest.csv) - 1 ))
  echo "$A: $n / $m"
done

# which indices are missing, if any
cd $TASKD/mockL64
"$PPSQJ_PYTHON" - <<'PY'
import csv, os, glob
rows = list(csv.DictReader(open("manifest.csv")))
have = {int(os.path.basename(p).split("_")[1][:5]) for p in glob.glob("results/*.json")}
miss = sorted(set(range(len(rows))) - have)
print(f"{len(have)}/{len(rows)} present; missing: {miss}")
PY

# peak memory actually used, to check the requests were right
sacct -j <JOBID> --format=JobID,MaxRSS,ReqMem -n | sort -k2 -h | tail -5

# any non-empty stderr
find $TASKD/*/logs -name '*.err' -size +0 | head
```

Resubmitting an arm is safe: `run_cell.py` skips any row whose result JSON
already exists, so a partial array is topped up rather than redone.

---

## 7. Per-arm analysis

Runs on Ruche or after the transfer back; it needs only numpy.

```bash
for A in $ARMS; do
  echo "===== $A"
  ( cd $TASKD/$A && PYTHON="$PPSQJ_PYTHON" bash analyse_results.sh )
done
```

Each arm writes its own `arm_summary.json`. Per-arm analysis deliberately does
**not** compute curves, crossings or the `N_c` comparison; those are cross-arm
and belong to step 9.

---

## 8. Bring the results back to the Mac

```bash
# ON YOUR LAPTOP
export TD=/Users/catlover1337/Documents/ppsQJ_m2/research/tasks/active/TASK-2026-09-02-MOCK-PRODUCTION
for A in mockL32 mockL48 mockL64 mockL64nc2048 mockNC128L32 mockNC128L48 mockNC128L64; do
  rsync -avz --progress \
      <you>@ruche.mesocentre.universite-paris-saclay.fr:$HOME/ppsQJ_m2/research/tasks/active/TASK-2026-09-02-MOCK-PRODUCTION/$A/results/ \
      $TD/$A/results/
done

# and the logs, which are the only record of what the scheduler actually did --
# the elapsed-time and concurrency evidence in COST_MODEL.md was reconstructed
# from exactly these files for the predecessor, so keep them
rsync -avz --include='*/' --include='*.out' --include='*.err' --exclude='*' \
    <you>@ruche.mesocentre.universite-paris-saclay.fr:$HOME/ppsQJ_m2/research/tasks/active/TASK-2026-09-02-MOCK-PRODUCTION/ \
    $TD/
```

---

## 9. The combined analysis — the one place M1–M7 are evaluated

```bash
# ON YOUR LAPTOP
cd /Users/catlover1337/Documents/ppsQJ_m2
.venv/bin/python3 research/tasks/active/TASK-2026-09-02-MOCK-PRODUCTION/analysis/mock_production_analysis.py \
    | tee research/tasks/active/TASK-2026-09-02-MOCK-PRODUCTION/MOCK_PRODUCTION_ANALYSIS.txt
```

It prints, in order: the `CMI(lambda)` curves with per-point increments and `r`;
the per-cell population diagnostics; the crossing analysis per `L`-pair and per
`N_c` class; the high-`N_c` vs low-`N_c` comparison; the `Delta_N(lambda)` shape
check; the M1–M7 verdicts; and the list of things the analysis may not say. It
writes `MOCK_PRODUCTION_RESULTS.json` and the four figures under
`analysis/figures/`.

It needs `matplotlib` for the figures only; without it the analysis prints
`figures SKIPPED` and completes.

Then write the outcomes into `FALSIFICATION_RESULTS.md` — a **new** file. Do not
edit `FALSIFICATION_PLAN.md`, `SUCCESS_CRITERIA.md` or `analysis_spec.yaml`; the
plan and its results are deliberately separate files, and editing a frozen
artifact after its stage closed is error `M5`.

---

## Reminders

- **No agent submits anything.** Not now, not after a gate, not after a
  successful single-task test.
- **Do not edit a manifest by hand.** Regenerate with `tools/build_arms.py` and
  re-run every preflight. The preflight catches hand-edits — all fourteen
  injected faults in `VALIDATION.md` §6 were caught — but regeneration is the
  supported path.
- **Do not change the `--array` range.** One task per manifest row; the
  preflight exits non-zero if that stops being true.
- **Do not add a lambda point after seeing a result.** If the grid turns out not
  to bracket the crossing, that is an INCONCLUSIVE for M3 and a child task, not
  an extension of this one.
- **Git from the Mac only, never from Ruche.**
