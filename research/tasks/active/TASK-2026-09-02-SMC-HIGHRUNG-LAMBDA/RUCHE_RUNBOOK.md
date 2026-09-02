# RUCHE_RUNBOOK — exact commands, for the human to type

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA.

**No agent submitted anything, at any stage.** `research/RESOURCE_POLICY.md` §4
forbids it unconditionally. Every submission command below is for you to type.
The preflight scripts contain no scheduler call, cannot submit, and
`preflight.py` asserts that fact about `run_preflight.sh` before passing.

---

## 0. Set these once per shell

```bash
export PPSQJ_REPO=$HOME/ppsQJ_m2                      # adjust to your Ruche path
export TASKD=$PPSQJ_REPO/research/tasks/active/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA
export PPSQJ_PYTHON=$WORKDIR/envs/pps_qj/bin/python   # the validated interpreter
export PATH="$(dirname "$PPSQJ_PYTHON"):$PATH"
```

---

## 1. Push from the laptop, pull on Ruche

A research session never pushes — `.claude/hooks/guard_research.py` rule **G2**
denies `git push`. So the push is yours:

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
```

If you would rather not push, `rsync` the task directory across — but note the
package needs the **repository**, not just the task folder, because `run_cell.py`
imports the tracked `pps_qj`:

```bash
# ON YOUR LAPTOP
rsync -av --exclude 'results/' --exclude '__pycache__/' \
    /Users/catlover1337/Documents/ppsQJ_m2/research/tasks/active/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/ \
    <you>@ruche:$HOME/ppsQJ_m2/research/tasks/active/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/
```

---

## 2. Check the environment and the partitions

```bash
"$PPSQJ_PYTHON" -c "import sys, numpy; print(sys.executable, numpy.__version__)"
"$PPSQJ_PYTHON" -c "import pps_qj; print(pps_qj.__file__)"
```

There is no conda on Ruche; this is a plain prefix on the work filesystem. If
`pps_qj` does not import, do nothing — `run_cell.py` puts the repository root on
`sys.path` itself. **PyYAML is not required**: the frozen analysis imports none
and the preflight falls back to a dependency-free reader. **Never install
packages from inside a batch job.**

The package hard-codes Ruche's partition limits as `cpu_short` 1 h, `cpu_med`
4 h, `cpu_long` 7 days. Confirm they are still current before you submit:

```bash
sinfo -o "%20P %10l %10L %6D %t" | head -20
scontrol show partition cpu_med  | grep -i maxtime
scontrol show partition cpu_long | grep -i maxtime
```

If any MaxTime has changed, update `RUCHE_PARTITIONS` in
`shared/preflight.py`, re-run `tools/build_arms.py`, and re-run every preflight
before submitting.

Check whether your allocation gives 64 slots **per array** or 64 **in total** —
it changes the elapsed-time estimate from ~10 h to ~19 h (`COST_MODEL.md`):

```bash
sacctmgr show assoc where user=$USER format=Account,Partition,MaxJobs,MaxSubmit,GrpTRES
squeue -u $USER
```

---

## 3. Preflight every arm. All four must exit 0.

```bash
for A in armA512 armA1024 armB armC; do
  echo "===== $A"
  ( cd $TASKD/$A && PYTHON="$PPSQJ_PYTHON" bash run_preflight.sh ) || echo "FAILED: $A"
done
```

Confirm, per arm:

| arm | rows | array | partition | --time | --mem | core-h | slowest |
|---|---:|---|---|---|---|---:|---:|
| `armA512`  | 48  | `0-47%64`  | cpu_long | 12:00:00 | 3G | 241.4 | 5.03 h |
| `armA1024` | 32  | `0-31%64`  | cpu_long | 24:00:00 | 5G | 321.9 | 10.06 h |
| `armB`     | 288 | `0-287%64` | cpu_med  | 03:00:00 | 2G | 167.1 | 0.60 h |
| `armC`     | 96  | `0-95%64`  | cpu_long | 12:00:00 | 3G | 482.9 | 5.20 h |

and that in every arm the **FROZEN DESIGN** and **RUNTIME SELF-CONTAINMENT**
blocks are all `OK`, including `bundle sha256 … matches manifest`,
`import instrumented+pps_qj+numpy`, and
`no overlap with predecessors … 0 collisions`.

`analysis-spec sha256` must read
`e79e34365475824200667aeb3a8f250563705c453a309a0daa3ff397622d1f14` in all four.
If it does not, the frozen spec has been edited and you should stop.

---

## 4. Run ONE task by hand before queueing 464

Worth the 36 minutes. It validates the interpreter, the imports, the bundle gate
and the wall-time estimate on a real compute node, and `run_cell.py` is
idempotent so the result is kept and the array will skip it.

```bash
cd $TASKD/armB
srun --partition=cpu_med --time=01:00:00 --mem=2G --cpus-per-task=1 \
     "$PPSQJ_PYTHON" run_cell.py 0 "$PWD/results"
```

Expect `[ok] idx=0 L=64 N_c=1024 wall=~2100s`. If the wall time is far from
2100 s, the cost model is off for this machine and you should re-read
`COST_MODEL.md` before committing 1213 core-hours.

---

## 5. Submit. YOU type these. Queue `armA1024` first — it is the long pole.

```bash
cd $TASKD/armA1024 && sbatch submit.slurm
cd $TASKD/armA512  && sbatch submit.slurm
cd $TASKD/armB     && sbatch submit.slurm
cd $TASKD/armC     && sbatch submit.slurm
```

**Recommended for simultaneous submission tonight: all four above.**

If your allocation gives 64 slots **in total** rather than per array, submit
only the first three tonight (730 core-hours, ~11.4 h) and hold `armC` for
tomorrow. In that case also lower each array's `%64` cap so they share the
allocation instead of starving each other:

```bash
scontrol update jobid=<JOBID> arraytaskthrottle=24
```

**Do not submit `armA2048_optional` tonight.** Read `NC2048_AUDIT.md`; the
decision rule for tomorrow is "queue it if and only if F2 comes back SUPPORTED".

---

## 6. Monitor and check completeness

```bash
squeue -u $USER -o "%.10i %.14j %.10P %.8T %.10M %.10l %.6D %R"
sacct -X -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode | head -30

# how many array tasks failed, per arm
sacct -j <JOBID> --format=JobID,State,ExitCode -n | grep -cv COMPLETED

# results actually on disk vs manifest rows expected
for A in armA512 armA1024 armB armC; do
  n=$(find $TASKD/$A/results -name '*.json' | wc -l)
  m=$(( $(wc -l < $TASKD/$A/manifest.csv) - 1 ))
  echo "$A: $n / $m"
done

# which indices are missing, if any
cd $TASKD/armC
"$PPSQJ_PYTHON" - <<'PY'
import csv, os, glob
rows = list(csv.DictReader(open("manifest.csv")))
have = {int(os.path.basename(p).split("_")[1][:5]) for p in glob.glob("results/*.json")}
miss = sorted(set(range(len(rows))) - have)
print(f"{len(have)}/{len(rows)} present; missing: {miss}")
PY

# peak memory actually used, to check the request was right
sacct -j <JOBID> --format=JobID,MaxRSS,ReqMem -n | sort -k2 -h | tail -5

# any non-empty stderr
find $TASKD/*/logs -name '*.err' -size +0 | head
```

Resubmitting an arm is safe: `run_cell.py` skips any row whose result JSON
already exists, so a partial array can be topped up by resubmitting the same
script.

---

## 7. Per-arm analysis

Run it on Ruche or after the rsync back; it needs only numpy.

```bash
for A in armA512 armA1024 armB armC; do
  echo "===== $A"
  ( cd $TASKD/$A && PYTHON="$PPSQJ_PYTHON" bash analyse_results.sh )
done
```

---

## 8. Bring the results back to the Mac

```bash
# ON YOUR LAPTOP
export TD=/Users/catlover1337/Documents/ppsQJ_m2/research/tasks/active/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA
for A in armA512 armA1024 armB armC; do
  rsync -avz --progress \
      <you>@ruche.mesocentre.universite-paris-saclay.fr:$HOME/ppsQJ_m2/research/tasks/active/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/$A/results/ \
      $TD/$A/results/
done

# and the logs, which are the only record of what the scheduler actually did
rsync -avz --include='*/' --include='*.out' --include='*.err' --exclude='*' \
    <you>@ruche.mesocentre.universite-paris-saclay.fr:$HOME/ppsQJ_m2/research/tasks/active/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/ \
    $TD/
```

---

## 9. Combined analysis — the one that evaluates F1–F7

```bash
# ON YOUR LAPTOP
cd /Users/catlover1337/Documents/ppsQJ_m2
.venv/bin/python3 research/tasks/active/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/analysis/combined_analysis.py \
    | tee research/tasks/active/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/COMBINED_ANALYSIS.txt
```

It prints, in order: the `N_c` convergence table at L = 128 with direct
rung-to-rung differences and CIs; the `gamma` window scan; the lambda stencils at
both L with `d_-`, `d_+`, `q` and the S1–S4 diagnostics; the per-L-class
production recommendation; and then the frozen F1–F7 verdicts. It also writes
`COMBINED_RESULTS.json`.

Then write the outcomes into `FALSIFICATION_RESULTS.md` — a **new** file. Do not
edit `FALSIFICATION_PLAN.md`; the plan and its results are deliberately separate
files, and editing a frozen artifact after its stage closed is error `M5`.

---

## Reminders

- **No agent submits anything.** Not now, not after a gate, not after a
  successful single-task test.
- **Do not edit a manifest by hand.** Regenerate with `tools/build_arms.py` and
  re-run every preflight; the preflight will catch a hand-edit but a
  regeneration is the supported path.
- **Do not change the `--array` range.** One task per manifest row; the preflight
  exits nonzero if that stops being true.
- **Git from the Mac only, never from Ruche.**
