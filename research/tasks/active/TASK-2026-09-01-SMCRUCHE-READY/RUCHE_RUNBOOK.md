# RUCHE_RUNBOOK — exact commands, for the human to type

TASK-2026-09-01-SMCRUCHE-READY §9.

**No agent submitted anything, at any stage.** `research/RESOURCE_POLICY.md` §4
forbids it unconditionally. Every submission command below is for you to type.
The preflight scripts contain no scheduler call and cannot submit.

Set these once per shell, everywhere below:

```bash
export PPSQJ_REPO=$HOME/ppsQJ_m2                     # adjust to your Ruche path
export ARMS=$PPSQJ_REPO/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY
export PPSQJ_PYTHON=$WORKDIR/envs/pps_qj/bin/python  # the validated interpreter
```

**Updated by TASK-2026-09-01-SMCRUCHE-PACKFIX** after the first ARM 1 attempt
failed. Three things changed and all are deployment-only:

1. `instrumented.py` is now **bundled and tracked** under `support/`, so the
   package works from a clean clone. The first attempt died with
   `ModuleNotFoundError: No module named 'instrumented'` because it was imported
   from an **untracked** research-task directory.
2. `submit.slurm` now declares `--partition` explicitly. The first attempt had
   none, so the scheduler chose `cpu_short` (MaxTime 1 h) against a 4 h request.
3. `submit.slurm` resolves the Python interpreter explicitly from
   `PPSQJ_PYTHON` rather than trusting `python3` on the batch `PATH`.

`PPSQJ_REPO` is **no longer required** — the package derives the repository root
from its own location — but setting it is harmless and makes step 7's pooling
work.

---

## 1. Sync the integration branch onto Ruche

```bash
ssh <you>@ruche.mesocentre.universite-paris-saclay.fr
cd $HOME/ppsQJ_m2
git fetch origin
git checkout smccert-integration
git pull                      # IMPORTANT: pull the PACKFIX commit
git log -3 --oneline
git status                    # expect a clean tree

# the bundled instrumentation MUST be present; this is what the first attempt lacked
ls -l research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/support/instrumented.py
sha256sum research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/support/instrumented.py
#   expect 0a33c4034cda70ea635cf715ee0b160d9f29e75ceacde0de89628ff2c533032d
```

If the branch is not on the remote yet, push it **from your laptop** — a research
session does not push:

```bash
# ON YOUR LAPTOP
cd /Users/catlover1337/Documents/ppsQJ_m2
git push -u origin smccert-integration
```

Or copy the two arm packages directly, which is enough to run but leaves you
without the repo `run_cell.py` imports:

```bash
rsync -av --exclude results/ --exclude __pycache__/ \
    /Users/catlover1337/Documents/ppsQJ_m2/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/ \
    <you>@ruche:$HOME/ppsQJ_m2/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/
```

`run_cell.py` needs the **tracked** `support/instrumented.py` (bundled inside the
package) and the **tracked** `pps_qj` package, so **the repository must be
present**, not only an arm folder. It no longer needs any untracked
research-task directory — that was the first attempt's failure.

## 2. Activate the environment

**There is no conda on Ruche.** The environment is a plain prefix on the work
filesystem, activated through `PATH`:

```bash
export PATH="$WORKDIR/envs/pps_qj/bin:$PATH"
export PPSQJ_PYTHON=$WORKDIR/envs/pps_qj/bin/python

"$PPSQJ_PYTHON" -c "import sys, numpy; print(sys.executable, numpy.__version__)"
"$PPSQJ_PYTHON" -c "import pps_qj; print(pps_qj.__file__)"
```

If `pps_qj` does not import, either install it editable once —

```bash
cd $PPSQJ_REPO && "$PPSQJ_PYTHON" -m pip install -e .
```

— or do nothing: `run_cell.py` puts the repository root on `sys.path` itself, so
a plain clone works without installation.

**PyYAML is not required.** The frozen analysis imports none, and the preflight
falls back to a dependency-free reader. The login node reported
`No module named 'yaml'`; that is fine and is now reported as such rather than as
a degraded message. **Never install packages from inside a batch job.**

The batch script does not depend on this interactive `PATH`: it resolves
`PPSQJ_PYTHON` itself and prints what it resolved.

## 3. ARM 1 preflight

```bash
cd $ARMS/arm1
bash run_preflight.sh
```

Confirm before going further:

- `manifest rows` = **112**, `submit.slurm --array` = `0-111%64  OK`
- `L` 96, `T` 96, `zeta` 0.35, `lambda` 0.3032, `dtau_mult` 6, systematic
- `N_c ladder` 128, 256, 512 with `R` 32, 32, 48
- `expected core-hours` ≈ **62.2**
- `analysis-spec sha256` `ef3e20b18bcc508e…`
- `--partition` line reads **`cpu_med  MaxTime 4 h  vs requested 4 h`** with `OK`
- the whole **RUNTIME SELF-CONTAINMENT** block is `OK`, including
  `bundle sha256 … matches manifest` and `import instrumented+pps_qj+numpy`
- the last line says **`PREFLIGHT PASSED`**

`run_preflight.sh` now **exits 1** if any of that fails. If it does, stop — the
job would die on the node exactly as the first attempt did.

Then run exactly one task interactively before queueing 112:

```bash
time "$PPSQJ_PYTHON" run_cell.py 0 ./results   # expect ~30-50 min, one JSON written
ls -la results/
```

It prints the interpreter, the resolved `instrumented.py` and the resolved
`pps_qj` before doing any work, so you can see it picked up the **bundled**
copy under `support/` and not something else.

If that single task takes wildly more than ~0.9 core-hours, **stop and re-cost**.
The SMCSTAT campaign discovered a 2.45× cost error only after committing.

## 4. ARM 1 submission

```bash
cd $ARMS/arm1
mkdir -p logs results
sbatch submit.slurm
```

Note the job id it prints. Task 0 is already done from step 3 and will be
skipped — `run_cell.py` is idempotent and never recomputes a completed row.

To submit a smaller slice first, override the array on the command line rather
than editing the file:

```bash
sbatch --array=0-11%12 submit.slurm          # a 12-task pilot
```

## 5. Monitor ARM 1

```bash
squeue -u $USER
squeue -u $USER -j <JOBID> -t PENDING,RUNNING
squeue -u $USER -h -t RUNNING -j <JOBID> | wc -l      # running now
watch -n 60 "squeue -u $USER | tail -20"
ls $ARMS/arm1/results | wc -l                          # completed rows, of 112
```

## 6. Inspect ARM 1 failures

```bash
sacct -j <JOBID> --format=JobID%20,State,ExitCode,Elapsed,MaxRSS,ReqMem -X
sacct -j <JOBID> --state=FAILED,TIMEOUT,OUT_OF_MEMORY --format=JobID%20,State,ExitCode -X
grep -l . $ARMS/arm1/logs/*.err | head
tail -40 $ARMS/arm1/logs/arm1_<JOBID>_<TASKID>.err
```

`MaxRSS` above ~1.5G on any task means the 2G request is too tight — raise
`--mem` and resubmit only the failed rows:

```bash
sbatch --array=<comma,separated,failed,ids> --mem=4G submit.slurm
```

## 7. Analyse ARM 1

```bash
cd $ARMS/arm1
bash analyse_results.sh
```

It pools the cluster rows with the completed local `A-P96` and `A-BUD` blocks
automatically (`PPSQJ_REPO` must be set), giving the full ladder
`N_c` = 32 / 64 / 128 / 256 / 512 at `R` = 128 / 64 / 48 / 48 / 48.

Read the verdict against the frozen rule in `arm1/analysis_spec.yaml`:
**SUPPORTED** / **KILLED** / **INCONCLUSIVE**, with INCONCLUSIVE dominating, and
gated to UNTESTED AT HIGH VIF if the median measured VIF is below 40.

Save the output:

```bash
bash analyse_results.sh | tee arm1_analysis_$(date -u +%Y%m%dT%H%M%SZ).txt
```

## 8. Copy ARM 1 results back

```bash
# FROM YOUR LAPTOP
rsync -av <you>@ruche:$HOME/ppsQJ_m2/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/arm1/results/ \
    /Users/catlover1337/Documents/ppsQJ_m2/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/arm1/results/
rsync -av <you>@ruche:$HOME/ppsQJ_m2/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/arm1/arm1_analysis_*.txt \
    /Users/catlover1337/Documents/ppsQJ_m2/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/arm1/
```

---

## **STOP HERE AND READ THE ARM 1 VERDICT BEFORE ARM 2**

If ARM 1 returns **KILLED**, `N_eff` saturates, more particles cannot rescue
`L` ≥ 128, and ARM 2 is answering a question the programme no longer has. That
is a redirect of the whole campaign, not a tuning change. **Do not spend 194
core-hours on ARM 2 before reading ARM 1.**

---

## 9. ARM 2 preflight

```bash
cd $ARMS/arm2
bash run_preflight.sh
```

Confirm: `manifest rows` = **192**, `--array` = `0-191%64  OK`, `L` 128,
`T` 128, `zeta` 0.35, `lambda` 0.3032, `N_c` 64/128/256 at `R` = 64 each,
`expected core-hours` ≈ **194.1**, spec sha256 `3edf6ce746f676d0…`,
`--partition` = **`cpu_long  MaxTime 168 h  vs requested 8 h`**, and
**`PREFLIGHT PASSED`**.

ARM 2 uses `cpu_long` because its frozen `--time` is `08:00:00` and `cpu_med`
caps at 4 h. The wall request was not reduced to fit a smaller partition.

**Then run one task. This matters more here than for ARM 1**, because the
`L` = 128 cost is *derived* from a 96→128 ratio of 2.250 and not measured — no
`L` = 128 run exists anywhere in the programme:

```bash
time "$PPSQJ_PYTHON" run_cell.py 0 ./results   # expect ~1-2.5 h
```

If it lands outside ~1–2.5 h, the 194 core-hour estimate is wrong. Re-cost before
queueing 192 tasks.

## 10. ARM 2 submission

```bash
cd $ARMS/arm2
mkdir -p logs results
sbatch submit.slurm
```

## 11. Monitor ARM 2

```bash
squeue -u $USER -j <JOBID>
ls $ARMS/arm2/results | wc -l                # of 192
sacct -j <JOBID> --format=JobID%20,State,ExitCode,Elapsed,MaxRSS -X | head -30
```

## 12. Analyse ARM 2

```bash
cd $ARMS/arm2
bash analyse_results.sh | tee arm2_analysis_$(date -u +%Y%m%dT%H%M%SZ).txt
```

**CALIBRATED** only if the achieved MDE|B| ≤ 3.5. The projection is ≈3.4, so it
only just clears. If it lands above 3.5 the honest outcome is **UNRESOLVED** and
the calibration table must not be widened to accommodate it.

## 13. Copy ARM 2 results back

```bash
# FROM YOUR LAPTOP
rsync -av <you>@ruche:$HOME/ppsQJ_m2/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/arm2/results/ \
    /Users/catlover1337/Documents/ppsQJ_m2/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/arm2/results/
rsync -av <you>@ruche:$HOME/ppsQJ_m2/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/arm2/arm2_analysis_*.txt \
    /Users/catlover1337/Documents/ppsQJ_m2/research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY/arm2/
```

---

## After both arms

Regenerate the calibration and the production table locally:

```bash
cd /Users/catlover1337/Documents/ppsQJ_m2
C=research/tasks/active/TASK-2026-08-31-SMCCERT
.venv/bin/python3 $C/analysis/analyse_cert.py CERT-1
.venv/bin/python3 $C/analysis/emit_calibration.py
.venv/bin/python3 tools/plan_cloning_statistics.py \
    --L 128 --T 128 --zeta 0.35 --lam 0.3032 --target-sem 0.010
```

The last command currently returns `CALIBRATION_REQUIRED`. After ARM 2 it should
return a configuration — **and if it still refuses, that is the correct answer**,
not a bug: it means the achieved MDE|B| exceeded 3.5.

## One caveat to carry into any promotion

`ADVERSARIAL_SANITY_CHECK.md` §1 found that the two disjoint seed blocks at the
`L` = 64 cell **formally disagree** on `B` (difference CI [+0.302, +6.254],
p ≈ 0.031). A random-effects widening raises that cell's `N_c` floor by ≈30%.
The shipped calibration intervals are therefore **too narrow**, and this should
be resolved before anything in
`tools/calibration/bias_calibration.json` is promoted to `research/state/**`.
