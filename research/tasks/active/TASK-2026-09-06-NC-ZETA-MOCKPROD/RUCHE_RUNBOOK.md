# RUCHE_RUNBOOK — TASK-2026-09-06-NC-ZETA-MOCKPROD

**Terminal state: `READY_FOR_HUMAN_SUBMISSION`.**

**No agent submitted anything and no agent may.**
`research/RESOURCE_POLICY.md` §4, unconditionally, at every stage, gate and
approval level. Every command below is typed **by the researcher**. Nothing in
this package contains a scheduler call — preflight `P12` checks that, and
negative controls prove the preflight's checks can fail.

Labels `[E]` `[I]` `[C]` `[J]`.

---

## 0. First: this package was repaired on 2026-09-08

`[E]` The first submission of this task — **jobs 1694328 – 1694621** — produced
**zero result JSONs**. Every array task died before the sampler with
`FileNotFoundError` on `shared/manifest.csv`; the arrays were cancelled and no
scientific data from that submission is retained. Cause, repair and the checks
that now prove it: **`RUCHE_INCIDENT_2026-09-08.md`**.

`[E]` The repair is **packaging only**. `shared/run_cell.py` is unchanged byte
for byte, and no seed, grid point, manifest row, cost figure or sampler
parameter moved. What changed: every arm now carries its own byte-identical
`run_cell.py`, `shared/run_pack.py` invokes it, and each `submit.slurm` derives
`PPSQJ_REPO` from its own depth.

`[E]` **The cluster copy must be refreshed before anything is submitted.** An
arm directory on Ruche that predates the repair has no `run_cell.py` in it; the
job will exit 2 with `no arm-local run_cell.py` rather than fail silently, but
it will not run. Refresh the whole task directory — the arms, `shared/`,
`support/` and `conditional/support/` — then re-run the preflight **on Ruche**
(§1), because the preflight's P18–P20 test the files that are actually there.

---

## 0.1. Before anything

```bash
cd <repo>
export PPSQJ_PYTHON=$WORKDIR/envs/pps_qj/bin/python      # the validated interpreter
```

`[E]` There is no conda on Ruche. `submit.slurm` resolves the interpreter
explicitly rather than trusting an inherited `PATH`, and exits 2 if
`$PPSQJ_PYTHON` is not executable.

## 1. Preflight — run this first, every time

```bash
cd research/tasks/active/TASK-2026-09-06-NC-ZETA-MOCKPROD
bash shared/run_preflight.sh
```

`[E]` 24 checks per production arm, 22 for the control. It must print
`ALL ARMS PASS PREFLIGHT.` and exit 0. It reads only; it cannot submit.

`[E]` P18, P19 and P20 are the checks added by the 2026-09-08 repair. P19
*executes* `run_pack.py` from the arm and asks which executor it would run; P20
*executes* the arm-local executor from an unrelated working directory with an
index past the end of the manifest, and requires it to reach `IndexError` having
written nothing outside the arm. Both measure a resolution rather than reading
one out of the source, because the fault was a resolution.

`[E]` To satisfy yourself the checks are real, run the injected-fault suite —
16 faults, each into a **copy** of an arm staged at the arm's real depth, each
required to be rejected with a named code — followed by an end-to-end
reproduction of the 2026-09-08 failure and of the repair:

```bash
.venv/bin/python3 tools/negative_controls.py     # expect: 16 of 16, then 4 reproduction steps ok
.venv/bin/python3 tools/smoke_test.py            # expect: 13 of 13
.venv/bin/python3 tools/check_predecessor.py     # expect: predecessor isolation OK
```

`[E]` **One-task smoke test before any full array.** After refreshing the
cluster copy and passing the preflight there, run a single pack by hand on a
compute-capable node — no scheduler, one row-pack, in the cheapest arm — and
confirm result JSONs appear **in that arm**:

```bash
cd research/tasks/active/TASK-2026-09-06-NC-ZETA-MOCKPROD/M_z010_nc128
export PPSQJ_PYTHON=$WORKDIR/envs/pps_qj/bin/python
export PPSQJ_REPO=$(cd ../../../../.. && pwd)
$PPSQJ_PYTHON ../shared/run_pack.py --resolve      # must print RUN_CELL .../M_z010_nc128/run_cell.py
$PPSQJ_PYTHON ../shared/run_pack.py 0              # pack 0: the cheapest pack in the campaign
ls results/ | wc -l                                # must be > 0
```

The `--resolve` line is the whole incident in one command: before the repair it
printed a path in `shared/`. Delete the smoke-test JSONs afterwards, or leave
them — they are ordinary rows of pack 0 and the executor is idempotent, so the
real array will skip them rather than recompute them.

## 2. Submission order

`[E]` **Submit the committed arms. Do NOT submit `conditional/` yet.**

Cheapest first, so that a packaging problem surfaces on a 4 core-hour arm and
not on a 431 core-hour one:

```bash
cd research/tasks/active/TASK-2026-09-06-NC-ZETA-MOCKPROD

# --- wave 1: zeta = 0.10, 102 core-h total, the cheapest zeta -------------
for a in M_z010_nc128 M_z010_nc256 M_z010_nc512 M_z010_nc1024 M_z010_nc2048; do
  ( cd $a && <your submit command> submit.slurm )
done

# --- wave 2: zeta = 0.20, 221 core-h --------------------------------------
for a in M_z020_nc128 M_z020_nc256 M_z020_nc512 M_z020_nc1024 M_z020_nc2048; do
  ( cd $a && <your submit command> submit.slurm )
done

# --- wave 3: zeta = 0.70 up to N_c = 1024, 837 core-h ---------------------
for a in M_z070_nc128 M_z070_nc256 M_z070_nc512 M_z070_nc1024; do
  ( cd $a && <your submit command> submit.slurm )
done

# --- the discretisation control, 2.5 core-h -------------------------------
( cd E_dtau_z010 && <your submit command> submit.slurm )
```

`[E]` `<your submit command>` is deliberately left as a placeholder. Writing the
literal command here would put a submission line in an agent-authored file, and
this package does not contain one anywhere.

`[E]` **`E_dtau_z010` may be submitted at any time but cannot be ANALYSED until
`M_z010_nc512` has returned**, because its `dtau_mult = 6` leg lives in that arm.
The analysis reports `NOT EVALUABLE` and names the missing leg rather than
guessing.

`[I]` Wave 1 is 102 core-hours and finishes in well under an hour of compute.
`[J]` It is worth letting it complete and running §4 on it before submitting
wave 3, which is 8x its cost — the survey's whole point is that rung differences
are visible, and if wave 1 shows nothing visible at any rung, that is information
about `R` that changes what wave 3 is worth.

## 2b. Gate the two large `zeta = 0.70` arms on a timing readback

`[E]` `rho(zeta)` has **no production-regime validation anywhere in this
repository** — it comes from local probes at `N_c <= 500` with tiny `n_steps`,
and `rho(0.70)` is additionally understated by 7–9 % (`COST_MODEL.md` §4).
`[J]` So before submitting `M_z070_nc1024` (431 core-h) and, if released,
`conditional/M_z070_nc2048` (862 core-h), read the wall times back from the
first `zeta = 0.70` arm that returns:

```bash
sacct -j <jobid_of_M_z070_nc128> --format=JobID,Elapsed,TotalCPU,MaxRSS
```

`[E]` If the measured per-clone-window rate at `zeta = 0.70` exceeds the model
by more than the 1.40 pessimistic band, **stop and re-time the two large arms
before submitting them.** That is the cheapest possible check on the weakest
`zeta`-dependent input in the package, and `M_z070_nc128` is 67 core-hours.

## 3. The one measurement worth taking on the day, and it is free

```bash
sacct -j <jobid> --format=JobID,JobName,Elapsed,MaxRSS,State
```

`[E]` **There is no `MaxRSS` from any Ruche job anywhere in this repository.**
Every memory number in this programme, including every `--mem` in this package,
descends from local macOS `ru_maxrss` probes on a different allocator. One
`sacct` line on a completed `M_z070_nc2048` or `M_z010_nc2048` task would be the
first real measurement of this sampler's memory on the cluster, it settles the
question for the whole programme, and it costs nothing. `[J]` Please run it.

Also worth capturing, for the same reason:

```bash
sacct -j <jobid> --format=JobID,Elapsed,TotalCPU  > analysis/measured_walltimes.txt
```

`[E]` `COST_MODEL.md` predicts specific per-arm core-hours. `FALSIFICATION_PLAN.md`
F13 pre-registers that the model is killed if measured exceeds the pessimistic
figure — and the correction is recorded whichever way it errs, because the
programme has now been wrong in both directions.

## 4. Analysis

```bash
.venv/bin/python3 analysis/mockprod_analysis.py
```

`[E]` Runs on **whatever has returned**. Missing cells print as `MISSING` and
are never imputed; a rung with missing populations classifies `INCONCLUSIVE`
rather than being quietly skipped. Writes `MOCKPROD_ANALYSIS.txt`,
`MOCKPROD_RESULTS.json` and `analysis/figures/`.

`[E]` It implements `ANALYSIS_SPEC.yaml` and nothing else. A post-hoc analysis
is a **child task** (`research/tools/child_task.py propose`), never an edit to
the spec or the script.

## 5. The conditional rung

`[E]` **`conditional/M_z070_nc2048` is 861.6 core-hours — 42.6 % of the full
design — and is not submitted with the rest.** Read
`CONDITIONAL_SUBMISSION.md` first. It is complete, preflight-clean and ready;
the interlock is a decision, not a formality.

## 6. If a job dies

`[E]` **Requeue it; it is idempotent.** `run_cell.py` skips any row whose output
JSON already exists, so a timed-out or requeued pack costs only what it has left
to do, and a pack may be re-run freely.

`[E]` **But that mitigation is EMPTY where the jobs are longest.** A pack can
only skip *completed rows*, and in the two most expensive arms almost every pack
is a single row:

| arm | packs | rows per pack | one row |
|---|---:|---|---:|
| `conditional/M_z070_nc2048` | 432 | **1** | 4.9 h |
| `M_z070_nc1024` | 408 | 1 (384 of them) | 2.5 h |

`[I]` A `--time` overrun on one of those tasks loses the whole task **on every
requeue**, because there is no completed row to skip and no checkpointing in
this code path. `[E]` The `--time` margins there are 2.2x and 2.4x the
pessimistic figure, so an overrun means the cost model was wrong by more than
that — but if it happens, requeuing will not help and the fix is to raise
`--time`, not to resubmit. `[J]` Found by the red team; the runbook previously
implied a mitigation that does not exist in exactly the arms where it would
matter most.

`[E]` A pack whose individual row fails records the failure and **continues** —
one bad row does not discard the pack's other completed rows, which are already
durable on disk. The analysis counts missing rows explicitly and never treats an
absent row as a value.

`[E]` If a job dies on `ModuleNotFoundError`, check that `support/instrumented.py`
came with the checkout. That exact failure killed the first Ruche job of
`TASK-2026-09-01-SMCRUCHE-PACKFIX` because the sampler then lived in an untracked
directory. It is bundled and hash-checked at job start here.

## 7. What this campaign may not be used to say

`[E]` No `lambda_c(zeta)`. No boundary exponent; `DISP-PHI-001` stays open. No
`N_c^req(zeta)` and no fit of one. No transfer of a `zeta = 0.35` coefficient to
another `zeta`. Nothing about `L > 64` or `N_c > 2048`. **No `N_c` produced here
is converged, and the word "certified" does not apply to any result of this
survey.** `ROUGHLY STABLE` means the curves overlap within the resolution of a
`R = 16` survey, and nothing more.
