#!/usr/bin/env python3
"""Generate every arm package for TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION.

THE DESIGN IS FROZEN IN THIS FILE. Editing an arm's manifest by hand is an
error; regenerate here and re-run every check recorded in ../VALIDATION.md.

This script writes files. It contains no scheduler call and cannot submit.
"""
import os, sys, csv, json, math, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
from cost_model import (n_steps, wall_s, wall_s_affine, wall_s_maxrate, mem_mb,
                        elapsed_h, AFFINE, RATE_MAX_MS, FIT_RANGE,
                        PESSIMISTIC, DTAU_MULT, NC)

# --- THE FROZEN 17-POINT LAMBDA GRID ---------------------------------------
# The predecessor's 13 points EXTENDED DOWNWARD by four, at the same
# delta_lambda = 0.010. Grid indices 0-3 are NEW and computed by this task;
# indices 4-16 are the predecessor's 13 points and are REUSED, never recomputed.
#
#     GRID[4:]  ==  [round(0.2332 + 0.010*i, 4) for i in range(13)]
#
# asserted below, so the join cannot drift by a floating-point hair.
GRID = [round(0.1932 + 0.010 * i, 4) for i in range(17)]
OLD_GRID = [round(0.2332 + 0.010 * i, 4) for i in range(13)]
NEW_IDX = [0, 1, 2, 3]
NEW_LAMS = [GRID[i] for i in NEW_IDX]
assert GRID[4:] == OLD_GRID, "the extended grid does not contain the old grid"
assert NEW_LAMS == [0.1932, 0.2032, 0.2132, 0.2232], NEW_LAMS

ZETA = 0.35
R = 24
FIELDS = ["arm", "L", "T", "N_c", "zeta", "lam", "dtau_mult",
          "resample_scheme", "seed"]

# --- SEEDS -----------------------------------------------------------------
# Block [32e6, 33e6). The predecessor allocated [31,000,000, 31,612,047]; the
# floor here is 387,953 above its ceiling, so disjointness is STRUCTURAL and
# not merely observed. Lane rule, unchanged in form from the predecessor:
#
#     seed = seed_base[arm] + 1000 * grid_index + replicate_index
#
# grid_index indexes the FULL 17-POINT grid, so lanes 4-16 are permanently
# reserved for the already-measured lambdas and can never be handed to a new
# one by accident. This task uses lanes 0-3 only.
SEED_FLOOR = 32_000_000
SEED_CEIL = 33_000_000

# --- PARTITION -------------------------------------------------------------
# cpu_med for ALL THREE arms, by explicit instruction and on measured evidence,
# NOT by the predecessor's "smallest partition that fits --time" rule. See
# ../SCHEDULER_DECISION.md. Two of these arms would fit inside cpu_short's
# 1 h MaxTime; they are not sent there, because on this account cpu_short is
# serialised by QOSMaxJobsPerUserLimit and delivers no parallelism.
PARTITION = "cpu_med"
CONCURRENCY = 64

ARMS = [
    dict(name="lowlamL32", arm="X32", L=32, T=32.0, time="00:20:00", mem="1G",
         purpose="LOWLAM-L32: the four new low-lambda points at L = 32, "
                 "T = 32, N_c = 1024, R = 24. Completes the L = 32 curve to "
                 "17 points."),
    dict(name="lowlamL48", arm="X48", L=48, T=48.0, time="00:45:00", mem="1G",
         purpose="LOWLAM-L48: the four new low-lambda points at L = 48, "
                 "T = 48, N_c = 1024, R = 24. Completes the L = 48 curve to "
                 "17 points."),
    dict(name="lowlamL64", arm="X64", L=64, T=64.0, time="02:00:00", mem="2G",
         purpose="LOWLAM-L64: the four new low-lambda points at L = 64, "
                 "T = 64, N_c = 1024, R = 24. Completes the L = 64 curve to "
                 "17 points and is this campaign's wall-clock long pole."),
]
for i, a in enumerate(ARMS):
    a["seed_base"] = SEED_FLOOR + 100_000 * i


def render(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def main():
    seeds_all, summary = {}, []
    for a in ARMS:
        d = os.path.join(TASK, a["name"])
        os.makedirs(os.path.join(d, "results"), exist_ok=True)
        render(os.path.join(d, "results", ".gitkeep"), "")

        rows = []
        for gi in NEW_IDX:
            lam = GRID[gi]
            for i in range(R):
                s = a["seed_base"] + 1000 * gi + i
                assert SEED_FLOOR <= s < SEED_CEIL, f"seed {s} outside the block"
                assert s not in seeds_all, f"seed collision {s}"
                seeds_all[s] = a["name"]
                rows.append({"arm": a["arm"], "L": a["L"], "T": a["T"],
                             "N_c": NC, "zeta": ZETA, "lam": lam,
                             "dtau_mult": DTAU_MULT,
                             "resample_scheme": "systematic", "seed": s})
        # lineterminator="\n": the csv module defaults to CRLF, which makes
        # `git diff --check` report every manifest row as trailing whitespace.
        with open(os.path.join(d, "manifest.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

        ws = [wall_s(a["L"], a["T"], l) for l in NEW_LAMS]
        slow = max(ws) / 3600.0
        ch = sum(w * R for w in ws) / 3600.0
        el = elapsed_h(len(rows), ch, slow, CONCURRENCY)
        mem = mem_mb(a["L"])
        summary.append(dict(
            name=a["name"], arm=a["arm"], tasks=len(rows), L=a["L"], N_c=NC,
            R=R, n_lambda=len(NEW_IDX), lams=NEW_LAMS,
            n_steps=[n_steps(a["L"], a["T"], l) for l in NEW_LAMS],
            wall_s_affine=[round(wall_s_affine(a["L"], a["T"], l), 1)
                           for l in NEW_LAMS],
            wall_s_maxrate=[round(wall_s_maxrate(a["L"], a["T"], l), 1)
                            for l in NEW_LAMS],
            wall_s_adopted=[round(w, 1) for w in ws],
            slowest_h=round(slow, 4), pess_slowest_h=round(slow * PESSIMISTIC, 4),
            core_h=round(ch, 2), pess_core_h=round(ch * PESSIMISTIC, 2),
            elapsed_h=round(el, 3), pess_elapsed_h=round(el * PESSIMISTIC, 3),
            concurrency=CONCURRENCY, mem_mb=round(mem), partition=PARTITION,
            time=a["time"], mem_req=a["mem"],
            seed_lo=min(r["seed"] for r in rows),
            seed_hi=max(r["seed"] for r in rows)))

        for f in ("run_cell.py", "preflight.py", "run_preflight.sh",
                  "analyse_arm.py", "analyse_results.sh"):
            shutil.copy2(os.path.join(TASK, "shared", f), os.path.join(d, f))
            if f.endswith(".sh"):
                os.chmod(os.path.join(d, f), 0o755)

        render(os.path.join(d, "submit.slurm"), slurm(a, len(rows), slow, ch, el))
        render(os.path.join(d, "README.md"), readme(a, rows, slow, ch, el, mem))
        print(f"{a['name']:<12} {len(rows):>4} rows  slowest {slow * 60:5.1f} min  "
              f"{ch:6.2f} core-h  elapsed@{CONCURRENCY} {el * 60:5.1f} min  "
              f"seeds {min(r['seed'] for r in rows)}-{max(r['seed'] for r in rows)}")

    json.dump(sorted(seeds_all), open(os.path.join(HERE, "allocated_seeds.json"), "w"))
    json.dump(dict(grid=GRID, new_lambdas=NEW_LAMS, new_idx=NEW_IDX,
                   partition=PARTITION, concurrency=CONCURRENCY, arms=summary),
              open(os.path.join(HERE, "cost_summary.json"), "w"), indent=1)
    tot = sum(s["core_h"] for s in summary)
    worst = max(s["elapsed_h"] for s in summary)
    print(f"\n{len(seeds_all)} seeds allocated, all distinct, "
          f"range {min(seeds_all)}-{max(seeds_all)}")
    print(f"{sum(s['tasks'] for s in summary)} tasks, {tot:.2f} core-hours "
          f"({tot * PESSIMISTIC:.2f} pessimistic)")
    print(f"campaign elapsed if all three run concurrently = slowest arm = "
          f"{worst:.2f} h ({worst * PESSIMISTIC:.2f} h pessimistic), "
          f"EXCLUDING queue wait")
    return 0


def slurm(a, n, slow, ch, el):
    lams = ", ".join(f"{l:g}" for l in NEW_LAMS)
    ns = ", ".join(str(n_steps(a["L"], a["T"], l)) for l in NEW_LAMS)
    tag = a["arm"].lower()
    L = a["L"]
    aa, bb = AFFINE[L]
    return f"""#!/bin/bash
#SBATCH --job-name=lowlam-{tag}
#SBATCH --partition={PARTITION}
#SBATCH --array=0-{n - 1}%{CONCURRENCY}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={a['mem']}
#SBATCH --time={a['time']}
#SBATCH --output=logs/{tag}_%A_%a.out
#SBATCH --error=logs/{tag}_%A_%a.err
#
# ============================================================================
# {a['name']} -- TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage and gate. The researcher types the submission
# command by hand. Neither this file nor the preflight contains one.
#
# PURPOSE
#   {a['purpose']}
#
#   Only the FOUR NEW lambdas are here. The thirteen lambdas
#   0.2332 ... 0.3532 are ABSENT from this manifest on purpose:
#   TASK-2026-09-02-MOCK-PRODUCTION already measured them at this exact
#   (L, T, zeta, N_c, dtau_mult, resample_scheme) and they are carried in
#   ../frozen_inputs/predecessor_nc1024_populations.csv. Recomputing them
#   would have cost ~195 core-hours and bought nothing. See
#   ../REUSE_AND_DEDUP_AUDIT.md.
#
# CELL (frozen; do not hand-edit -- regenerate with tools/build_arms.py)
#   L = {L}, T = {a['T']:g}, zeta = {ZETA}, N_c = {NC},
#   4 NEW lambdas, grid indices 0-3 of the frozen 17-point grid:
#     {lams}
#   n_steps: {ns}
#   dtau_mult = {DTAU_MULT:g}, systematic resampling,
#   R = {R} independent populations per lambda.
#
# COST -- FITTED TO MEASURED RUCHE wall_s, never to a requested --time.
#   {n} array tasks, ~{ch:.2f} core-hours total, slowest single task
#   ~{slow * 60:.1f} min predicted ({slow * PESSIMISTIC * 60:.1f} min pessimistic),
#   elapsed ~{el * 60:.1f} min at the cap below ({el * PESSIMISTIC * 60:.1f} min
#   pessimistic), EXCLUDING queue wait, peak ~{mem_mb(L):.0f} MB per task.
#
#   Model: wall_s = {aa:.6f} * n_steps + {bb:.2f}, least squares over the
#   predecessor's completed N_c=1024 runs at this L (n_steps span
#   {FIT_RANGE[L][0]}-{FIT_RANGE[L][1]}), taken against the alternative
#   {RATE_MAX_MS[L]:.3f} ms/clone-window worst observed rate, larger adopted.
#   Provenance: ../COST_MODEL.md.
#
# PARTITION -- cpu_med, by design, for ALL THREE arms.
#   This is NOT the predecessor's "smallest partition that fits --time" rule.
#   The preceding campaign showed cpu_short is effectively serialised for this
#   account by QOSMaxJobsPerUserLimit while cpu_med delivered real parallelism.
#   --time={a['time']} would fit cpu_short's 1 h MaxTime on two of the three
#   arms; it is deliberately not sent there. See ../SCHEDULER_DECISION.md.
#   preflight.py requires cpu_med and EXITS NONZERO on anything else.
#
# --array=0-{n - 1}  : ONE task per manifest row, exactly. Do not change the
#                range; preflight.py fails if it stops matching the manifest.
#   %{CONCURRENCY}        : a concurrency cap only. 96 tasks at %{CONCURRENCY} is exactly two
#                waves, which is why the elapsed figure above is floored at
#                two slowest-tasks rather than at throughput.
# ============================================================================

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# A batch job does not reliably inherit an interactive PATH, so the interpreter
# is resolved EXPLICITLY here rather than trusting `python3`. There is no conda
# on Ruche; this is a plain venv/prefix on the work filesystem.
PPSQJ_PYTHON="${{PPSQJ_PYTHON:-/gpfs/workdir/ercetinut/envs/pps_qj/bin/python}}"
if [ ! -x "$PPSQJ_PYTHON" ]; then
    echo "PPSQJ_PYTHON is not executable: $PPSQJ_PYTHON" >&2
    echo "Export PPSQJ_PYTHON to the validated interpreter, e.g." >&2
    echo "  export PPSQJ_PYTHON=\\$WORKDIR/envs/pps_qj/bin/python" >&2
    exit 2
fi
export PATH="$(dirname "$PPSQJ_PYTHON"):$PATH"

echo "[{tag}] task ${{SLURM_ARRAY_TASK_ID}} on $(hostname) at $(date -u +%FT%TZ)"
echo "[{tag}] partition=${{SLURM_JOB_PARTITION:-?}}  python=$PPSQJ_PYTHON"
"$PPSQJ_PYTHON" -c 'import sys,numpy;print("[{tag}] resolved",sys.executable,"numpy",numpy.__version__)'

"$PPSQJ_PYTHON" run_cell.py "${{SLURM_ARRAY_TASK_ID}}" "${{SLURM_SUBMIT_DIR}}/results"
echo "[{tag}] task ${{SLURM_ARRAY_TASK_ID}} done at $(date -u +%FT%TZ)"
"""


def readme(a, rows, slow, ch, el, mem):
    L = a["L"]
    lams = ", ".join(f"{l:g}" for l in NEW_LAMS)
    ns = ", ".join(str(n_steps(L, a["T"], l)) for l in NEW_LAMS)
    sm, sx = min(r["seed"] for r in rows), max(r["seed"] for r in rows)
    aa, bb = AFFINE[L]
    return f"""# {a['name']} — TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION

{a['purpose']}

This arm computes **only the four new lambdas**. The thirteen already measured
at this exact cell by `TASK-2026-09-02-MOCK-PRODUCTION` are reused from
`../frozen_inputs/predecessor_nc1024_populations.csv` and are **not** recomputed.

| | |
|---|---|
| L | {L} |
| T | {a['T']:g} |
| zeta | {ZETA} |
| lambda (NEW) | {lams} |
| lambda (reused, not here) | 0.2332 … 0.3532, 13 points |
| grid | indices 0–3 of the frozen 17-point grid |
| N_c | {NC} |
| R per lambda | {R} |
| dtau_mult | {DTAU_MULT:g} (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | {ns} |
| array tasks | {len(rows)} |
| seeds | {sm}–{sx} (fresh and disjoint — see `../SEED_LEDGER.md`) |
| cost model | `wall_s = {aa:.6f}·n_steps + {bb:.2f}`, fitted to measured Ruche `wall_s` |
| slowest task | {slow * 60:.1f} min predicted, {slow * PESSIMISTIC * 60:.1f} min pessimistic |
| core-hours | {ch:.2f} predicted, {ch * PESSIMISTIC:.2f} pessimistic |
| elapsed at cap %{CONCURRENCY} | {el * 60:.1f} min predicted, {el * PESSIMISTIC * 60:.1f} min pessimistic (queue wait excluded) |
| peak memory | {mem:.0f} MB (requesting {a['mem']}) |
| partition | **{PARTITION}** (`--time={a['time']}`) — see `../SCHEDULER_DECISION.md` |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, transfer back and analysis — is in `../RUCHE_RUNBOOK.md`.
`run_preflight.sh` must exit 0 first; it submits nothing and contains no
scheduler call.

No agent submits this. You do.
"""


if __name__ == "__main__":
    sys.exit(main())
