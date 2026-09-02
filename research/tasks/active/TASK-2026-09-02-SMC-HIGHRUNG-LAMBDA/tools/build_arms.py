#!/usr/bin/env python3
"""Generate every arm package for TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA.

THE DESIGN IS FROZEN IN THIS FILE. Editing an arm's manifest by hand is an
error; regenerate here and re-run the validation recorded in VALIDATION.md.

This script writes files. It contains no scheduler call and cannot submit.
"""
import os, sys, csv, json, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
from cost_model import n_steps, wall_s, mem_mb, PESSIMISTIC, DTAU_MULT

LAM_M, LAM_0, LAM_P = 0.2932, 0.3032, 0.3132     # FROZEN stencil, dlam = 0.010
ZETA = 0.35
FIELDS = ["arm", "L", "T", "N_c", "zeta", "lam", "dtau_mult", "resample_scheme", "seed"]

ARMS = [
    dict(name="armA512", arm="A512", L=128, T=128.0, N_c=512,
         lams=[(LAM_0, 0)], R=48, seed_base=30_100_000,
         partition="cpu_long", time="12:00:00", mem="3G", recommend=True,
         purpose="ARM A rung 1: L=128, N_c=512 at the central lambda. It also "
                 "supplies ARM C's central stencil point, which is therefore "
                 "NOT recomputed in armC."),
    dict(name="armA1024", arm="A1024", L=128, T=128.0, N_c=1024,
         lams=[(LAM_0, 0)], R=32, seed_base=30_200_000,
         partition="cpu_long", time="24:00:00", mem="5G", recommend=True,
         purpose="ARM A rung 2: L=128, N_c=1024 at the central lambda. This is "
                 "the wall-clock long pole of the whole campaign; submit it first."),
    dict(name="armB", arm="B", L=64, T=64.0, N_c=1024,
         lams=[(LAM_M, 0), (LAM_0, 1), (LAM_P, 2)], R=96, seed_base=30_300_000,
         partition="cpu_med", time="03:00:00", mem="2G", recommend=True,
         purpose="ARM B: cheap-L, high-population three-point lambda stencil. "
                 "Does a large enough population make CMI(lambda) a "
                 "statistically coherent local curve?"),
    dict(name="armC", arm="C", L=128, T=128.0, N_c=512,
         lams=[(LAM_M, 0), (LAM_P, 1)], R=48, seed_base=30_400_000,
         partition="cpu_long", time="12:00:00", mem="3G", recommend=True,
         purpose="ARM C: the same stencil at L=128, NEIGHBOURING lambdas only. "
                 "The central point is armA512's; it is not duplicated here."),
    dict(name="armA2048_optional", arm="A2048", L=128, T=128.0, N_c=2048,
         lams=[(LAM_0, 0)], R=16, seed_base=30_500_000,
         partition="cpu_long", time="48:00:00", mem="9G", recommend=False,
         purpose="OPTIONAL EXTENSION, NOT RECOMMENDED TONIGHT. See "
                 "NC2048_AUDIT.md: a single task is ~20 h predicted (~28 h "
                 "pessimistic) and cannot finish overnight, and whether it is "
                 "worth running at all is exactly what Delta_512->1024 answers "
                 "by morning."),
]


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
        for lam, lane in a["lams"]:
            for i in range(a["R"]):
                s = a["seed_base"] + 1000 * lane + i
                assert s not in seeds_all, f"seed collision {s}"
                seeds_all[s] = a["name"]
                rows.append({"arm": a["arm"], "L": a["L"], "T": a["T"],
                             "N_c": a["N_c"], "zeta": ZETA, "lam": lam,
                             "dtau_mult": DTAU_MULT,
                             "resample_scheme": "systematic", "seed": s})
        # lineterminator="\n": the csv module defaults to CRLF, which makes
        # `git diff --check` report every manifest row as trailing whitespace.
        # The predecessor's manifests are CRLF because it lacked this line.
        with open(os.path.join(d, "manifest.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

        ws = [wall_s(a["L"], a["T"], a["N_c"], lam) for lam, _ in a["lams"]]
        slow, ch = max(ws) / 3600.0, sum(w * a["R"] for w in ws) / 3600.0
        mem = mem_mb(a["L"], a["N_c"])
        summary.append(dict(name=a["name"], arm=a["arm"], tasks=len(rows),
                            L=a["L"], N_c=a["N_c"], R=a["R"],
                            lams=[l for l, _ in a["lams"]],
                            slowest_h=round(slow, 3),
                            pess_h=round(slow * PESSIMISTIC, 3),
                            core_h=round(ch, 1),
                            pess_core_h=round(ch * PESSIMISTIC, 1),
                            mem_mb=round(mem), partition=a["partition"],
                            time=a["time"], mem_req=a["mem"],
                            recommend=a["recommend"]))

        for f in ("run_cell.py", "preflight.py", "run_preflight.sh",
                  "analyse_arm.py", "analyse_results.sh"):
            shutil.copy2(os.path.join(TASK, "shared", f), os.path.join(d, f))
            if f.endswith(".sh"):
                os.chmod(os.path.join(d, f), 0o755)

        render(os.path.join(d, "submit.slurm"), slurm(a, len(rows), slow, ch))
        render(os.path.join(d, "README.md"), readme(a, rows, slow, ch, mem))
        print(f"{a['name']:<22} {len(rows):>4} rows  slowest {slow:5.2f} h  "
              f"{ch:7.1f} core-h  seeds "
              f"{min(r['seed'] for r in rows)}-{max(r['seed'] for r in rows)}")

    json.dump(sorted(seeds_all), open(os.path.join(HERE, "allocated_seeds.json"), "w"))
    json.dump(summary, open(os.path.join(HERE, "cost_summary.json"), "w"), indent=1)
    print(f"\n{len(seeds_all)} seeds allocated, all distinct, "
          f"range {min(seeds_all)}-{max(seeds_all)}")
    return 0


def slurm(a, n, slow, ch):
    rec = ("RECOMMENDED for the overnight campaign."
           if a["recommend"] else
           "NOT RECOMMENDED TONIGHT -- an optional extension. Read "
           "NC2048_AUDIT.md before queueing it.")
    lams = ", ".join(f"{l:g}" for l, _ in a["lams"])
    tag = a["arm"].lower()
    return f"""#!/bin/bash
#SBATCH --job-name=hrl-{tag}
#SBATCH --partition={a['partition']}
#SBATCH --array=0-{n - 1}%64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={a['mem']}
#SBATCH --time={a['time']}
#SBATCH --output=logs/{tag}_%A_%a.out
#SBATCH --error=logs/{tag}_%A_%a.err
#
# ============================================================================
# {a['name']} -- TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage and gate. The researcher types the submission
# command by hand. Neither this file nor the preflight contains one.
#
# {rec}
#
# PURPOSE
#   {a['purpose']}
#
# CELL (frozen; do not hand-edit -- regenerate with tools/build_arms.py)
#   L = {a['L']}, T = {a['T']:g}, zeta = {ZETA}, N_c = {a['N_c']},
#   lambda in {{{lams}}}, dtau_mult = {DTAU_MULT:g}, systematic resampling,
#   R = {a['R']} independent populations per lambda.
#
# COST (from tools/cost_model.py; re-run run_preflight.sh, do not trust this)
#   {n} array tasks, ~{ch:.0f} core-hours total, slowest single task
#   ~{slow:.2f} h predicted ({slow * PESSIMISTIC:.2f} h pessimistic),
#   peak ~{mem_mb(a['L'], a['N_c']):.0f} MB per task.
#
#   Unlike the predecessor package, the L = {a['L']} rate is MEASURED on Ruche
#   from completed ARM1/ARM2 runs of this identical code path. It is not the
#   Mac-probe extrapolation the predecessor had to flag at +/-50 %.
#
# PARTITION -- the SMALLEST valid partition for this arm's request, chosen from
# the request, not inherited from the predecessor.
#   Ruche MaxTime: cpu_short 1 h, cpu_med 4 h, cpu_long 7 d.
#   requested {a['time']}  ->  {a['partition']}
#   preflight.py re-derives this and EXITS NONZERO on any mismatch.
#
# --array=0-{n - 1}  : ONE task per manifest row, exactly. Do not change the
#                 range; preflight.py fails if it stops matching the manifest.
#   %64         : a concurrency cap only. LOWER IT if you queue several arms at
#                 once and the allocation gives you 64 slots in total rather
#                 than 64 per array -- see RUCHE_RUNBOOK.md.
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


def readme(a, rows, slow, ch, mem):
    lams = ", ".join(f"{l:g}" for l, _ in a["lams"])
    sm, sx = min(r["seed"] for r in rows), max(r["seed"] for r in rows)
    ns = ", ".join(str(n_steps(a["L"], a["T"], l)) for l, _ in a["lams"])
    head = ("**Recommended for the overnight campaign.**" if a["recommend"] else
            "**Optional extension. NOT recommended tonight — see "
            "`../NC2048_AUDIT.md`.**")
    return f"""# {a['name']} — TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA

{head}

{a['purpose']}

| | |
|---|---|
| L | {a['L']} |
| T | {a['T']:g} |
| zeta | {ZETA} |
| lambda | {lams} |
| N_c | {a['N_c']} |
| R per lambda | {a['R']} |
| dtau_mult | {DTAU_MULT:g} (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | {ns} |
| array tasks | {len(rows)} |
| seeds | {sm}–{sx} (fresh and disjoint — see `../SEED_LEDGER.md`) |
| slowest task | {slow:.2f} h predicted, {slow * PESSIMISTIC:.2f} h pessimistic |
| core-hours | {ch:.1f} predicted, {ch * PESSIMISTIC:.1f} pessimistic |
| peak memory | {mem:.0f} MB (requesting {a['mem']}) |
| partition | {a['partition']} (`--time={a['time']}`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, rsync back and analysis — is in `../RUCHE_RUNBOOK.md`. `run_preflight.sh`
must exit 0 first; it submits nothing and contains no scheduler call.

No agent submits this. You do.
"""


if __name__ == "__main__":
    sys.exit(main())
