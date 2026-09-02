#!/usr/bin/env python3
"""Generate every arm package for TASK-2026-09-02-MOCK-PRODUCTION.

THE DESIGN IS FROZEN IN THIS FILE. Editing an arm's manifest by hand is an
error; regenerate here and re-run every check recorded in ../VALIDATION.md.

This script writes files. It contains no scheduler call and cannot submit.
"""
import os, sys, csv, json, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
from cost_model import (n_steps, wall_s, mem_mb, elapsed_h, rate_ms,
                        PESSIMISTIC, DTAU_MULT)

# --- THE FROZEN LAMBDA GRID -----------------------------------------------
# 13 points, delta_lambda = 0.010, identical at every L. Endpoints justified
# from the measured zeta=0.35 corpus in ../LAMBDA_GRID_DECISION.md. The three
# ARM-B lambdas 0.2932/0.3032/0.3132 are grid indices 6, 7, 8 and are REUSED,
# never recomputed.
GRID = [round(0.2332 + 0.010 * i, 4) for i in range(13)]
REUSED_L64_NC1024 = {6: 0.2932, 7: 0.3032, 8: 0.3132}

ZETA = 0.35
FIELDS = ["arm", "L", "T", "N_c", "zeta", "lam", "dtau_mult", "resample_scheme", "seed"]

# lam_idx entries are indices into GRID, so a seed lane is tied to a lambda and
# not to an arm-local position. mockL64 skips lanes 6-8 because those cells
# already exist; the skipped lanes are never reallocated.
ALL = list(range(13))
NEW64 = [i for i in ALL if i not in REUSED_L64_NC1024]
CENTRE3 = [6, 7, 8]

ARMS = [
    dict(name="mockL32", arm="M32", L=32, T=32.0, N_c=1024, lam_idx=ALL, R=24,
         seed_base=31_000_000, partition="cpu_short", time="01:00:00", mem="1G",
         concurrency=64, recommend=True, group="main",
         purpose="MOCK-L32: the full 13-point CMI(lambda) scan at L = 32, "
                 "T = 32, N_c = 1024. The cheapest of the three curves and the "
                 "one with no historical counterpart at any N_c."),
    dict(name="mockL48", arm="M48", L=48, T=48.0, N_c=1024, lam_idx=ALL, R=24,
         seed_base=31_100_000, partition="cpu_med", time="02:00:00", mem="1G",
         concurrency=64, recommend=True, group="main",
         purpose="MOCK-L48: the full 13-point CMI(lambda) scan at L = 48, "
                 "T = 48, N_c = 1024."),
    dict(name="mockL64", arm="M64", L=64, T=64.0, N_c=1024, lam_idx=NEW64, R=24,
         seed_base=31_200_000, partition="cpu_med", time="03:00:00", mem="2G",
         concurrency=64, recommend=True, group="main",
         purpose="MOCK-L64: the 13-point scan at L = 64, T = 64, N_c = 1024, "
                 "MINUS the three lambdas already measured at R = 96 by "
                 "TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA ARM B. Ten new lambdas, "
                 "not thirteen. This is the campaign's wall-clock long pole."),
    dict(name="mockL64nc2048", arm="M64H", L=64, T=64.0, N_c=2048,
         lam_idx=CENTRE3, R=24, seed_base=31_300_000,
         partition="cpu_med", time="04:00:00", mem="3G",
         concurrency=64, recommend=True, group="main",
         purpose="MOCK-L64-NC2048: the shape check. Three lambdas at N_c = 2048 "
                 "against ARM B's N_c = 1024 at exactly the same three lambdas, "
                 "to measure whether the finite-population correction "
                 "Delta_N(lambda) = I_2048 - I_1024 is a common shift, "
                 "lambda-dependent, or unresolved. No 1/N_c law is fitted."),
    dict(name="mockNC128L32", arm="C32", L=32, T=32.0, N_c=128, lam_idx=ALL, R=48,
         seed_base=31_400_000, partition="cpu_short", time="01:00:00", mem="1G",
         concurrency=64, recommend=True, group="companion",
         purpose="MATCHED LOW-N_c COMPANION at L = 32. Same grid, same "
                 "dtau_mult = 6, same estimator, same code, only N_c differs. "
                 "See ../NC128_COMPANION_RATIONALE.md: the historical N_c = 128 "
                 "corpus is dtau_mult = 12 and shares NO exactly compatible "
                 "cell with this campaign, so without this arm brief sections 9C "
                 "and 12 have no matched comparison to make."),
    dict(name="mockNC128L48", arm="C48", L=48, T=48.0, N_c=128, lam_idx=ALL, R=48,
         seed_base=31_500_000, partition="cpu_short", time="01:00:00", mem="1G",
         concurrency=64, recommend=True, group="companion",
         purpose="MATCHED LOW-N_c COMPANION at L = 48."),
    dict(name="mockNC128L64", arm="C64", L=64, T=64.0, N_c=128, lam_idx=ALL, R=48,
         seed_base=31_600_000, partition="cpu_short", time="01:00:00", mem="1G",
         concurrency=64, recommend=True, group="companion",
         purpose="MATCHED LOW-N_c COMPANION at L = 64. This is also the one "
                 "cell class where the historical dtau_mult = 12 corpus exists "
                 "at the same L and N_c, so it isolates the discretisation "
                 "systematic from the population-size effect."),
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
        for gi in a["lam_idx"]:
            lam = GRID[gi]
            for i in range(a["R"]):
                s = a["seed_base"] + 1000 * gi + i
                assert s not in seeds_all, f"seed collision {s}"
                seeds_all[s] = a["name"]
                rows.append({"arm": a["arm"], "L": a["L"], "T": a["T"],
                             "N_c": a["N_c"], "zeta": ZETA, "lam": lam,
                             "dtau_mult": DTAU_MULT,
                             "resample_scheme": "systematic", "seed": s})
        # lineterminator="\n": the csv module defaults to CRLF, which makes
        # `git diff --check` report every manifest row as trailing whitespace.
        with open(os.path.join(d, "manifest.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

        ws = [wall_s(a["L"], a["T"], a["N_c"], GRID[i]) for i in a["lam_idx"]]
        slow = max(ws) / 3600.0
        ch = sum(w * a["R"] for w in ws) / 3600.0
        el = elapsed_h(ch, slow, a["concurrency"])
        mem = mem_mb(a["L"], a["N_c"])
        summary.append(dict(name=a["name"], arm=a["arm"], group=a["group"],
                            tasks=len(rows), L=a["L"], N_c=a["N_c"], R=a["R"],
                            n_lambda=len(a["lam_idx"]),
                            lams=[GRID[i] for i in a["lam_idx"]],
                            rate_ms=round(rate_ms(a["L"], a["N_c"]), 3),
                            slowest_h=round(slow, 3),
                            pess_slowest_h=round(slow * PESSIMISTIC, 3),
                            core_h=round(ch, 1),
                            pess_core_h=round(ch * PESSIMISTIC, 1),
                            elapsed_h=round(el, 2),
                            pess_elapsed_h=round(el * PESSIMISTIC, 2),
                            concurrency=a["concurrency"],
                            mem_mb=round(mem), partition=a["partition"],
                            time=a["time"], mem_req=a["mem"],
                            recommend=a["recommend"]))

        for f in ("run_cell.py", "preflight.py", "run_preflight.sh",
                  "analyse_arm.py", "analyse_results.sh"):
            shutil.copy2(os.path.join(TASK, "shared", f), os.path.join(d, f))
            if f.endswith(".sh"):
                os.chmod(os.path.join(d, f), 0o755)

        render(os.path.join(d, "submit.slurm"), slurm(a, len(rows), slow, ch, el))
        render(os.path.join(d, "README.md"), readme(a, rows, slow, ch, el, mem))
        print(f"{a['name']:<16} {len(rows):>4} rows  slowest {slow:5.2f} h  "
              f"{ch:7.1f} core-h  elapsed@{a['concurrency']} {el:5.2f} h  seeds "
              f"{min(r['seed'] for r in rows)}-{max(r['seed'] for r in rows)}")

    json.dump(sorted(seeds_all), open(os.path.join(HERE, "allocated_seeds.json"), "w"))
    json.dump(dict(grid=GRID, arms=summary),
              open(os.path.join(HERE, "cost_summary.json"), "w"), indent=1)
    main_h = sum(s["core_h"] for s in summary if s["group"] == "main")
    comp_h = sum(s["core_h"] for s in summary if s["group"] == "companion")
    worst = max(s["elapsed_h"] for s in summary)
    print(f"\n{len(seeds_all)} seeds allocated, all distinct, "
          f"range {min(seeds_all)}-{max(seeds_all)}")
    print(f"main arms {main_h:.1f} core-h, companion arms {comp_h:.1f} core-h, "
          f"total {main_h + comp_h:.1f} core-h")
    print(f"campaign elapsed if all arms run concurrently = slowest arm = "
          f"{worst:.2f} h ({worst * PESSIMISTIC:.2f} h pessimistic)")
    return 0


def slurm(a, n, slow, ch, el):
    lams = ", ".join(f"{GRID[i]:g}" for i in a["lam_idx"])
    tag = a["arm"].lower()
    c = a["concurrency"]
    reuse = ("\n#   The three lambdas 0.2932 / 0.3032 / 0.3132 are ABSENT from this\n"
             "#   manifest on purpose: TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA ARM B\n"
             "#   already holds R = 96 populations at each of them, at this exact\n"
             "#   (L, T, zeta, N_c, dtau_mult, resample_scheme). Recomputing them\n"
             "#   would have cost ~39 core-hours and bought nothing. See\n"
             "#   ../REUSE_AND_DEDUP_AUDIT.md.\n#"
             if a["name"] == "mockL64" else "")
    return f"""#!/bin/bash
#SBATCH --job-name=mock-{tag}
#SBATCH --partition={a['partition']}
#SBATCH --array=0-{n - 1}%{c}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={a['mem']}
#SBATCH --time={a['time']}
#SBATCH --output=logs/{tag}_%A_%a.out
#SBATCH --error=logs/{tag}_%A_%a.err
#
# ============================================================================
# {a['name']} -- TASK-2026-09-02-MOCK-PRODUCTION
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage and gate. The researcher types the submission
# command by hand. Neither this file nor the preflight contains one.
#
# PURPOSE
#   {a['purpose']}
#{reuse}
# CELL (frozen; do not hand-edit -- regenerate with tools/build_arms.py)
#   L = {a['L']}, T = {a['T']:g}, zeta = {ZETA}, N_c = {a['N_c']},
#   {len(a['lam_idx'])} lambdas on the frozen 13-point grid:
#     {lams}
#   dtau_mult = {DTAU_MULT:g}, systematic resampling,
#   R = {a['R']} independent populations per lambda.
#
# COST (from tools/cost_model.py; re-run run_preflight.sh, do not trust this)
#   {n} array tasks, ~{ch:.0f} core-hours total, slowest single task
#   ~{slow:.2f} h predicted ({slow * PESSIMISTIC:.2f} h pessimistic),
#   elapsed ~{el:.2f} h at the cap below ({el * PESSIMISTIC:.2f} h pessimistic),
#   peak ~{mem_mb(a['L'], a['N_c']):.0f} MB per task.
#
#   The rate is {rate_ms(a['L'], a['N_c']):.3f} ms per clone-window, anchored on
#   wall_s ACTUALLY RECORDED by completed Ruche jobs of this identical code
#   path -- never on a requested --time. Provenance: ../COST_MODEL.md.
#
# PARTITION -- the SMALLEST valid partition for this arm's request, chosen from
# the request, not inherited from any predecessor.
#   Ruche MaxTime: cpu_short 1 h, cpu_med 4 h, cpu_long 7 d.
#   requested {a['time']}  ->  {a['partition']}
#   preflight.py re-derives this and EXITS NONZERO on any mismatch.
#
# --array=0-{n - 1}  : ONE task per manifest row, exactly. Do not change the
#                 range; preflight.py fails if it stops matching the manifest.
#   %{c}         : a concurrency cap only. On 2026-09-02 the predecessor's ARM B
#                 demonstrated that this allocation grants 64 concurrent slots
#                 PER ARRAY with three arrays live at once. LOWER IT if the
#                 accounting check in ../RUCHE_RUNBOOK.md section 2 says your
#                 slots are shared in total rather than per array.
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
    lams = ", ".join(f"{GRID[i]:g}" for i in a["lam_idx"])
    sm, sx = min(r["seed"] for r in rows), max(r["seed"] for r in rows)
    ns = ", ".join(str(n_steps(a["L"], a["T"], GRID[i])) for i in a["lam_idx"])
    grp = ("**Main mock-production arm.**" if a["group"] == "main" else
           "**Matched low-`N_c` companion arm.** An addition to the task brief's "
           "arm list; the reason it exists is in `../NC128_COMPANION_RATIONALE.md`, "
           "and dropping it costs the campaign nothing except brief sections 9C "
           "and 12.")
    return f"""# {a['name']} — TASK-2026-09-02-MOCK-PRODUCTION

{grp}

{a['purpose']}

| | |
|---|---|
| L | {a['L']} |
| T | {a['T']:g} |
| zeta | {ZETA} |
| lambda | {lams} |
| lambdas | {len(a['lam_idx'])} of the frozen 13-point grid |
| N_c | {a['N_c']} |
| R per lambda | {a['R']} |
| dtau_mult | {DTAU_MULT:g} (the CERTIFIED production value — never the corpus 12) |
| resampling | systematic |
| n_steps | {ns} |
| array tasks | {len(rows)} |
| seeds | {sm}–{sx} (fresh and disjoint — see `../SEED_LEDGER.md`) |
| rate | {rate_ms(a['L'], a['N_c']):.3f} ms/clone-window (`../COST_MODEL.md`) |
| slowest task | {slow:.2f} h predicted, {slow * PESSIMISTIC:.2f} h pessimistic |
| core-hours | {ch:.1f} predicted, {ch * PESSIMISTIC:.1f} pessimistic |
| elapsed at cap %{a['concurrency']} | {el:.2f} h predicted, {el * PESSIMISTIC:.2f} h pessimistic |
| peak memory | {mem:.0f} MB (requesting {a['mem']}) |
| partition | {a['partition']} (`--time={a['time']}`) |

## Running it

The exact command sequence — preflight, submission, monitoring, completeness
check, transfer back and analysis — is in `../RUCHE_RUNBOOK.md`.
`run_preflight.sh` must exit 0 first; it submits nothing and contains no
scheduler call.

No agent submits this. You do.
"""


if __name__ == "__main__":
    sys.exit(main())
