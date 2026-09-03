#!/usr/bin/env python3
"""Generate the BLOCKED, CONDITIONAL arm packages under ../conditional/.

Every arm here is prepared and none may be submitted before its named
adjudication. Three independent mechanisms enforce that, because one is a
comment and comments do not stop anything:

  1. They live under ../conditional/, which no loop in ../RUCHE_RUNBOOK.md
     enumerates and which has its own Human Gate heading in
     ../CONDITIONAL_SUBMISSION.md.
  2. Every job script begins with a HARD INTERLOCK: it exits 3, before it
     touches the sampler, unless a release file named for that exact arm
     exists. Submitting one by accident costs a few seconds of an array of
     no-ops, not core-hours and not a wrong answer.
  3. Their run_preflight.sh FAILS while the interlock is armed, so a
     "preflight everything" sweep reports them as blocked rather than ready.

The release file's content is not checked and cannot be: it is a place for the
researcher to record which adjudication released the arm, and the point of the
interlock is that a human wrote it.

Writes files. Contains no scheduler call and cannot submit.
"""
import os, sys, csv, json, shutil, collections

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
COND = os.path.join(TASK, "conditional")
sys.path.insert(0, HERE)
import design as D
from cost_model import (n_steps, rate_ms, wall_s, mem_mb, mem_request_gb,
                        elapsed_h, slurm_time, PESSIMISTIC)
from build_arms import FIELDS, SHARED

# ---------------------------------------------------------------------------
# The blocking condition of each arm is PRE-REGISTERED here, before any datum
# from the immediate group exists. None of them refers to an observed value:
# a trigger that is written after the number arrives is not a trigger.
# ---------------------------------------------------------------------------
ARMS = [
    dict(name="cond_D2_L128_nc4096", tag="XD4096", group="D2",
         L=128, T=128.0, cells=[(0.3032, 6.0, 4096)], R=8,
         gate="CAMPAIGN D ADJUDICATION",
         trigger="Recommend this arm if, on the L = 128 ladder completed by "
                 "campaign D, EITHER |Delta_1024| = |I_2048 - I_1024| is "
                 "resolved OUTSIDE the frozen material tolerance tau_I = "
                 "0.006 (i.e. the 95 % interval excludes [-tau_I, +tau_I]), "
                 "OR no plateau criterion P1-P5 of ../SUCCESS_CRITERIA.yaml is "
                 "satisfied at the top of that ladder. Do NOT recommend it "
                 "because the observed Delta_1024 'looks large'.",
         purpose="CONDITIONAL. L = 128, T = 128, lambda = 0.3032, N_c = 4096, "
                 "R = 8. One further rung on the hardest ladder in the "
                 "programme. Read the wall-time line before releasing this: a "
                 "single population is a multi-day job."),
    dict(name="cond_M96_nc1024", tag="XM96A", group="M96",
         L=96, T=96.0, cells=[(l, 6.0, 1024) for l in D.MOCK9_GRID],
         R=D.MOCK96_R_STAGE1,
         gate="CAMPAIGN C ADJUDICATION -- AND ONLY ONE OF THE TWO M96 ARMS",
         trigger="Release ONLY if campaign C identifies N_c = 1024 as the "
                 "smallest N_c meeting the frozen production adequacy "
                 "criterion at L = 96. If it identifies 2048, release "
                 "cond_M96_nc2048 INSTEAD. Never both: they are the same "
                 "physical scan at two population sizes and running both is "
                 "duplicated compute, not a robustness check.",
         purpose="CONDITIONAL, STAGE 1. L = 96 mock-production scan over the "
                 "frozen 9-point grid at N_c = 1024, R = 12."),
    dict(name="cond_M96_nc2048", tag="XM96B", group="M96",
         L=96, T=96.0, cells=[(l, 6.0, 2048) for l in D.MOCK9_GRID],
         R=D.MOCK96_R_STAGE1,
         gate="CAMPAIGN C ADJUDICATION -- AND ONLY ONE OF THE TWO M96 ARMS",
         trigger="Release ONLY if campaign C identifies N_c = 2048 as the "
                 "smallest N_c meeting the frozen production adequacy "
                 "criterion at L = 96, or if it identifies none and the "
                 "researcher accepts a scan at the largest calibrated rung. "
                 "Never together with cond_M96_nc1024.",
         purpose="CONDITIONAL, STAGE 1. L = 96 mock-production scan over the "
                 "frozen 9-point grid at N_c = 2048, R = 12."),
    dict(name="cond_M128_nc2048", tag="XM128A", group="M128",
         L=128, T=128.0, cells=[(l, 6.0, 2048) for l in D.MOCK9_GRID],
         R=D.MOCK128_R_STAGE1,
         gate="CAMPAIGN D ADJUDICATION -- STRONGLY GATED",
         trigger="Release ONLY if campaign D's N_c = 2048 rung PASSES the "
                 "frozen adequacy screen at L = 128. If it fails, the "
                 "conditional N_c = 4096 central rung comes first and this "
                 "arm stays blocked. An adequate N_c must be identified "
                 "BEFORE a 9-point scan at this L is run at all.",
         purpose="CONDITIONAL, STAGE 1. L = 128 mock-production scan over the "
                 "frozen 9-point grid at N_c = 2048, R = 8."),
    dict(name="cond_M128_nc4096", tag="XM128B", group="M128",
         L=128, T=128.0, cells=[(l, 6.0, 4096) for l in D.MOCK9_GRID],
         R=D.MOCK128_R_STAGE1,
         gate="CAMPAIGN D AND cond_D2_L128_nc4096 ADJUDICATION",
         trigger="Release ONLY if N_c = 2048 FAILS the adequacy screen at "
                 "L = 128 and the conditional N_c = 4096 central rung then "
                 "PASSES it. Read the core-hour line before releasing: this is "
                 "the most expensive object in the whole campaign by a wide "
                 "margin and it should not be the first way the programme "
                 "learns that L = 128 is unaffordable.",
         purpose="CONDITIONAL, STAGE 1. L = 128 mock-production scan over the "
                 "frozen 9-point grid at N_c = 4096, R = 8."),
    dict(name="cond_LOWZ_nc64", tag="XZ64", group="LOWZ",
         L=D.LOWZ_L, T=float(D.LOWZ_L), cells=[(D.LOWZ_LAM, 6.0, 64)],
         R=D.LOWZ_R, zeta=D.LOWZ_ZETA,
         gate="OPTIONAL -- NOT PART OF THE zeta = 0.35 CALIBRATION",
         trigger="Design 2 of TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING. It "
                 "is deliberately NOT in the immediate group: the programme "
                 "wants the zeta = 0.35 calibration understood before it "
                 "spends anything on a second zeta. Release only as an "
                 "explicit decision to buy that one test now.",
         purpose="OPTIONAL. L = 64, T = 64, zeta = 0.10, lambda = 0.3032, "
                 "N_c = 64, R = 48. 'Matched lambda' is read as THE SAME "
                 "lambda, not the same offset from a putative lambda_c: "
                 "matching on a critical-law offset would import the law "
                 "under test."),
    dict(name="cond_LOWZ_nc256", tag="XZ256", group="LOWZ",
         L=D.LOWZ_L, T=float(D.LOWZ_L), cells=[(D.LOWZ_LAM, 6.0, 256)],
         R=D.LOWZ_R, zeta=D.LOWZ_ZETA,
         gate="OPTIONAL -- NOT PART OF THE zeta = 0.35 CALIBRATION",
         trigger="As cond_LOWZ_nc64. The pre-registered kill criterion needs "
                 "BOTH population sizes: drift at zeta = 0.10 greater than or "
                 "equal to drift at zeta = 0.35 kills the guided-residual "
                 "mechanism and revives Born-rarity reasoning. Release both "
                 "or neither.",
         purpose="OPTIONAL. L = 64, T = 64, zeta = 0.10, lambda = 0.3032, "
                 "N_c = 256, R = 48."),
]
for i, a in enumerate(ARMS):
    a["seed_base"] = 33_500_000 + 20_000 * i
    a.setdefault("zeta", D.ZETA)


def build(a):
    rows = []
    for ci, (lam, dm, N) in enumerate(a["cells"]):
        for ri in range(a["R"]):
            s = a["seed_base"] + 1000 * ci + ri
            assert D.SEED_FLOOR <= s < D.SEED_CEIL
            rows.append({"arm": a["tag"], "L": a["L"], "T": a["T"], "N_c": N,
                         "zeta": a["zeta"], "lam": lam, "dtau_mult": dm,
                         "resample_scheme": D.SCHEME, "seed": s})
    return rows


def cost(a, rows):
    by = collections.Counter((r["N_c"], r["lam"], r["dtau_mult"]) for r in rows)
    ws = {k: wall_s(a["L"], a["T"], k[1], k[0], k[2]) for k in by}
    ns = {k: n_steps(a["L"], a["T"], k[1], k[2]) for k in by}
    slow_h = max(ws.values()) / 3600.0
    core_h = sum(ws[k] * n for k, n in by.items()) / 3600.0
    t = slurm_time(slow_h * PESSIMISTIC)
    hours = int(t[:2])
    return dict(slow_h=slow_h, core_h=core_h,
                mem_mb=max(mem_mb(a["L"], k[0], ns[k]) for k in by),
                mem_gb=max(mem_request_gb(a["L"], k[0], ns[k], 1.35) for k in by),
                elapsed_h=elapsed_h(len(rows), core_h, slow_h, D.CONCURRENCY),
                time=t,
                partition="cpu_med" if hours <= 4 else "cpu_long",
                rates=sorted({round(rate_ms(a["L"], k[0]), 3) for k in by}),
                n_steps=sorted({v for v in ns.values()}))


INTERLOCK = """
# ---------------------------------------------------------------------------
# HARD INTERLOCK. THIS ARM IS BLOCKED.
#
#   {gate}
#
# {trigger}
#
# Every array task exits 3 immediately, before importing the sampler, unless
# the researcher has created the release file below by hand. Submitting this
# arm before the adjudication therefore costs a few seconds of no-ops rather
# than {core_h:.0f} core-hours and a result nobody is entitled to interpret.
#
#   touch ../GATE_RELEASED_{name}
#   # and write into it which adjudication released it, and when
# ---------------------------------------------------------------------------
RELEASE="$SLURM_SUBMIT_DIR/../GATE_RELEASED_{name}"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED: {name} has not been released." >&2
    echo "  Gate: {gate}" >&2
    echo "  Create $RELEASE only after that adjudication." >&2
    echo "  See ../CONDITIONAL_SUBMISSION.md." >&2
    exit 3
fi
echo "[{tag}] released by: $(head -c 400 "$RELEASE")"
"""


def cellblock(a, rows):
    by = collections.Counter((r["N_c"], r["lam"], r["dtau_mult"]) for r in rows)
    return "\n".join(
        "#     N_c=%-5d lambda=%-7g dtau_mult=%-4g K=%-5d rows=%d"
        % (N, lam, dm, n_steps(a["L"], a["T"], lam, dm), by[(N, lam, dm)])
        for (N, lam, dm) in sorted(by))


def slurm(a, rows, c):
    n = len(rows)
    tag = a["tag"].lower()
    return f"""#!/bin/bash
#SBATCH --job-name=BLOCKED-{tag}
#SBATCH --partition={c['partition']}
#SBATCH --array=0-{n - 1}%{D.CONCURRENCY}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={c['mem_gb']}G
#SBATCH --time={c['time']}
#SBATCH --output=logs/{tag}_%A_%a.out
#SBATCH --error=logs/{tag}_%A_%a.err
#
# ============================================================================
# {a['name']} -- TASK-2026-09-03-NC-PLATEAU-CALIBRATION
#
#            *** CONDITIONAL -- DO NOT SUBMIT ***
#            *** {a['gate']}
#
# NOT SUBMITTED BY ANY AGENT, EVER. research/RESOURCE_POLICY.md section 4.
# And not by the researcher either, until the gate above is adjudicated.
#
# PURPOSE
#   {a['purpose']}
#
# CELLS
#   L = {a['L']}, T = {a['T']:g}, zeta = {a['zeta']}, {D.SCHEME} resampling, R = {a['R']}
{cellblock(a, rows)}
#
# COST -- measured Ruche rates {c['rates']} ms per clone-window
#   {n} tasks, ~{c['core_h']:.0f} core-hours ({c['core_h'] * PESSIMISTIC:.0f} pessimistic).
#   slowest single task ~{c['slow_h']:.2f} h ({c['slow_h'] * PESSIMISTIC:.2f} h pessimistic).
#   elapsed ~{c['elapsed_h']:.2f} h at the cap below, queue wait EXCLUDED.
# ============================================================================

set -euo pipefail
{INTERLOCK.format(gate=a['gate'], trigger=a['trigger'], name=a['name'],
                  tag=tag, core_h=c['core_h'])}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

PPSQJ_PYTHON="${{PPSQJ_PYTHON:-/gpfs/workdir/ercetinut/envs/pps_qj/bin/python}}"
if [ ! -x "$PPSQJ_PYTHON" ]; then
    echo "PPSQJ_PYTHON is not executable: $PPSQJ_PYTHON" >&2
    exit 2
fi
export PATH="$(dirname "$PPSQJ_PYTHON"):$PATH"

echo "[{tag}] task ${{SLURM_ARRAY_TASK_ID}} on $(hostname) at $(date -u +%FT%TZ)"
"$PPSQJ_PYTHON" run_cell.py "${{SLURM_ARRAY_TASK_ID}}" "${{SLURM_SUBMIT_DIR}}/results"
echo "[{tag}] task ${{SLURM_ARRAY_TASK_ID}} done at $(date -u +%FT%TZ)"
"""


PREFLIGHT_SH = """#!/bin/bash
# Preflight for a BLOCKED conditional arm.
#
# It refuses while the interlock is armed. That is the point: a
# "preflight everything" sweep must report this arm as BLOCKED, not READY.
# There is no scheduler-submission command in this file.
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
RELEASE="../GATE_RELEASED_{name}"
if [ ! -f "$RELEASE" ]; then
    echo "BLOCKED  {name}"
    echo "  gate: {gate}"
    echo "  {trigger}"
    echo
    echo "  This arm is prepared and validated but is NOT ready for submission."
    echo "  See ../CONDITIONAL_SUBMISSION.md."
    exit 3
fi
PY=${{PYTHON:-python3}}
mkdir -p results
exec "$PY" preflight.py
"""


def main():
    os.makedirs(COND, exist_ok=True)
    seeds, summary = set(), []
    for a in ARMS:
        rows = build(a)
        assert not (seeds & {r["seed"] for r in rows}), "conditional seed collision"
        seeds |= {r["seed"] for r in rows}
        c = cost(a, rows)
        d = os.path.join(COND, a["name"])
        os.makedirs(os.path.join(d, "results"), exist_ok=True)
        os.makedirs(os.path.join(d, "logs"), exist_ok=True)
        open(os.path.join(d, "results", ".gitkeep"), "w").close()
        open(os.path.join(d, "logs", ".gitkeep"), "w").close()
        with open(os.path.join(d, "manifest.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        # run_cell/analyse are shared and unchanged; the preflight is NOT (it
        # would report a blocked arm as ready), so the conditional arms get the
        # refusing wrapper instead.
        for f in ("run_cell.py", "analyse_arm.py", "analyse_results.sh"):
            shutil.copy2(os.path.join(TASK, "shared", f), os.path.join(d, f))
            if f.endswith(".sh"):
                os.chmod(os.path.join(d, f), 0o755)
        p = os.path.join(d, "run_preflight.sh")
        open(p, "w").write(PREFLIGHT_SH.format(
            name=a["name"], gate=a["gate"],
            trigger=a["trigger"].replace('"', "'")))
        os.chmod(p, 0o755)
        open(os.path.join(d, "submit.slurm"), "w").write(slurm(a, rows, c))
        open(os.path.join(d, "README.md"), "w").write(
            f"# {a['name']} — **CONDITIONAL, BLOCKED**\n\n"
            f"> **{a['gate']}**\n>\n> {a['trigger']}\n\n"
            f"{a['purpose']}\n\n"
            f"| | |\n|---|---|\n"
            f"| L, T | {a['L']}, {a['T']:g} |\n| zeta | {a['zeta']} |\n"
            f"| lambda | {', '.join(f'{c0[0]:g}' for c0 in a['cells'])} |\n"
            f"| N_c | {sorted({c0[2] for c0 in a['cells']})} |\n"
            f"| R | {a['R']} |\n| tasks | {len(rows)} |\n"
            f"| seeds | {min(r['seed'] for r in rows)}–"
            f"{max(r['seed'] for r in rows)} |\n"
            f"| core-hours | {c['core_h']:.0f} "
            f"({c['core_h'] * PESSIMISTIC:.0f} pessimistic) |\n"
            f"| slowest task | {c['slow_h']:.2f} h "
            f"({c['slow_h'] * PESSIMISTIC:.2f} h pessimistic) |\n"
            f"| partition / time / mem | {c['partition']} / {c['time']} / "
            f"{c['mem_gb']}G |\n\n"
            f"`run_preflight.sh` exits 3 while blocked, and every array task "
            f"exits 3 before touching the sampler unless "
            f"`../GATE_RELEASED_{a['name']}` exists. **No agent submits this, "
            f"and neither should you until the gate above is adjudicated.**\n")
        summary.append(dict(name=a["name"], group=a["group"], gate=a["gate"],
                            trigger=a["trigger"], L=a["L"], zeta=a["zeta"],
                            R=a["R"], tasks=len(rows),
                            lambdas=sorted({c0[0] for c0 in a["cells"]}),
                            N_c=sorted({c0[2] for c0 in a["cells"]}),
                            core_h=round(c["core_h"], 1),
                            pess_core_h=round(c["core_h"] * PESSIMISTIC, 1),
                            slowest_h=round(c["slow_h"], 2),
                            pess_slowest_h=round(c["slow_h"] * PESSIMISTIC, 2),
                            elapsed_h=round(c["elapsed_h"], 2),
                            partition=c["partition"], time=c["time"],
                            mem_req=f"{c['mem_gb']}G",
                            seed_lo=min(r["seed"] for r in rows),
                            seed_hi=max(r["seed"] for r in rows)))
        print(f"{a['name']:<22} {len(rows):>4} rows  slow {c['slow_h']:7.2f} h  "
              f"{c['core_h']:9.1f} core-h  {c['partition']:<8} {c['time']}  "
              f"{c['mem_gb']:>3}G   BLOCKED")
    json.dump(summary, open(os.path.join(HERE, "conditional_summary.json"), "w"),
              indent=1)
    imm = json.load(open(os.path.join(HERE, "allocated_seeds.json")))
    assert not (set(imm) & seeds), "conditional seeds overlap the immediate group"
    print(f"\n{len(seeds)} conditional seeds, disjoint from the "
          f"{len(imm)} immediate ones")
    print(f"{sum(s['tasks'] for s in summary)} tasks, "
          f"{sum(s['core_h'] for s in summary):.0f} core-hours if EVERY "
          f"conditional arm were released (they will not all be)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
