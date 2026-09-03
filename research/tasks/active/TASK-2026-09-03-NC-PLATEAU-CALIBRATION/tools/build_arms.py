#!/usr/bin/env python3
"""Generate every arm package for TASK-2026-09-03-NC-PLATEAU-CALIBRATION.

The design lives in tools/design.py; the costs in tools/cost_model.py; the
reuse decisions in EXISTING_POPULATION_INVENTORY.csv. This file only assembles.
Editing a manifest, a submit script or an arm README by hand is an error --
regenerate here and re-run every check in ../VALIDATION.md.

Writes files. Contains no scheduler call and cannot submit.
"""
import os, sys, csv, json, math, shutil, collections

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
import design as D
from cost_model import (n_steps, rate_ms, wall_s, mem_mb, mem_request_gb,
                        elapsed_h, slurm_time, PESSIMISTIC, PACKING)

FIELDS = ["arm", "L", "T", "N_c", "zeta", "lam", "dtau_mult",
          "resample_scheme", "seed"]
SHARED = ("run_cell.py", "preflight.py", "run_preflight.sh",
          "analyse_arm.py", "analyse_results.sh")

# ---------------------------------------------------------------------------
# REUSE. Cells that already hold exact-compatible production populations, with
# the R they hold. A cell listed here is NEVER recomputed; where this task wants
# more replicates it TOPS UP from replicate index R_existing.
#   (L, T, N_c, lam, dtau_mult) -> (R_existing, source)
# Verified against the raw result JSONs by tools/dedup_scan.py, which fails if
# any of these disagrees with what is actually on disk.
# ---------------------------------------------------------------------------
REUSE = {
    (64, 64.0, 1024, 0.3032, 6.0): (96, "TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/armB"),
    (64, 64.0, 2048, 0.3032, 6.0): (24, "TASK-2026-09-02-MOCK-PRODUCTION/mockL64nc2048"),
    (64, 64.0, 1024, 0.2232, 6.0): (24, "TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/lowlamL64"),
    (64, 64.0, 1024, 0.2332, 6.0): (24, "TASK-2026-09-02-MOCK-PRODUCTION/mockL64"),
    (64, 64.0, 1024, 0.2432, 6.0): (24, "TASK-2026-09-02-MOCK-PRODUCTION/mockL64"),
    (32, 32.0, 1024, 0.2232, 6.0): (24, "TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/lowlamL32"),
    (32, 32.0, 1024, 0.2332, 6.0): (24, "TASK-2026-09-02-MOCK-PRODUCTION/mockL32"),
    (32, 32.0, 1024, 0.2432, 6.0): (24, "TASK-2026-09-02-MOCK-PRODUCTION/mockL32"),
    (48, 48.0, 1024, 0.2232, 6.0): (24, "TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/lowlamL48"),
    (48, 48.0, 1024, 0.2332, 6.0): (24, "TASK-2026-09-02-MOCK-PRODUCTION/mockL48"),
    (48, 48.0, 1024, 0.2432, 6.0): (24, "TASK-2026-09-02-MOCK-PRODUCTION/mockL48"),
    (96, 96.0, 128, 0.3032, 6.0): (32, "TASK-2026-09-01-SMCRUCHE-READY/arm1"),
    (96, 96.0, 256, 0.3032, 6.0): (32, "TASK-2026-09-01-SMCRUCHE-READY/arm1"),
    (96, 96.0, 512, 0.3032, 6.0): (48, "TASK-2026-09-01-SMCRUCHE-READY/arm1"),
    (128, 128.0, 64, 0.3032, 6.0): (64, "TASK-2026-09-01-SMCRUCHE-READY/arm2"),
    (128, 128.0, 128, 0.3032, 6.0): (64, "TASK-2026-09-01-SMCRUCHE-READY/arm2"),
    (128, 128.0, 256, 0.3032, 6.0): (64, "TASK-2026-09-01-SMCRUCHE-READY/arm2"),
    (128, 128.0, 512, 0.3032, 6.0): (48, "TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/armA512"),
    (128, 128.0, 1024, 0.3032, 6.0): (32, "TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/armA1024"),
}


def cells_A():
    """Campaign A -- deep central N_c ladder at L = 64. One cell per arm."""
    out = []
    out.append(dict(name="A_L64_nc2048_topup", tag="A2048", group="A",
                    L=64, T=64.0, N_c=2048, cells=[(0.3032, 6.0)], R=48,
                    purpose="CAMPAIGN A top-up. L = 64, lambda = 0.3032, "
                            "N_c = 2048: 24 exact-compatible populations already "
                            "exist, and 24 more are added here to bring the rung "
                            "to the R = 48 the frozen tau_I demands. The "
                            "existing 24 are NOT recomputed."))
    for N in (4096, 8192):
        out.append(dict(name=f"A_L64_nc{N}", tag=f"A{N}", group="A",
                        L=64, T=64.0, N_c=N, cells=[(0.3032, 6.0)], R=48,
                        purpose=f"CAMPAIGN A. L = 64, T = 64, lambda = 0.3032, "
                                f"N_c = {N}, R = 48. New top rung of the deep "
                                f"central ladder: the rung that decides whether "
                                f"a high-N_c plateau is OBSERVED rather than "
                                f"inferred by eye."))
    return out


def cells_B():
    out = []
    for N in D.B_NCS:
        out.append(dict(name=f"B_L64_cross_nc{N}", tag=f"B{N}", group="B",
                        L=64, T=64.0, N_c=N,
                        cells=[(l, 6.0) for l in D.B_GRID], R=D.B_R,
                        purpose=f"CAMPAIGN B. L = 64, T = 64, the frozen 7-point "
                                f"transition-region grid 0.2182-0.2482 at "
                                f"N_c = {N}, matched R = {D.B_R}. Tests whether "
                                f"finite-N_c distorts the SHAPE of CMI(lambda) "
                                f"where the low-L locator sits, not just its level."))
    return out


def cells_B2():
    out = []
    for L in D.B2_LS:
        for N in D.B2_NCS:
            out.append(dict(name=f"B2_L{L}_nc{N}", tag=f"Y{L}N{N}", group="B2",
                            L=L, T=float(L), N_c=N,
                            cells=[(l, 6.0) for l in D.B2_GRID], R=D.B2_R,
                            purpose=f"CAMPAIGN B2. L = {L}, T = {L}, the same "
                                    f"frozen 7-point grid as campaign B, at "
                                    f"N_c = {N}, matched R = {D.B2_R}. Puts the "
                                    f"low-L reference curve on the SAME grid at "
                                    f"the SAME N_c as L = 64, so the locator "
                                    f"test of section 4B is a FULLY MATCHED "
                                    f"cross-L comparison and not a one-sided "
                                    f"diagnostic. On a 3-lambda grid the frozen "
                                    f"crossing protocol flags every interior "
                                    f"crossing ENDPOINT_INDUCED by construction; "
                                    f"on 7 points both have a guard point on "
                                    f"each side. The three lambdas that already "
                                    f"exist at N_c = 1024 are TOPPED UP, never "
                                    f"recomputed."))
    return out


def cells_C():
    return [dict(name=f"C_L96_nc{N}", tag=f"C{N}", group="C",
                 L=96, T=96.0, N_c=N, cells=[(0.3032, 6.0)], R=D.C_R,
                 purpose=f"CAMPAIGN C. L = 96, T = 96, lambda = 0.3032, "
                         f"N_c = {N}, R = {D.C_R}. Fills the L = 64 / L = 128 gap. "
                         f"The existing L = 96 ladder (N_c = 128, 256, 512) "
                         f"REJECTS a clean I = I_inf + B/N over its measured "
                         f"range; these rungs test whether it enters a simpler "
                         f"high-N regime. They do not assume it does.")
            for N in D.C_NCS]


def cells_D():
    return [dict(name="D_L128_nc2048", tag="D2048", group="D",
                 L=128, T=128.0, N_c=2048, cells=[(0.3032, 6.0)], R=D.D_R,
                 purpose="CAMPAIGN D. L = 128, T = 128, lambda = 0.3032, "
                         "N_c = 2048, R = 16. A SCREENING rung: R = 16 resolves "
                         "a shift of the size the 512 -> 1024 step showed "
                         "(-0.0602 +- 0.0234) and CANNOT certify convergence. "
                         "That asymmetry is pre-registered in ../SUCCESS_CRITERIA.yaml.")]


def cells_E():
    return [dict(name=f"E_L64_dtau_nc{N}", tag=f"E{N}", group="E",
                 L=64, T=64.0, N_c=N,
                 cells=[(0.3032, dm) for dm in D.E_DTAUS], R=D.E_R,
                 purpose=f"CAMPAIGN E. L = 64, T = 64, lambda = 0.3032, "
                         f"N_c = {N}, dtau_mult in {{3, 6, 12}} giving "
                         f"K = 816 / 408 / 204, matched R = {D.E_R}. The "
                         f"Feynman-Kac weight is exact at any window size, so "
                         f"the TARGET MEASURE IS EXACTLY UNCHANGED across the "
                         f"three sub-cells; only where selection is applied "
                         f"moves. dtau_mult is a discretisation control and "
                         f"never a physical parameter.")
            for N in D.E_NCS]


ARMS = cells_A() + cells_B() + cells_B2() + cells_C() + cells_D() + cells_E()
for i, a in enumerate(ARMS):
    a["seed_base"] = D.SEED_FLOOR + 20_000 * i


def norm_cells(a):
    """(lam, dtau_mult, N_c) triples for this arm, in frozen order."""
    out = []
    for c in a["cells"]:
        out.append((c[0], c[1], c[2] if len(c) > 2 else a["N_c"]))
    return out


def build_rows(a, seeds_all):
    rows = []
    for ci, (lam, dm, N) in enumerate(norm_cells(a)):
        key = (a["L"], a["T"], N, round(lam, 6), dm)
        r_have = REUSE.get(key, (0, None))[0]
        # top up from the existing replicate index, never from 0: a fresh
        # population must not carry a replicate label an existing one holds.
        for ri in range(r_have, max(a["R"], r_have)):
            s = a["seed_base"] + 1000 * ci + ri
            assert D.SEED_FLOOR <= s < D.SEED_CEIL, f"seed {s} outside the block"
            assert s not in seeds_all, f"seed collision {s}"
            seeds_all[s] = a["name"]
            rows.append({"arm": a["tag"], "L": a["L"], "T": a["T"], "N_c": N,
                         "zeta": D.ZETA, "lam": lam, "dtau_mult": dm,
                         "resample_scheme": D.SCHEME, "seed": s})
    return rows


def arm_cost(a, rows):
    # rows may come from the builder (typed) or from csv.DictReader (all str).
    # Coerce here, once, rather than trusting the caller: the preflight passes
    # the CSV and an earlier version silently compared a str N_c against an int.
    by = collections.Counter((int(r["N_c"]), float(r["lam"]),
                              float(r["dtau_mult"])) for r in rows)
    ws = {k: wall_s(a["L"], a["T"], k[1], k[0], k[2]) for k in by}
    slow_h = max(ws.values()) / 3600.0
    core_h = sum(ws[k] * n for k, n in by.items()) / 3600.0
    ns = {k: n_steps(a["L"], a["T"], k[1], k[2]) for k in by}
    mem = max(mem_mb(a["L"], k[0], ns[k]) for k in by)
    gb = max(mem_request_gb(a["L"], k[0], ns[k], margin=1.35) for k in by)
    el = elapsed_h(len(rows), core_h, slow_h, D.CONCURRENCY)
    t = slurm_time(slow_h * PESSIMISTIC)
    hours = int(t[:2])
    part = "cpu_med" if hours <= D.PARTITION_MAXTIME_H["cpu_med"] else "cpu_long"
    return dict(slow_h=slow_h, core_h=core_h, mem_mb=mem, mem_gb=gb,
                elapsed_h=el, time=t, partition=part,
                n_steps=sorted({v for v in ns.values()}),
                rates=sorted({round(rate_ms(a["L"], k[0]), 3) for k in by}))


def render(path, text):
    open(path, "w").write(text)


def main():
    seeds_all, summary = {}, []
    for a in ARMS:
        rows = build_rows(a, seeds_all)
        c = arm_cost(a, rows)
        d = os.path.join(TASK, a["name"])
        os.makedirs(os.path.join(d, "results"), exist_ok=True)
        os.makedirs(os.path.join(d, "logs"), exist_ok=True)
        render(os.path.join(d, "results", ".gitkeep"), "")
        render(os.path.join(d, "logs", ".gitkeep"), "")
        with open(os.path.join(d, "manifest.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        for f in SHARED:
            shutil.copy2(os.path.join(TASK, "shared", f), os.path.join(d, f))
            if f.endswith(".sh"):
                os.chmod(os.path.join(d, f), 0o755)
        render(os.path.join(d, "submit.slurm"), slurm(a, rows, c))
        render(os.path.join(d, "README.md"), readme(a, rows, c))

        reused = sum(REUSE.get((a["L"], a["T"], N, round(l, 6), dm), (0, None))[0]
                     for l, dm, N in norm_cells(a))
        summary.append(dict(
            name=a["name"], tag=a["tag"], group=a["group"], L=a["L"], T=a["T"],
            R=a["R"], cells=[[l, dm, N] for l, dm, N in norm_cells(a)],
            tasks=len(rows), reused_populations=reused,
            seed_lo=min(r["seed"] for r in rows),
            seed_hi=max(r["seed"] for r in rows),
            core_h=round(c["core_h"], 2),
            pess_core_h=round(c["core_h"] * PESSIMISTIC, 2),
            slowest_h=round(c["slow_h"], 3),
            pess_slowest_h=round(c["slow_h"] * PESSIMISTIC, 3),
            elapsed_h=round(c["elapsed_h"], 3),
            pess_elapsed_h=round(c["elapsed_h"] * PESSIMISTIC, 3),
            mem_mb=round(c["mem_mb"]), mem_req=f"{c['mem_gb']}G",
            partition=c["partition"], time=c["time"],
            n_steps=c["n_steps"], rate_ms=c["rates"]))
        print(f"{a['name']:<22} {len(rows):>4} rows  reuse {reused:>3}  "
              f"slow {c['slow_h']:6.2f} h  {c['core_h']:8.2f} core-h  "
              f"{c['partition']:<8} {c['time']}  {c['mem_gb']:>2}G  "
              f"seeds {min(r['seed'] for r in rows)}-{max(r['seed'] for r in rows)}")

    json.dump(sorted(seeds_all), open(os.path.join(HERE, "allocated_seeds.json"), "w"))
    json.dump(dict(arms=summary, seed_floor=D.SEED_FLOOR, seed_ceil=D.SEED_CEIL,
                   concurrency=D.CONCURRENCY, pessimistic=PESSIMISTIC,
                   tau_I=D.TAU_I, tau_lambda=D.TAU_LAMBDA, tau_D=D.TAU_D),
              open(os.path.join(HERE, "cost_summary.json"), "w"), indent=1)

    tot = sum(s["core_h"] for s in summary)
    print(f"\n{len(seeds_all)} seeds, all distinct, "
          f"{min(seeds_all)}-{max(seeds_all)}")
    print(f"{sum(s['tasks'] for s in summary)} fresh tasks, "
          f"{sum(s['reused_populations'] for s in summary)} populations reused")
    print(f"{tot:.1f} core-hours ({tot * PESSIMISTIC:.1f} pessimistic)")
    print(f"longest single task {max(s['slowest_h'] for s in summary):.2f} h "
          f"({max(s['pess_slowest_h'] for s in summary):.2f} h pessimistic)")
    return 0


def _cellblock(a, rows):
    by = collections.Counter((r["N_c"], r["lam"], r["dtau_mult"]) for r in rows)
    out = []
    for (N, lam, dm) in sorted(by):
        k = (a["L"], a["T"], N, round(lam, 6), dm)
        rr = REUSE.get(k, (0, None))
        out.append("#     N_c=%-5d lambda=%-7g dtau_mult=%-4g K=%-5d fresh=%-3d %s"
                   % (N, lam, dm, n_steps(a["L"], a["T"], lam, dm), by[(N, lam, dm)],
                      ("reuse=%d from %s" % rr) if rr[0] else "reuse=0"))
    return "\n".join(out)


def slurm(a, rows, c):
    n = len(rows)
    tag = a["tag"].lower()
    return f"""#!/bin/bash
#SBATCH --job-name=ncplat-{tag}
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
# {a['name']} -- TASK-2026-09-03-NC-PLATEAU-CALIBRATION  (campaign {a['group']})
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage, gate and approval level. The researcher types
# the submission command by hand. Neither this file nor the preflight contains
# one.
#
# PURPOSE
#   {a['purpose']}
#
# CELLS (frozen; do not hand-edit -- regenerate with tools/build_arms.py)
#   L = {a['L']}, T = {a['T']:g}, zeta = {D.ZETA}, {D.SCHEME} resampling
{_cellblock(a, rows)}
#   Populations listed as `reuse` ALREADY EXIST as exact-compatible completed
#   runs and are absent from this manifest on purpose. Recomputing them would
#   buy nothing. See ../REUSE_LEDGER.csv.
#
# COST -- from per-clone-window rates MEASURED on Ruche on this identical code
# path, never from a requested --time and never from a laptop probe.
#   {n} array tasks, ~{c['core_h']:.1f} core-hours
#   ({c['core_h'] * PESSIMISTIC:.1f} pessimistic).
#   slowest single task ~{c['slow_h']:.2f} h ({c['slow_h'] * PESSIMISTIC:.2f} h pessimistic).
#   elapsed ~{c['elapsed_h']:.2f} h at the cap below
#   ({c['elapsed_h'] * PESSIMISTIC:.2f} h pessimistic), EXCLUDING queue wait.
#   adopted rate(s) {c['rates']} ms per clone-window.
#   Provenance and the model's own failure modes: ../COST_MODEL.md.
#
# --time={c['time']} is >= 1.6x the PESSIMISTIC slowest task. It was chosen from
#   the cost model and the partition was then chosen to fit it, never the
#   other way round.
#
# --mem={c['mem_gb']}G is 1.35x a MEASURED peak-RSS model ({c['mem_mb']:.0f} MB).
#   The formula every predecessor package used (128 + 2*N_c*per_clone) is
#   UNDER-conservative by about a factor two -- direct ru_maxrss measurements
#   are in ../COST_MODEL.md section "Memory". At the N_c this campaign reaches
#   that stops being a rounding error.
#
# PARTITION -- {c['partition']}. cpu_med when --time fits its 4 h MaxTime,
#   cpu_long otherwise. cpu_short is never used at any --time: it is
#   effectively serialised for this account by QOSMaxJobsPerUserLimit.
#   preflight.py recomputes this rule and EXITS NONZERO on a mismatch.
#
# --array=0-{n - 1}  : ONE task per manifest row, exactly. Do not change the range;
#              preflight.py fails if it stops matching the manifest.
#   %{D.CONCURRENCY}      : a concurrency cap only, not a grant. If the allocation gives
#              64 slots in TOTAL rather than per array, lower it -- the %N cap
#              is the ONLY number in this file that is safe to hand-edit.
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


def readme(a, rows, c):
    by = collections.Counter((r["N_c"], r["lam"], r["dtau_mult"]) for r in rows)
    tbl = ["| N_c | lambda | dtau_mult | K | fresh here | already exist | source of the existing |",
           "|---:|---:|---:|---:|---:|---:|---|"]
    for (N, lam, dm) in sorted(by):
        rr = REUSE.get((a["L"], a["T"], N, round(lam, 6), dm), (0, "—"))
        tbl.append(f"| {N} | {lam:g} | {dm:g} | {n_steps(a['L'], a['T'], lam, dm)} | "
                   f"{by[(N, lam, dm)]} | {rr[0]} | {rr[1] if rr[0] else '—'} |")
    return f"""# {a['name']} — TASK-2026-09-03-NC-PLATEAU-CALIBRATION (campaign {a['group']})

{a['purpose']}

{chr(10).join(tbl)}

| | |
|---|---|
| zeta | {D.ZETA} |
| T | {a['T']:g} (T = L) |
| resampling | {D.SCHEME} |
| target R per cell | {a['R']} |
| array tasks | {len(rows)} |
| seeds | {min(r['seed'] for r in rows)}–{max(r['seed'] for r in rows)}, fresh and structurally disjoint (`../SEED_LEDGER.md`) |
| adopted rate | {c['rates']} ms per clone-window, measured on Ruche |
| slowest task | {c['slow_h']:.2f} h predicted, {c['slow_h'] * PESSIMISTIC:.2f} h pessimistic |
| core-hours | {c['core_h']:.1f} predicted, {c['core_h'] * PESSIMISTIC:.1f} pessimistic |
| elapsed at cap %{D.CONCURRENCY} | {c['elapsed_h']:.2f} h predicted, {c['elapsed_h'] * PESSIMISTIC:.2f} h pessimistic, **queue wait excluded** |
| peak memory | {c['mem_mb']:.0f} MB modelled from direct measurement (requesting {c['mem_gb']}G) |
| partition | **{c['partition']}** `--time={c['time']}` |

Populations listed under *already exist* are **not** in `manifest.csv`. They are
exact-compatible completed runs of this identical code path at this identical
cell and are reused as they are.

## Running it

`../RUCHE_RUNBOOK.md` has the exact command sequence. `run_preflight.sh` must
exit 0 first; it submits nothing and contains no scheduler call.

**No agent submits this. You do.**
"""


if __name__ == "__main__":
    sys.exit(main())
