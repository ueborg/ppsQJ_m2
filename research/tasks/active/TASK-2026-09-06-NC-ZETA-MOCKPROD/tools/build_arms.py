#!/usr/bin/env python3
"""Build every arm of TASK-2026-09-06-NC-ZETA-MOCKPROD.

Regenerates manifest.csv, packs.csv, submit.slurm and README.md for each arm,
plus SEED_LEDGER.md and analysis/campaign_cost.json. Deterministic: running it
twice produces byte-identical output. It contains NO scheduler call and cannot
submit anything.

Arm keying is (zeta, N_c), three L inside. Rate and memory depend on (L, N_c)
and cost depends on lambda, so heterogeneity inside an arm is absorbed by the
packer, which sizes packs by predicted seconds. Keying this way puts the single
most expensive rung of the campaign -- zeta = 0.70, N_c = 2048, 42.9 % of the
whole cost -- in exactly one directory, which is what makes it separable as a
conditional arm.
"""
from __future__ import annotations
import csv, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
import cost_model as CM

GRID = {0.10: [0.040, 0.055, 0.070, 0.085, 0.100, 0.115, 0.130, 0.145, 0.160],
        0.20: [0.080, 0.1025, 0.125, 0.1475, 0.170, 0.1925, 0.215, 0.2375, 0.260],
        0.70: [0.250, 0.284, 0.318, 0.351, 0.385, 0.419, 0.453, 0.486, 0.520]}
LS = [32, 48, 64]
NCS = [128, 256, 512, 1024, 2048]
R = 16
SEED_BASE = 37_000_000        # 36 000 000 is reserved by NC-ZETA-STAGE1
SEED_STRIDE = 100_000
PACK_MIN_SEC = 600.0
CONCURRENCY = 64
CONDITIONAL = ("M_z070_nc2048",)

# The discretisation control (brief section 8). The dtau_mult = 6 leg is NOT
# here: it already exists inside M_z010_nc512 at the same R and the same
# lambda, because lambda = 0.100 is an on-grid point.
CONTROL = dict(arm="E_dtau_z010", zeta=0.10, L=64, N_c=512, lam=0.100,
               dtau_mults=[3.0, 12.0], R=R)


def arm_name(z, nc):
    return "M_z%03d_nc%d" % (round(z * 100), nc)


def main_rows(z, nc):
    """Deterministic row order: L outer, lambda inner, then replicate."""
    rows = []
    for L in LS:
        for lam in GRID[z]:
            for _ in range(R):
                rows.append(dict(L=L, T=float(L), N_c=nc, zeta=z, lam=lam,
                                 dtau_mult=6.0, resample_scheme="systematic"))
    return rows


def control_rows():
    rows = []
    for dt in CONTROL["dtau_mults"]:
        for _ in range(CONTROL["R"]):
            rows.append(dict(L=CONTROL["L"], T=float(CONTROL["L"]), N_c=CONTROL["N_c"],
                             zeta=CONTROL["zeta"], lam=CONTROL["lam"],
                             dtau_mult=dt, resample_scheme="systematic"))
    return rows


def pack(rows):
    """Greedy: accumulate consecutive rows until the pack reaches PACK_MIN_SEC.
    A single row longer than that is its own pack. Rows are already ordered so
    that adjacent rows have similar cost, which keeps packs homogeneous."""
    packs, start, cum = [], 0, 0.0
    for i, r in enumerate(rows):
        cum += r["_sec"]
        if cum >= PACK_MIN_SEC or i == len(rows) - 1:
            packs.append((len(packs), start, i - start + 1, cum))
            start, cum = i + 1, 0.0
    return packs


def snap_time(sec):
    """Smallest readable HH:MM:SS at or above sec."""
    for h, m in [(0, 20), (0, 45), (1, 0), (2, 0), (3, 0), (4, 0), (6, 0), (8, 0),
                 (12, 0), (18, 0), (24, 0), (36, 0), (48, 0), (72, 0), (96, 0),
                 (120, 0), (168, 0)]:
        if h * 3600 + m * 60 >= sec:
            return "%02d:%02d:00" % (h, m), h * 3600 + m * 60
    raise ValueError("no --time fits %.0f s; cpu_long MaxTime is 168 h" % sec)


def build(name, rows, index, outdir, purpose, extra_notes):
    for r in rows:
        r["_sec"] = CM.row_seconds(r["L"], r["N_c"], r["lam"], r["zeta"], r["dtau_mult"])
    for j, r in enumerate(rows):
        r["seed"] = SEED_BASE + SEED_STRIDE * index + j
    packs = pack(rows)
    slowest_pack = max(p[3] for p in packs)
    slowest_row = max(r["_sec"] for r in rows)
    core_h = sum(r["_sec"] for r in rows) / 3600.0
    tlimit, tsec = snap_time(1.6 * CM.PESSIMISTIC * slowest_pack)
    part = "cpu_med" if tsec <= 4 * 3600 else "cpu_long"
    maxL, maxNc = max(r["L"] for r in rows), max(r["N_c"] for r in rows)
    memreq = CM.gib_request(maxL, maxNc)
    ntask = len(packs)
    # Elapsed: FIFO list-scheduling of the packs onto CONCURRENCY slots, in
    # manifest order -- which is what a Slurm array with %N actually does. The
    # wave formula ceil(ntask/N) * slowest_pack is badly pessimistic here
    # because pack durations inside an arm vary by an order of magnitude
    # (lambda spans 4x within a zeta), so most waves are nowhere near the
    # slowest pack. The floor below keeps it honest: elapsed can never be less
    # than one slowest pack, and never less than the throughput bound.
    slots = [0.0] * CONCURRENCY
    for p_ in packs:
        i = min(range(CONCURRENCY), key=lambda k: slots[k])
        slots[i] += p_[3]
    elapsed_h = max(max(slots) * CM.PACKING / 3600.0,
                    core_h / CONCURRENCY * CM.PACKING,
                    slowest_pack / 3600.0)

    os.makedirs(os.path.join(outdir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "results"), exist_ok=True)

    with open(os.path.join(outdir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("arm,L,T,N_c,zeta,lam,dtau_mult,resample_scheme,seed".split(","))
        for r in rows:
            w.writerow([name, r["L"], r["T"], r["N_c"], r["zeta"], r["lam"],
                        r["dtau_mult"], r["resample_scheme"], r["seed"]])
    with open(os.path.join(outdir, "packs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pack", "start", "count", "est_sec"])
        for p in packs:
            w.writerow([p[0], p[1], p[2], round(p[3], 1)])

    shared_rel = os.path.relpath(os.path.join(TASK, "shared"), outdir)
    hdr = SLURM.format(
        shared=shared_rel,
        name=name, lname=name.lower(), part=part, last=ntask - 1, conc=CONCURRENCY,
        mem=memreq, time=tlimit, purpose=purpose, extra=extra_notes,
        npop=len(rows), ntask=ntask, core_h=core_h, pess=core_h * CM.PESSIMISTIC,
        slowrow=slowest_row / 60.0, slowpack=slowest_pack / 60.0,
        pslowpack=slowest_pack * CM.PESSIMISTIC / 60.0, elapsed=elapsed_h,
        memmb=CM.mem_mb(maxL, maxNc), packmin=int(PACK_MIN_SEC))
    with open(os.path.join(outdir, "submit.slurm"), "w") as f:
        f.write(hdr)

    return dict(arm=name, dir=os.path.relpath(outdir, TASK), index=index,
                populations=len(rows), array_tasks=ntask, core_h=round(core_h, 2),
                pessimistic_core_h=round(core_h * CM.PESSIMISTIC, 2),
                slowest_row_s=round(slowest_row, 1),
                slowest_pack_s=round(slowest_pack, 1),
                elapsed_h=round(elapsed_h, 2), partition=part, time=tlimit,
                mem=memreq, mem_model_mb=round(CM.mem_mb(maxL, maxNc), 1),
                seed_lo=rows[0]["seed"], seed_hi=rows[-1]["seed"],
                conditional=name in CONDITIONAL)


SLURM = """#!/bin/bash
#SBATCH --job-name=mockprod-{lname}
#SBATCH --partition={part}
#SBATCH --array=0-{last}%{conc}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output=logs/{name}_%A_%a.out
#SBATCH --error=logs/{name}_%A_%a.err
#
# ============================================================================
# {name} -- TASK-2026-09-06-NC-ZETA-MOCKPROD
#
# NOT SUBMITTED BY ANY AGENT. research/RESOURCE_POLICY.md section 4 forbids it
# unconditionally, at every stage, gate and approval level. The researcher types
# the submission command by hand. Neither this file nor the preflight contains
# one.
#
# PURPOSE
#   {purpose}
#
# {extra}
#
# SCALE
#   {npop} populations packed into {ntask} array tasks so that no array task is
#   shorter than {packmin} s. packs.csv records the (start, count) of every pack.
#   run_pack.py invokes the CERTIFIED run_cell.py once per row in a fresh
#   process: packing changes the scheduling and never the computation, and a row
#   produced by a packed task is EXACT-COMPATIBLE with one produced unpacked.
#   The path to it is {shared}, computed per arm rather than hard-coded: the
#   conditional arm lives one directory deeper and a literal ../shared would not
#   resolve there. Preflight P17 checks that it resolves; negative control N14
#   breaks it and requires rejection.
#   Idempotent -- a completed row is never recomputed, so a requeued or
#   timed-out pack costs only what it has left to do.
#
# COST -- fitted to wall_s recorded by completed Ruche jobs of this code path.
#   {core_h:.1f} core-hours ({pess:.1f} pessimistic at x1.40).
#   slowest single population {slowrow:.1f} min; slowest pack {slowpack:.1f} min
#   ({pslowpack:.1f} min pessimistic); elapsed ~{elapsed:.2f} h at %{conc},
#   EXCLUDING queue wait, which is expected to dominate the short arms.
#
#   rate = rate35(L, N_c) * small_batch(N_c) * rho(zeta) * f_lam(lambda, zeta).
#   rate35 is Ruche-MEASURED and FLAT in N_c: the N_c^0.1871 growth law adopted
#   by TASK-2026-09-03-NC-PLATEAU-CALIBRATION is refuted by that campaign's own
#   returned rungs (L=64 N_c=4096/8192 measured 5.05/5.11 ms against 6.6/7.5
#   predicted; L=128 N_c=2048 measured 22.15 against 31.7). See ../COST_MODEL.md.
#
# --time is >= 1.6 x the PESSIMISTIC slowest PACK, snapped up. Partition is then
#   chosen to fit the time: cpu_med if it fits the 4 h MaxTime, else cpu_long.
#   cpu_short is NEVER used -- it is effectively serialised for this account by
#   QOSMaxJobsPerUserLimit, so its %N cap is not real
#   (TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/SCHEDULER_DECISION.md).
#
# --mem={mem} against a modelled peak of {memmb:.0f} MB (1.45 x the inherited
#   formula; the formula alone under-predicts real peak RSS by up to 1.6x,
#   measured at 15 cells). zeta does not enter memory.
#
# --array=0-{last} : ONE task per pack, exactly. preflight.py fails if this
#   stops matching packs.csv. %{conc} is a concurrency cap only.
# ============================================================================

set -euo pipefail

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
    echo "Export PPSQJ_PYTHON to the validated interpreter, e.g." >&2
    echo "  export PPSQJ_PYTHON=\\$WORKDIR/envs/pps_qj/bin/python" >&2
    exit 2
fi
export PATH="$(dirname "$PPSQJ_PYTHON"):$PATH"

echo "[{name}] task ${{SLURM_ARRAY_TASK_ID}} on $(hostname) at $(date -u +%FT%TZ)"
echo "[{name}] partition=${{SLURM_JOB_PARTITION:-?}}  python=$PPSQJ_PYTHON"
"$PPSQJ_PYTHON" -c 'import sys,numpy;print("[{name}] resolved",sys.executable,"numpy",numpy.__version__)'

"$PPSQJ_PYTHON" {shared}/run_pack.py "${{SLURM_ARRAY_TASK_ID}}"
echo "[{name}] task ${{SLURM_ARRAY_TASK_ID}} done at $(date -u +%FT%TZ)"
"""


def run():
    summary, index = [], 0
    for z in sorted(GRID):
        for nc in NCS:
            name = arm_name(z, nc)
            cond = name in CONDITIONAL
            outdir = os.path.join(TASK, "conditional" if cond else ".", name)
            purpose = ("zeta = %.2f, N_c = %d: the full 9-point lambda grid at "
                       "L = 32, 48, 64 (T = L), R = %d independent populations "
                       "per cell. One rung of the N_c ladder." % (z, nc, R))
            extra = ("# CONDITIONAL ARM -- DO NOT SUBMIT WITH THE REST.\n"
                     "#   This single rung is 42.9 %% of the whole campaign's core-hours.\n"
                     "#   Interlock in ../CONDITIONAL_SUBMISSION.md must be satisfied first."
                     if cond else
                     "# Committed arm. Submit per ../HUMAN_SUBMISSION.md.")
            summary.append(build(name, main_rows(z, nc), index, outdir, purpose, extra))
            index += 1
    summary.append(build(
        CONTROL["arm"], control_rows(), index,
        os.path.join(TASK, CONTROL["arm"]),
        ("Discretisation sanity check (brief section 8). L = 64, N_c = 512, "
         "zeta = 0.10, lambda = 0.100, dtau_mult in {3, 12}, R = 16. It detects "
         "an OBVIOUS dtau dependence at low zeta and nothing more."),
        ("# THE dtau_mult = 6 LEG IS NOT IN THIS ARM. It already exists inside\n"
         "#   M_z010_nc512 at the same L, N_c, zeta, lambda and R, because\n"
         "#   lambda = 0.100 is an on-grid point. Running it again under a second\n"
         "#   seed block and comparing the two as one measurement is the error this\n"
         "#   avoids. SUBMISSION DEPENDENCY: analyse only after M_z010_nc512 returns.\n"
         "#   dtau_mult != 6 rows are a control and are NEVER pooled with the\n"
         "#   production corpus.")))
    return summary


if __name__ == "__main__":
    s = run()
    with open(os.path.join(TASK, "analysis", "campaign_cost.json"), "w") as f:
        json.dump(s, f, indent=1)
    comm = [a for a in s if not a["conditional"]]
    cond = [a for a in s if a["conditional"]]
    print("%-18s %6s %6s %9s %9s %10s %8s %-9s %-9s %s"
          % ("arm", "pops", "tasks", "core-h", "pess", "slowpack-m", "elapsed", "part", "time", "mem"))
    for a in s:
        print("%-18s %6d %6d %9.1f %9.1f %10.1f %8.2f %-9s %-9s %s%s"
              % (a["arm"], a["populations"], a["array_tasks"], a["core_h"],
                 a["pessimistic_core_h"], a["slowest_pack_s"] / 60, a["elapsed_h"],
                 a["partition"], a["time"], a["mem"],
                 "   [CONDITIONAL]" if a["conditional"] else ""))
    print("\nCOMMITTED : %d arms, %d populations, %d array tasks, %.1f core-h (%.1f pessimistic)"
          % (len(comm), sum(a["populations"] for a in comm), sum(a["array_tasks"] for a in comm),
             sum(a["core_h"] for a in comm), sum(a["pessimistic_core_h"] for a in comm)))
    print("CONDITIONAL: %d arm, %d populations, %.1f core-h (%.1f pessimistic)"
          % (len(cond), sum(a["populations"] for a in cond),
             sum(a["core_h"] for a in cond), sum(a["pessimistic_core_h"] for a in cond)))
    tot = sum(a["core_h"] for a in s)
    print("SAVED by holding the conditional rung: %.1f core-h = %.1f %% of the full design"
          % (sum(a["core_h"] for a in cond), 100 * sum(a["core_h"] for a in cond) / tot))
