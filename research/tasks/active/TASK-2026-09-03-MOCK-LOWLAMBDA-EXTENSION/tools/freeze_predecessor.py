#!/usr/bin/env python3
"""Freeze the completed predecessor's N_c = 1024 populations into this task.

TASK-2026-09-02-MOCK-PRODUCTION is COMPLETE and its archive is READ ONLY. This
script reads it and writes ONE tracked CSV inside this task. It writes nothing
anywhere else and it modifies no predecessor file.

WHY A SNAPSHOT IS REQUIRED, not a convenience
---------------------------------------------
`.gitignore` carries a bare `results/` rule, so every one of the predecessor's
864 returned JSONs is UNTRACKED: `git ls-files` finds only `.gitkeep` under each
`*/results/`. A clean checkout of this repository therefore has no route to the
predecessor's measured populations at all. The predecessor hit exactly this and
solved it exactly this way (`frozen_inputs/armB_populations.csv`); this task
inherits the pattern rather than inventing one.

WHAT IS TAKEN
-------------
The 39 completed primary cells, and only those:

    L = 32, 48, 64      N_c = 1024      dtau_mult = 6.0      zeta = 0.35
    lambda = 0.2332 ... 0.3532  (the predecessor's frozen 13-point grid)

    L=32  mockL32/results/*.json                 13 lambdas x R=24   = 312 rows
    L=48  mockL48/results/*.json                 13 lambdas x R=24   = 312 rows
    L=64  mockL64/results/*.json                 10 lambdas x R=24   = 240 rows
    L=64  frozen_inputs/armB_populations.csv      3 lambdas x R=96   = 288 rows
                                                                     -------
                                                                      1152 rows

The three L=64 central lambdas 0.2932/0.3032/0.3132 are ABSENT from mockL64 by
the predecessor's own design; they live in its frozen ARM-B snapshot at R = 96
and are carried through here at full R = 96 so that this task can apply the
predecessor's OWN matched-R block rule (block A = the first 24 in seed order)
rather than inheriting a pre-cut subset it cannot audit.

WHAT IS DELIBERATELY EXCLUDED
-----------------------------
  * mockNC128L32 / mockNC128L48 / mockNC128L64 -- the N_c = 128 matched
    companion arms were CANCELLED and returned zero results. Nothing from them
    exists and nothing from them is read. Asserted below, not assumed.
  * mockL64nc2048 -- the N_c = 2048 shape-check arm. It is a different
    population size and has no place in a curve-shape / crossing extension at
    N_c = 1024. Asserted excluded below.
  * frozen_inputs/historical_corpus_zeta035.csv -- dtau_mult = 12.0, not
    poolable, and this task makes no use of it whatsoever.

This script contains no scheduler call and cannot submit anything.
"""
import os, sys, csv, json, glob, hashlib, collections

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
ACTIVE = os.path.abspath(os.path.join(TASK, os.pardir))
PRED = os.path.join(ACTIVE, "TASK-2026-09-02-MOCK-PRODUCTION")
PRED_ID = "TASK-2026-09-02-MOCK-PRODUCTION"

OLD_GRID = [round(0.2332 + 0.010 * i, 4) for i in range(13)]
CENTRE3 = [0.2932, 0.3032, 0.3132]
NC = 1024
ZETA = 0.35
DTAU_MULT = 6.0

FIELDS = ["source_task", "source_arm", "source_file", "L", "T", "zeta", "lam",
          "N_c", "dtau_mult", "resample_scheme", "seed", "status", "wall_s",
          "n_steps", "cmi_weighted_mean", "cmi_within_var", "n_nonfinite",
          "n_distinct_anc_final", "gess_final", "ess_cum_final",
          "ess_frac_mean", "brentq_fallbacks"]

# arms that must contribute, and arms that must NOT
TAKE = {"mockL32": 32, "mockL48": 48, "mockL64": 64}
REFUSE = ("mockL64nc2048", "mockNC128L32", "mockNC128L48", "mockNC128L64")


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if not os.path.isdir(PRED):
        sys.exit(f"predecessor archive not found: {PRED}")

    rows, problems = [], []

    # --- the three N_c=1024 result arms -----------------------------------
    for arm, L in sorted(TAKE.items()):
        paths = sorted(glob.glob(os.path.join(PRED, arm, "results", "*.json")))
        if not paths:
            problems.append(f"{arm}: no result JSONs found")
            continue
        for p in paths:
            d = json.load(open(p))
            if d.get("status") != "ok":
                problems.append(f"{p}: status={d.get('status')!r}")
                continue
            if int(d["N_c"]) != NC:
                problems.append(f"{p}: N_c={d['N_c']} is not {NC}")
                continue
            if int(d["L"]) != L:
                problems.append(f"{p}: L={d['L']} is not {L}")
                continue
            lam = round(float(d["lam"]), 4)
            if lam not in OLD_GRID:
                problems.append(f"{p}: lam={lam} is off the predecessor grid")
                continue
            rows.append(dict(
                source_task=PRED_ID, source_arm=arm,
                source_file=f"{PRED_ID}/{arm}/results/{os.path.basename(p)}",
                L=int(d["L"]), T=float(d["T"]), zeta=float(d["zeta"]), lam=lam,
                N_c=int(d["N_c"]), dtau_mult=float(d["dtau_mult"]),
                resample_scheme=d["resample_scheme"], seed=int(d["seed"]),
                status="ok", wall_s=float(d["wall_s"]), n_steps=int(d["n_steps"]),
                cmi_weighted_mean=repr(float(d["cmi_weighted_mean"])),
                cmi_within_var=repr(float(d["cmi_within_var"])),
                n_nonfinite=int(d["n_nonfinite"]),
                n_distinct_anc_final=int(d["n_distinct_anc_final"]),
                gess_final=repr(float(d["gess_final"])),
                ess_cum_final=repr(float(d["ess_cum_final"])),
                ess_frac_mean=repr(float(d["ess_frac_mean"])),
                brentq_fallbacks=int(d["brentq_fallbacks"])))

    # --- the reused ARM-B centre triple, carried through at full R = 96 ----
    ab = os.path.join(PRED, "frozen_inputs", "armB_populations.csv")
    n_armb = 0
    for r in csv.DictReader(open(ab)):
        if r["status"] != "ok":
            problems.append(f"armB row seed={r['seed']}: status={r['status']!r}")
            continue
        lam = round(float(r["lam"]), 4)
        if lam not in CENTRE3 or int(r["N_c"]) != NC or int(r["L"]) != 64:
            problems.append(f"armB row seed={r['seed']}: unexpected cell")
            continue
        rows.append(dict(
            source_task=r["source_task"], source_arm="armB(via " + PRED_ID + ")",
            source_file=r["source_file"], L=64, T=float(r["T"]),
            zeta=float(r["zeta"]), lam=lam, N_c=NC,
            dtau_mult=float(r["dtau_mult"]), resample_scheme=r["resample_scheme"],
            seed=int(r["seed"]), status="ok", wall_s=float(r["wall_s"]),
            n_steps=int(r["n_steps"]),
            cmi_weighted_mean=r["cmi_weighted_mean"],
            cmi_within_var=r["cmi_within_var"],
            n_nonfinite=int(r["n_nonfinite"]),
            n_distinct_anc_final=int(r["n_distinct_anc_final"]),
            gess_final=r["gess_final"], ess_cum_final=r["ess_cum_final"],
            ess_frac_mean=r["ess_frac_mean"],
            brentq_fallbacks=int(r["brentq_fallbacks"])))
        n_armb += 1

    # --- the refusals, asserted rather than assumed ------------------------
    print("EXCLUSION ASSERTIONS (checked against the predecessor archive):")
    for arm in REFUSE:
        n = len(glob.glob(os.path.join(PRED, arm, "results", "*.json")))
        note = ("cancelled, zero results returned" if n == 0 else
                f"{n} results present and DELIBERATELY NOT READ")
        print(f"  {arm:<16} excluded -- {note}")

    # --- structure checks ---------------------------------------------------
    per = collections.Counter((r["L"], r["lam"]) for r in rows)
    for L in (32, 48, 64):
        for lam in OLD_GRID:
            want = 96 if (L == 64 and lam in CENTRE3) else 24
            got = per.get((L, lam), 0)
            if got != want:
                problems.append(f"cell L={L} lam={lam}: R={got}, expected {want}")
    seeds = [r["seed"] for r in rows]
    if len(set(seeds)) != len(seeds):
        problems.append("duplicate seeds inside the frozen snapshot")
    if sorted({r["N_c"] for r in rows}) != [NC]:
        problems.append(f"N_c set is {sorted({r['N_c'] for r in rows})}")
    if sorted({r["dtau_mult"] for r in rows}) != [DTAU_MULT]:
        problems.append("dtau_mult is not uniformly the certified 6.0")
    if sorted({r["zeta"] for r in rows}) != [ZETA]:
        problems.append("zeta is not uniformly 0.35")

    if problems:
        print("\nFREEZE FAILED:")
        for p in problems[:40]:
            print(f"  * {p}")
        return 1

    rows.sort(key=lambda r: (r["L"], r["lam"], r["seed"]))
    dest = os.path.join(TASK, "frozen_inputs", "predecessor_nc1024_populations.csv")
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {os.path.relpath(dest, TASK)}")
    print(f"  rows        {len(rows)}")
    print(f"  cells       {len(per)} (expect 39)")
    print(f"  L           {sorted({r['L'] for r in rows})}")
    print(f"  lambda      {len(OLD_GRID)} points {OLD_GRID[0]} .. {OLD_GRID[-1]}")
    print(f"  seeds       {min(seeds)} - {max(seeds)}, {len(set(seeds))} distinct")
    print(f"  sha256      {sha256(dest)}")
    print(f"  bytes       {os.path.getsize(dest)}")
    print("\nThe predecessor archive was READ and NOT MODIFIED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
