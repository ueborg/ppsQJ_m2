#!/usr/bin/env python3
"""Duplicate, seed and manifest audit for TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION.

    python3 tools/dedup_scan.py

Answers four questions, each independently of the preflight, so that a single
mistake in one checker cannot pass unnoticed:

  D1  does any manifest row recompute a physical cell that already exists
      anywhere in the task tree, at ANY R?
  D2  are all 288 seeds fresh, distinct, and disjoint from every seed used
      anywhere before?
  D3  does every arm have exactly 96 rows, indices 0-95, four lambdas x R=24?
  D4  do the reused and new halves of the grid form ONE design -- same zeta,
      N_c, dtau_mult, T=L, resampling and R?

A duplicate cell would be a straightforward waste of core-hours. The subtler
harm is that the analysis averages populations by (L, lambda), so a duplicate
would silently be POOLED with the reused predecessor cell, changing a published
number without changing any file that records it.

This script contains no scheduler call and cannot submit anything.
"""
import os, sys, csv, glob, json, collections

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
ACTIVE = os.path.abspath(os.path.join(TASK, os.pardir))
ARMS = ["lowlamL32", "lowlamL48", "lowlamL64"]
GRID = [round(0.1932 + 0.010 * i, 4) for i in range(17)]
NEW_LAMS = GRID[:4]
R = 24
NC = 1024


def cellkey(L, T, zeta, lam, N_c, dtau, scheme):
    return (int(L), float(T), float(zeta), round(float(lam), 6), int(N_c),
            float(dtau), scheme)


def main():
    problems = []
    print("=" * 78)
    print("  DUPLICATE / SEED / MANIFEST AUDIT")
    print("  TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION")
    print("=" * 78)

    # ---- this task's manifests -------------------------------------------
    mine, my_seeds = [], []
    print("\n  D3  MANIFEST STRUCTURE")
    for arm in ARMS:
        rows = list(csv.DictReader(open(os.path.join(TASK, arm,
                                                     "manifest.csv"))))
        lams = collections.Counter(round(float(r["lam"]), 4) for r in rows)
        seeds = [int(r["seed"]) for r in rows]
        ok = (len(rows) == 96 and sorted(lams) == NEW_LAMS
              and set(lams.values()) == {R} and len(set(seeds)) == 96)
        arr = "0-%d" % (len(rows) - 1)
        # the submit script must ask for exactly one task per row
        sl = open(os.path.join(TASK, arm, "submit.slurm")).read()
        arr_ok = ("--array=%s%%64" % arr) in sl
        part_ok = "--partition=cpu_med" in sl
        print("      %-12s rows=%3d  lambdas=%s  R=%s  seeds %d-%d  "
              "array=%s %s  cpu_med %s"
              % (arm, len(rows), sorted(lams), sorted(set(lams.values())),
                 min(seeds), max(seeds), arr,
                 "OK" if arr_ok else "MISMATCH",
                 "OK" if part_ok else "NO"))
        if not ok:
            problems.append("%s: manifest structure is not 4 lambdas x R=24 "
                            "= 96 rows" % arm)
        if not arr_ok:
            problems.append("%s: --array does not match the manifest" % arm)
        if not part_ok:
            problems.append("%s: not cpu_med" % arm)
        for r in rows:
            mine.append((arm, cellkey(r["L"], r["T"], r["zeta"], r["lam"],
                                      r["N_c"], r["dtau_mult"],
                                      r["resample_scheme"])))
        my_seeds += seeds

    # ---- D4 one design ----------------------------------------------------
    print("\n  D4  ONE DESIGN ACROSS THE JOIN")
    frozen = list(csv.DictReader(open(os.path.join(
        TASK, "frozen_inputs", "predecessor_nc1024_populations.csv"))))
    for label, rows in (("reused (13 lambdas)", frozen),
                        ("new    ( 4 lambdas)",
                         [dict(r) for arm in ARMS for r in
                          csv.DictReader(open(os.path.join(TASK, arm,
                                                           "manifest.csv")))])):
        z = sorted(set(float(r["zeta"]) for r in rows))
        n = sorted(set(int(r["N_c"]) for r in rows))
        dt = sorted(set(float(r["dtau_mult"]) for r in rows))
        sc = sorted(set(r["resample_scheme"] for r in rows))
        tl = all(abs(float(r["T"]) - int(r["L"])) < 1e-9 for r in rows)
        print("      %s  zeta=%s N_c=%s dtau=%s scheme=%s T==L=%s"
              % (label, z, n, dt, sc, tl))
        if z != [0.35] or n != [NC] or dt != [6.0] or sc != ["systematic"] \
                or not tl:
            problems.append("%s: the two halves of the grid are not one design"
                            % label)
    fr_R = collections.Counter((int(r["L"]), round(float(r["lam"]), 4))
                               for r in frozen)
    over = {k: v for k, v in fr_R.items() if v != R}
    print("      reused cells at R != %d: %s  (cut to block A in seed order "
          "by the analysis)" % (R, dict(over) if over else "none"))

    # ---- D1 duplicates ----------------------------------------------------
    print("\n  D1  DUPLICATE PHYSICAL CELLS")
    existing = collections.defaultdict(list)
    for r in frozen:
        existing[cellkey(r["L"], r["T"], r["zeta"], r["lam"], r["N_c"],
                         r["dtau_mult"], r["resample_scheme"])].append(
                             "frozen_inputs (predecessor)")
    # every OTHER manifest anywhere under research/tasks/active, so a clash
    # with a task nobody mentioned is still caught
    scanned = 0
    for p in sorted(glob.glob(os.path.join(ACTIVE, "*", "*", "manifest.csv"))):
        if os.path.abspath(p).startswith(TASK + os.sep):
            continue
        scanned += 1
        rel = os.path.relpath(p, ACTIVE)
        try:
            for r in csv.DictReader(open(p)):
                existing[cellkey(r["L"], r["T"], r["zeta"], r["lam"],
                                 r["N_c"], r["dtau_mult"],
                                 r["resample_scheme"])].append(rel)
        except (KeyError, ValueError):
            continue                      # a manifest of a different schema
    print("      scanned %d other manifests under research/tasks/active/"
          % scanned)
    print("      distinct pre-existing physical cells: %d" % len(existing))
    dup = sorted(set(k for _a, k in mine if k in existing))
    if dup:
        for k in dup[:10]:
            problems.append("DUPLICATE cell %s already exists in %s"
                            % (k, sorted(set(existing[k]))))
        print("      ** %d DUPLICATED CELLS **" % len(dup))
    else:
        print("      duplicates: 0   (the 12 new cells exist nowhere else)")

    # ---- D2 seeds ---------------------------------------------------------
    print("\n  D2  SEEDS")
    prior = set(json.load(open(os.path.join(HERE, "existing_seeds.json"))))
    alloc = json.load(open(os.path.join(HERE, "allocated_seeds.json")))
    print("      allocated here      %d, distinct %d, range %d-%d"
          % (len(my_seeds), len(set(my_seeds)), min(my_seeds), max(my_seeds)))
    print("      ledger of prior use %d seeds, max %d" % (len(prior), max(prior)))
    print("      overlap             %d" % len(set(my_seeds) & prior))
    print("      structural floor    %d > %d, so disjointness does not depend "
          "on the scan above" % (32_000_000, max(prior)))
    if len(set(my_seeds)) != 288:
        problems.append("expected 288 distinct seeds, got %d"
                        % len(set(my_seeds)))
    if set(my_seeds) & prior:
        problems.append("seed collision with prior use")
    if sorted(alloc) != sorted(set(my_seeds)):
        problems.append("tools/allocated_seeds.json does not match the "
                        "manifests")
    if min(my_seeds) <= max(prior):
        problems.append("the fresh block floor is not above every prior seed")

    print("\n" + "=" * 78)
    if problems:
        print("  AUDIT FAILED")
        for p in problems:
            print("    * %s" % p)
        print("=" * 78)
        return 1
    print("  AUDIT PASSED — 288 fresh non-overlapping seeds, 12 new physical")
    print("  cells that exist nowhere else, three 96-row cpu_med manifests, and")
    print("  one design either side of the join.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
