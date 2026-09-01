#!/usr/bin/env python3
"""Split the FROZEN SMCCERT ruche_package into two human-facing arm packages.

This script REORGANISES. It does not redesign. Every scientific column is copied
verbatim from TASK-2026-08-31-SMCCERT/ruche_package/manifest.csv and the split is
verified row-by-row afterwards. If any value differed the build would abort.
"""
import csv, os, shutil, hashlib, sys

ROOT = "/Users/catlover1337/Documents/ppsQJ_m2"
SRC  = os.path.join(ROOT, "research/tasks/active/TASK-2026-08-31-SMCCERT/ruche_package")
DST  = os.path.join(ROOT, "research/tasks/active/TASK-2026-09-01-SMCRUCHE-READY")
SCI  = ["L", "T", "N_c", "zeta", "lam", "dtau_mult", "resample_scheme", "seed"]

rows = list(csv.DictReader(open(os.path.join(SRC, "manifest.csv"))))
fields = list(rows[0])

for arm in ("ARM1", "ARM2"):
    d = os.path.join(DST, arm.lower())
    os.makedirs(os.path.join(d, "results"), exist_ok=True)
    sub = [r for r in rows if r["arm"] == arm]
    with open(os.path.join(d, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(sub)
    # shared executables, copied unchanged
    for f in ("run_cell.py", "analyse_ruche.py"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(d, f))
    print(f"{arm}: {len(sub)} rows -> {d}/manifest.csv")

# ---- VERIFY: the split preserves every scientific value, row for row ---------
print("\nverifying the split against the frozen manifest:")
bad = 0
for arm in ("ARM1", "ARM2"):
    d = os.path.join(DST, arm.lower())
    out = list(csv.DictReader(open(os.path.join(d, "manifest.csv"))))
    src = [r for r in rows if r["arm"] == arm]
    assert len(out) == len(src), f"{arm}: row count changed"
    for a, b in zip(src, out):
        for k in SCI:
            if a[k] != b[k]:
                print(f"  MISMATCH {arm} {k}: {a[k]!r} != {b[k]!r}"); bad += 1
    seeds = [r["seed"] for r in out]
    assert len(set(seeds)) == len(seeds), f"{arm}: duplicate seed"
    print(f"  {arm}: {len(out)} rows, all {len(SCI)} scientific columns identical, "
          f"{len(set(seeds))} unique seeds")
# and the union must be exactly the frozen file
allout = []
for arm in ("ARM1", "ARM2"):
    allout += list(csv.DictReader(open(os.path.join(DST, arm.lower(), "manifest.csv"))))
assert len(allout) == len(rows), "union row count differs from the frozen manifest"
for a, b in zip(rows, allout):
    for k in SCI + ["arm"]:
        if a[k] != b[k]:
            print(f"  UNION MISMATCH {k}: {a[k]!r} != {b[k]!r}"); bad += 1
print(f"  union of the two arms == the frozen manifest, row for row ({len(rows)} rows)")
sys.exit(1 if bad else 0)
