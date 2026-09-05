#!/usr/bin/env python3
"""Whole-repository inventory of every stored population, for reuse and dedup.

Read-only. Writes EXISTING_DATA_INVENTORY.csv and prints the compatibility
classification this task depends on. It scans research/tasks/**/results/*.json
rather than manifests, so a manifest row that never returned a result cannot be
counted as existing data.

Compatibility, three classes:
  EXACT      same (zeta, L, T, N_c, lam, dtau_mult, resample_scheme) as a cell
             this task would run -> the cell would be a DUPLICATE
  ADJACENT   same zeta but a cell this task does not run
  INCOMPATIBLE different zeta, or a non-production configuration
"""
import csv, glob, json, os, sys, collections

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 5)))
OUT = os.path.join(os.path.dirname(__file__), os.pardir, "EXISTING_DATA_INVENTORY.csv")

GRID = {0.10: [0.040, 0.055, 0.070, 0.085, 0.100, 0.115, 0.130, 0.145, 0.160],
        0.20: [0.080, 0.1025, 0.125, 0.1475, 0.170, 0.1925, 0.215, 0.2375, 0.260],
        0.70: [0.250, 0.284, 0.318, 0.351, 0.385, 0.419, 0.453, 0.486, 0.520]}
LS, NCS = [32, 48, 64], [128, 256, 512, 1024, 2048]
WANTED = {(z, L, float(L), nc, round(lam, 6), 6.0, "systematic")
          for z in GRID for L in LS for nc in NCS for lam in GRID[z]}
# the discretisation control cells
WANTED |= {(0.10, 64, 64.0, 512, 0.100, dt, "systematic") for dt in (3.0, 12.0)}

rows, cells = [], collections.Counter()
for p in glob.glob(os.path.join(ROOT, "research/tasks/**/results/*.json"), recursive=True):
    if os.sep + "scratch" + os.sep in p:
        continue          # synthetic / throwaway; never a stored population
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if not isinstance(d, dict) or "zeta" not in d or "N_c" not in d:
        continue
    try:
        key = (float(d["zeta"]), int(d["L"]), float(d.get("T", 0)), int(d["N_c"]),
               round(float(d["lam"]), 6), float(d.get("dtau_mult", 0)),
               d.get("resample_scheme") or "")
    except Exception:
        continue
    cells[key] += 1
    rows.append(key + (os.path.relpath(p, ROOT).split("/")[3], d.get("seed"), d.get("wall_s"),
                       d.get("n_steps"), d.get("status")))

def klass(k):
    if k in WANTED:
        return "EXACT"
    if k[5] != 6.0 or k[6] != "systematic":
        return "INCOMPATIBLE"
    return "ADJACENT" if k[0] in GRID else "INCOMPATIBLE"

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow("zeta L T N_c lam dtau_mult resample_scheme n_populations "
               "compatibility tasks".split())
    bytask = collections.defaultdict(set)
    for r in rows:
        bytask[r[:7]].add(r[7])
    for k in sorted(cells):
        w.writerow(list(k) + [cells[k], klass(k), "|".join(sorted(bytask[k]))])

n_exact = sum(v for k, v in cells.items() if klass(k) == "EXACT")
n_adj = sum(v for k, v in cells.items() if klass(k) == "ADJACENT")
print("populations scanned      : %d" % sum(cells.values()))
print("distinct cells           : %d" % len(cells))
print("cells this task would run: %d" % len(WANTED))
print("EXACT-compatible existing: %d populations in %d cells"
      % (n_exact, sum(1 for k in cells if klass(k) == "EXACT")))
print("ADJACENT (same zeta, other cell): %d populations" % n_adj)
print("zeta values present in the corpus: %s"
      % sorted({k[0] for k in cells}))
sys.exit(0 if n_exact == 0 else 3)
