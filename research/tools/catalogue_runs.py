#!/usr/bin/env python3
"""READ-ONLY catalogue of per-realisation metadata under a results root.
Writes nothing outside research/runs/_catalogue/. Charter tier T0."""
import json, os, re, sys, collections, csv
ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/ruche_pull"
OUT  = "research/runs/_catalogue"
os.makedirs(OUT, exist_ok=True)
FIELDS = ["L","T","zeta","lambda","N_c","dtau","real","alpha","w","burn_in",
          "seed","seeds","git_commit","job_id","n_real","task_id"]
DIRPAT = re.compile(r"L(\d+)_z([0-9.]+)_lam([0-9.]+)")
rows, present, cells = [], collections.Counter(), collections.defaultdict(list)
nfiles = 0
for dp, dn, fn in os.walk(ROOT):
    for f in fn:
        if not f.endswith(".json"): continue
        p = os.path.join(dp, f)
        try: d = json.load(open(p))
        except Exception: continue
        if not isinstance(d, dict): continue
        nfiles += 1
        for k in FIELDS:
            if k in d and d[k] is not None: present[k] += 1
        m = DIRPAT.search(os.path.basename(dp))
        dirmeta = {"dir_L": int(m.group(1)), "dir_zeta": float(m.group(2)),
                   "dir_lam": float(m.group(3))} if m else {}
        L, T = d.get("L"), d.get("T")
        consistent = (dirmeta.get("dir_L") == L) if (m and L) else None
        rows.append({"path": os.path.relpath(p, ROOT),
                     "campaign": os.path.relpath(dp, ROOT).split(os.sep)[0],
                     "L": L, "T": T,
                     "T_over_L": (T/L) if (L and T) else None,
                     "zeta": d.get("zeta"), "lambda": d.get("lambda", d.get("lam")),
                     "N_c": d.get("N_c"), "n_real": d.get("n_real"),
                     "dir_consistent": consistent})
        if L and T: cells[(rows[-1]["campaign"], L)].append(T/L)
print(f"root={ROOT}  json files parsed: {nfiles}")
print("\n--- field recoverability (fraction of parsed JSON) ---")
for k in FIELDS:
    c = present[k]
    if c: print(f"  {k:12s} {c:6d}/{nfiles}  {100*c/nfiles:5.1f}%")
missing = [k for k in FIELDS if not present[k]]
print("  ABSENT FROM EVERY FILE:", ", ".join(missing) if missing else "(none)")
print("\n--- T/L by campaign and L ---")
print(f"  {'campaign':<16}{'L':>5}{'n':>7}{'T/L min':>9}{'T/L max':>9}{'>=2?':>6}")
any2 = False
for (camp, L), v in sorted(cells.items()):
    ok = max(v) >= 2.0
    any2 |= ok
    print(f"  {camp:<16}{L:>5}{len(v):>7}{min(v):>9.3f}{max(v):>9.3f}{('YES' if ok else 'no'):>6}")
print(f"\n  ANY run with T/L >= 2 : {'YES' if any2 else 'NO'}")
bad = [r for r in rows if r["dir_consistent"] is False]
print(f"  dirname/JSON L mismatches: {len(bad)}")
with open(os.path.join(OUT, "ruche_pull_catalogue.csv"), "w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
print(f"\nwrote {OUT}/ruche_pull_catalogue.csv ({len(rows)} rows)")
