#!/usr/bin/env python3
"""
Aggregate guided-cloning production runs into a single pkl keyed by
(L, lam, zeta).  Covers:
  pps_clone_guided_prod   -- Cut B main grid
  pps_clone_guided_highL  -- L=160
  pps_clone_guided_ladder -- N_c ladder (second rung)
  pps_caseA_guided        -- Case A

Run on Habrok:
    cd ~/pps_qj
    python analysis/aggregate_guided.py

Outputs (in /scratch/$USER/pps_qj/):
    guided_prod_aggregate.pkl
    guided_highL_aggregate.pkl
    guided_ladder_aggregate.pkl
    caseA_guided_aggregate.pkl

Then scp each to ~/Downloads/ on Mac for analysis.
"""
import os, sys, pickle, json, numpy as np
from pathlib import Path
from collections import defaultdict

SCRATCH = Path(f"/scratch/{os.environ['USER']}/pps_qj")

def aggregate_dir(npz_dir: Path, label: str) -> dict:
    files = sorted(npz_dir.glob("clone_*.npz"))
    if not files:
        print(f"  [{label}] No .npz files found in {npz_dir}", flush=True)
        return {}
    print(f"  [{label}] Found {len(files)} files in {npz_dir}", flush=True)

    agg = {}
    skipped = 0
    for fp in files:
        try:
            d = np.load(fp, allow_pickle=True)
            L    = int(d["L"])
            lam  = float(d["lam"])
            zeta = float(d["zeta"])
            key  = (L, round(lam,6), round(zeta,6))

            entry = {
                "L": L, "lam": lam, "zeta": zeta,
                "N_c": int(d["N_c"]),
                "T": float(d["T"]),
                "S_mean":    float(d["S_mean"])    if "S_mean"    in d else np.nan,
                "S_err":     float(d["S_err"])     if "S_err"     in d else np.nan,
                "B_L_mean":  float(d["B_L_mean"])  if "B_L_mean"  in d else np.nan,
                "B_L_err":   float(d["B_L_err"])   if "B_L_err"   in d else np.nan,
                "CMI_mean":  float(d["CMI_mean"])  if "CMI_mean"  in d else np.nan,
                "CMI_err":   float(d["CMI_err"])   if "CMI_err"   in d else np.nan,
                "n_clones":  int(d["n_clones"])    if "n_clones"  in d else -1,
            }
            # if duplicate key, keep the one with more clones / lower B_L_err
            if key in agg:
                old_err = agg[key].get("B_L_err", np.inf) or np.inf
                new_err = entry["B_L_err"] or np.inf
                if new_err < old_err:
                    agg[key] = entry
            else:
                agg[key] = entry
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"    WARN: {fp.name}: {e}", flush=True)
    print(f"  [{label}] Aggregated {len(agg)} unique (L,lam,zeta) keys "
          f"({skipped} files skipped)", flush=True)
    return agg

campaigns = [
    ("pps_clone_guided_prod",   "guided_prod"),
    ("pps_clone_guided_highL",  "guided_highL"),
    ("pps_clone_guided_ladder", "guided_ladder"),
    ("pps_caseA_guided",        "caseA_guided"),
]

for dirname, label in campaigns:
    d = SCRATCH / dirname
    if not d.exists():
        print(f"  [{label}] Directory not found: {d}", flush=True)
        continue
    agg = aggregate_dir(d, label)
    if agg:
        out = SCRATCH / f"{label}_aggregate.pkl"
        with open(out, "wb") as f:
            pickle.dump(agg, f, protocol=4)
        print(f"  [{label}] Saved -> {out}", flush=True)

print("Done.", flush=True)
