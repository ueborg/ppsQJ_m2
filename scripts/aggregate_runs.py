#!/usr/bin/env python3
"""
Aggregate fss_*.npz output files from all runs into a single pickle.

Scans one or more directories for fss_*.npz files (produced by
worker_fss_direct), loads every file, and writes a single dict:

    {(L, lam, zeta): {key: value, ...}, ...}

saved as a pickle at the specified output path.

Usage:
    python scripts/aggregate_runs.py \\
        /scratch/s4629701/pps_qj/pps_run_A \\
        /scratch/s4629701/pps_qj/pps_run_B \\
        /scratch/s4629701/pps_qj/pps_run_C \\
        /scratch/s4629701/pps_qj/pps_dense_zeta \\
        /scratch/s4629701/pps_qj/pps_fss_test \\
        --output /scratch/s4629701/pps_qj/aggregate_new.pkl

The output pkl can be downloaded and loaded with:
    import pickle
    agg = pickle.load(open('aggregate_new.pkl', 'rb'))
    # agg[(128, 0.35, 1.0)]['B_L_mean'] etc.
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np

SCALAR_KEYS = [
    "L", "lam", "alpha", "w", "zeta", "T", "N_c",
    "S_mean", "S_std", "S_err", "S_var",
    "n_T_mean", "chi_k",
    "B_L_mean", "B_L_err",
    "S_renyi_2_mean", "S_renyi_2_err",
    "S_renyi_3_mean", "S_renyi_3_err",
    "elapsed",
]

def load_npz(path: Path) -> tuple | None:
    try:
        f = np.load(path, allow_pickle=False)
        L    = int(f["L"])
        lam  = round(float(f["lam"]),  6)
        zeta = round(float(f["zeta"]), 6)
        row  = {}
        for k in SCALAR_KEYS:
            row[k] = float(f[k]) if k in f else float("nan")
        row["source"] = path.parent.name   # which run directory
        return (L, lam, zeta), row
    except Exception as e:
        print(f"[warn] {path.name}: {e}", file=sys.stderr)
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="One or more directories containing fss_*.npz files")
    ap.add_argument("--output", "-o", type=Path, required=True,
                    help="Output pickle path")
    args = ap.parse_args()

    agg = {}
    n_loaded = n_skip = 0

    for d in args.dirs:
        if not d.exists():
            print(f"[skip] {d} does not exist", file=sys.stderr)
            continue
        files = sorted(d.rglob("fss_*.npz"))
        print(f"  {d.name}: {len(files)} files")
        for p in files:
            result = load_npz(p)
            if result is None:
                n_skip += 1
                continue
            key, row = result
            if key in agg:
                # keep the newer one if it has B_L and the existing doesn't
                existing_has_bl = not np.isnan(agg[key].get("B_L_mean", float("nan")))
                new_has_bl      = not np.isnan(row.get("B_L_mean", float("nan")))
                if not existing_has_bl and new_has_bl:
                    agg[key] = row
                    # else keep existing
            else:
                agg[key] = row
            n_loaded += 1

    print(f"\nTotal: {n_loaded} loaded, {n_skip} skipped, {len(agg)} unique (L, lam, zeta) points")

    # Summary table
    Ls    = sorted(set(k[0] for k in agg))
    zetas = sorted(set(k[2] for k in agg))
    lams  = sorted(set(k[1] for k in agg))
    print(f"L values:    {Ls}")
    print(f"zeta values: {zetas}")
    print(f"lam values:  {len(lams)} unique ({min(lams):.4f} to {max(lams):.4f})")

    n_with_bl     = sum(1 for v in agg.values() if not np.isnan(v.get("B_L_mean", float("nan"))))
    n_with_renyi  = sum(1 for v in agg.values() if not np.isnan(v.get("S_renyi_2_mean", float("nan"))))
    print(f"Points with B_L_mean:       {n_with_bl} / {len(agg)}")
    print(f"Points with S_renyi_2_mean: {n_with_renyi} / {len(agg)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(agg, f, protocol=4)
    print(f"\nSaved -> {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
