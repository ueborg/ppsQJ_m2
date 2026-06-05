#!/usr/bin/env python
"""Inventory cloning output: what (L, lambda, zeta) points exist, how many
seeds, what N_c, and the achieved precision -- so we know what to reuse vs
top up before committing a campaign.

Accepts any mix of:
  - a directory of clone_*.npz  (e.g. /scratch/$USER/pps_qj/pps_clone_dense)
  - a saved aggregate .pkl       (dict keyed by (L, lam, zeta))
Partial/locked files (e.g. a dir being written by a running job) are skipped.

Usage:
    python analysis/inventory_data.py /scratch/$USER/pps_qj/pps_clone_dense \
        /scratch/$USER/pps_qj/pps_clone_rescue  path/to/clone_aggregate.pkl \
        --out outputs/diagnostics/inventory.json
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FIELDS = ("L", "lam", "zeta", "N_c", "T", "n_real",
          "B_L_mean", "B_L_err", "CMI_mean", "CMI_err", "theta_hat")


def _rec_from_npz(path):
    try:
        with np.load(path, allow_pickle=False) as d:
            r = {}
            for k in FIELDS:
                if k in d.files:
                    v = d[k]
                    r[k] = v.item() if v.ndim == 0 else float(np.asarray(v).ravel()[0])
            r["_src"] = path.parent.name
            return r if "L" in r else None
    except Exception:
        return None  # truncated / mid-write / locked


def _recs_from_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    recs = []
    for key, rec in data.items():
        r = {"_src": Path(path).name}
        for k in FIELDS:
            if isinstance(rec, dict) and k in rec:
                v = rec[k]
                try:
                    r[k] = float(np.asarray(v).ravel()[0]) if np.ndim(v) else float(v)
                except Exception:
                    pass
        if "L" not in r and isinstance(key, (tuple, list)) and len(key) == 3:
            r["L"], r["lam"], r["zeta"] = float(key[0]), float(key[1]), float(key[2])
        if "L" in r:
            recs.append(r)
    return recs


def collect(paths):
    recs = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = sorted(p.glob("clone_*.npz"))
            ok = 0
            for f in files:
                r = _rec_from_npz(f)
                if r:
                    recs.append(r); ok += 1
            print(f"  {p}: {ok}/{len(files)} npz readable", flush=True)
        elif p.suffix == ".pkl":
            rr = _recs_from_pkl(p)
            recs.extend(rr)
            print(f"  {p}: {len(rr)} points from pickle", flush=True)
        else:
            print(f"  {p}: skipped (not a dir or .pkl)", flush=True)
    return recs


def _relerr(rec, obs):
    m, e = rec.get(f"{obs}_mean"), rec.get(f"{obs}_err")
    if m and e and np.isfinite(m) and np.isfinite(e) and m != 0:
        return abs(e / m)
    return float("nan")


def summarize(recs, target=0.05):
    by_L = defaultdict(list)
    for r in recs:
        by_L[int(round(r["L"]))].append(r)

    print("\n" + "=" * 84)
    print("COVERAGE BY L")
    print("=" * 84)
    print(f"{'L':>4} {'pts':>5} {'zetas':>6} {'N_c':>14} {'seeds':>7} "
          f"{'relErr B_L':>11} {'relErr CMI':>11} {'<target':>8}")
    summary = {"by_L": {}, "by_L_zeta": {}, "points": []}
    for L in sorted(by_L):
        rs = by_L[L]
        zetas = sorted({round(r["zeta"], 3) for r in rs})
        ncs = sorted({int(r["N_c"]) for r in rs if "N_c" in r})
        seeds = int(sum(r.get("n_real", 0) for r in rs))
        be = [_relerr(r, "B_L") for r in rs]; ce = [_relerr(r, "CMI") for r in rs]
        be = [x for x in be if np.isfinite(x)]; ce = [x for x in ce if np.isfinite(x)]
        med_b = float(np.median(be)) if be else float("nan")
        med_c = float(np.median(ce)) if ce else float("nan")
        n_ok = sum(1 for x in be if x <= target)
        ncs_s = ",".join(map(str, ncs)) if len(ncs) <= 4 else f"{ncs[0]}..{ncs[-1]}"
        print(f"{L:>4} {len(rs):>5} {len(zetas):>6} {ncs_s:>14} {seeds:>7} "
              f"{med_b:>10.1%} {med_c:>10.1%} {n_ok:>4}/{len(be):<3}")
        summary["by_L"][L] = dict(points=len(rs), zetas=zetas, N_c=ncs, seeds=seeds,
                                  median_relErr_BL=med_b, median_relErr_CMI=med_c,
                                  n_points_below_target=n_ok)

    # which zeta are covered at the reliable FSS triple (32/64/128)?
    triple = {32, 64, 128}
    z_by_L = {L: {round(r["zeta"], 3) for r in by_L[L]} for L in by_L}
    common = set.intersection(*[z_by_L[L] for L in triple if L in z_by_L]) \
        if triple.issubset(by_L) else set()
    print("\n" + "-" * 84)
    print(f"zeta covered at ALL of L=32,64,128 (FSS-ready): "
          f"{sorted(common) if common else 'NONE (missing one of the triple)'}")

    # per (L,zeta): how many lambda points + median rel-err (crossing readiness)
    print("\nPER (L, zeta): #lambda points / median B_L rel-err  "
          "[* = below target]")
    for L in sorted(by_L):
        by_z = defaultdict(list)
        for r in by_L[L]:
            by_z[round(r["zeta"], 3)].append(r)
        cells = []
        for z in sorted(by_z):
            rs = by_z[z]
            be = [_relerr(r, "B_L") for r in rs]
            be = [x for x in be if np.isfinite(x)]
            md = np.median(be) if be else float("nan")
            star = "*" if np.isfinite(md) and md <= target else " "
            cells.append(f"z{z}:{len(rs)}/{md:.0%}{star}")
            summary["by_L_zeta"][f"{L}_{z}"] = dict(n_lambda=len(rs),
                                                    median_relErr_BL=float(md))
        print(f"  L={L:>3}: " + "  ".join(cells))

    summary["points"] = [{k: r.get(k) for k in FIELDS if k in r} | {"src": r.get("_src")}
                         for r in recs]
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="inventory cloning outputs")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="outputs/diagnostics/inventory.json")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    print("Scanning:")
    recs = collect(args.paths)
    print(f"\nTotal points: {len(recs)}")
    if not recs:
        print("No readable data found."); return 1
    summary = summarize(recs, args.target)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nFull per-point inventory -> {out}")
    print("(paste that JSON here and I can do the global-fit feasibility + "
          "reuse/top-up plan)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
