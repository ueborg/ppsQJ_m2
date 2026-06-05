#!/usr/bin/env python
"""Finite-N_c bias from matched (L, lambda, zeta) points across two datasets
that share T but differ in N_c (e.g. dense N_c=250 vs v2 N_c=100 at L=128).

The two campaigns were run at different population sizes, so they are NOT
poolable as extra seeds -- but at matched T they form a 2-point 1/N_c ladder
per point. This:
  * measures how far the production-N_c B_L sits from the N_c->inf value,
  * flags whether the existing FSS (N_c decreasing with L) is bias-contaminated,
  * emits the extrapolated B_inf for a debiased FSS, with zero new simulation.

Usage:
    python analysis/nc_bias_pairs.py \
        --a /scratch/$USER/pps_qj/pps_clone_rescue \
        --b path/to/clone_aggregate_2_.pkl \
        --out outputs/diagnostics/nc_bias_pairs.json
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

F = ("L", "lam", "zeta", "N_c", "T", "n_real", "B_L_mean", "B_L_err")


def _from_npz(path):
    try:
        with np.load(path, allow_pickle=False) as d:
            r = {k: (d[k].item() if d[k].ndim == 0 else float(np.asarray(d[k]).ravel()[0]))
                 for k in F if k in d.files}
            return r if "L" in r else None
    except Exception:
        return None


def _load(path):
    """Return list of records from a dir of clone_*.npz or an aggregate .pkl."""
    p = Path(path)
    recs = []
    if p.is_dir():
        for f in sorted(p.glob("clone_*.npz")):
            r = _from_npz(f)
            if r:
                recs.append(r)
    elif p.suffix == ".pkl":
        data = pickle.load(open(p, "rb"))
        for key, rec in data.items():
            r = {k: float(rec[k]) for k in F if isinstance(rec, dict) and k in rec}
            if "L" not in r and isinstance(key, (tuple, list)) and len(key) == 3:
                r["L"], r["lam"], r["zeta"] = float(key[0]), float(key[1]), float(key[2])
            if "L" in r:
                recs.append(r)
    else:
        raise SystemExit(f"unsupported path: {path}")
    return recs


def _load_many(paths):
    out = []
    for p in paths:
        out.extend(_load(p))
    return out


def _index(recs):
    """key (L, lam, zeta, T) -> record, so matching enforces equal T."""
    idx = {}
    for r in recs:
        k = (int(round(r["L"])), round(float(r["lam"]), 3),
             round(float(r["zeta"]), 3), round(float(r.get("T", 0)), 3))
        idx[k] = r
    return idx


def _extrap(n_lo, b_lo, n_hi, b_hi):
    """Linear-in-1/N_c extrapolation to N_c->inf from two points."""
    x_lo, x_hi = 1.0 / n_lo, 1.0 / n_hi
    c = (b_lo - b_hi) / (x_lo - x_hi)
    b_inf = b_hi - c * x_hi
    return b_inf


def main(argv=None):
    ap = argparse.ArgumentParser(description="finite-N_c bias from matched pairs")
    ap.add_argument("--a", required=True, nargs="+", help="dataset A: scratch dir(s) or .pkl")
    ap.add_argument("--b", required=True, nargs="+", help="dataset B: scratch dir(s) or .pkl")
    ap.add_argument("--bl-min", type=float, default=0.3,
                    help="ignore points with |B_L|<this (deep-phase, bias meaningless)")
    ap.add_argument("--bl-max", type=float, default=12.0)
    ap.add_argument("--out", type=str, default="outputs/diagnostics/nc_bias_pairs.json")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    A, B = _index(_load_many(args.a)), _index(_load_many(args.b))
    print(f"A: {len(A)} points   B: {len(B)} points")
    by_L = defaultdict(list)
    for k in set(A) & set(B):
        ra, rb = A[k], B[k]
        na, nb = int(ra["N_c"]), int(rb["N_c"])
        if na == nb:
            continue  # same N_c: poolable, not a bias pair
        ba, bb = ra.get("B_L_mean"), rb.get("B_L_mean")
        if ba is None or bb is None or not (np.isfinite(ba) and np.isfinite(bb)):
            continue
        b_hi = ba if na > nb else bb          # higher-N_c (less biased) estimate
        b_lo = bb if na > nb else ba
        n_hi, n_lo = max(na, nb), min(na, nb)
        if not (args.bl_min <= abs(b_hi) <= args.bl_max):
            continue
        b_inf = _extrap(n_lo, b_lo, n_hi, b_hi)
        raw = (b_hi - b_lo) / b_hi                       # N_c sensitivity lo->hi
        resid = (b_hi - b_inf) / b_inf if b_inf else float("nan")  # bias left at hi
        by_L[k[0]].append(dict(lam=k[1], zeta=k[2], T=k[3], n_lo=n_lo, n_hi=n_hi,
                               b_lo=b_lo, b_hi=b_hi, b_inf=b_inf,
                               raw_sens=raw, resid_at_hi=resid))

    print("\n" + "=" * 78)
    print("FINITE-N_c BIAS at matched (L,lam,zeta,T)   [crossing-relevant B_L only]")
    print("  raw = (B_hi-B_lo)/B_hi ;  resid = bias still left in the HIGHER-N_c set")
    print("=" * 78)
    print(f"{'L':>4} {'pairs':>6} {'N_c lo/hi':>12} {'med|raw|':>9} {'med|resid_hi|':>13} {'verdict':>14}")
    summary = {}
    for L in sorted(by_L):
        rows = by_L[L]
        nlo = sorted({r["n_lo"] for r in rows}); nhi = sorted({r["n_hi"] for r in rows})
        raws = np.median([abs(r["raw_sens"]) for r in rows])
        resid = np.median([abs(r["resid_at_hi"]) for r in rows if np.isfinite(r["resid_at_hi"])])
        verdict = "clean" if resid < 0.05 else ("watch" if resid < 0.10 else "CONTAMINATED")
        ncs = f"{nlo[0] if nlo else '?'}/{nhi[0] if nhi else '?'}"
        print(f"{L:>4} {len(rows):>6} {ncs:>12} {raws:>8.1%} {resid:>12.1%} {verdict:>14}")
        summary[f"L{L}"] = dict(n_pairs=len(rows), N_c_lo=nlo, N_c_hi=nhi,
                                median_raw_sens=float(raws),
                                median_resid_at_hi=float(resid), verdict=verdict,
                                points=rows)

    worst = max((summary[k]["median_resid_at_hi"] for k in summary), default=0.0)
    print("\n" + "-" * 78)
    if worst < 0.05:
        print("=> Production-N_c B_L is ~unbiased everywhere: existing FSS stands,")
        print("   top-up is just SEEDS at the current N_c.")
    elif worst < 0.10:
        print("=> Mild bias at large L: debias the FSS with the B_inf column below,")
        print("   and prefer adding a 3rd higher N_c over more seeds.")
    else:
        print("=> Significant bias (worst {:.0%}): the single-dataset FSS is likely".format(worst))
        print("   L-dependently contaminated. New compute should add a HIGHER N_c at")
        print("   large L for a clean 1/N_c extrapolation, not just more seeds.")
    print("=" * 78)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"per-point extrapolations (B_inf) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
