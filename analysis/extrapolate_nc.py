#!/usr/bin/env python
"""Per-point 1/N_c extrapolation -> debiased B_inf, merged into an FSS-ready set.

The ladder gives B_L at L=128 for N_c in {250,500,800} (the rung pkls from
aggregate_ladder).  For each (L,lam,zeta) this fits B(N_c) = B_inf + b/N_c
(weighted by 1/err^2) and takes B_inf as the N_c->inf value.  With >=3 rungs
it also reports the fit residual so you can SEE whether the bias is really
~1/N_c (clean) or worse (ESS-collapse curvature -> residual systematic).

It then merges B_inf at L=128 with the existing CLEAN low-L points (default
L in {32,64} from the dense aggregate) into one pkl that scaling_form /
debias_collapse can consume directly for the (32,64,128) FSS.

    python analysis/extrapolate_nc.py \
        --rungs /scratch/$USER/pps_qj/ladder_nc250.pkl \
                /scratch/$USER/pps_qj/ladder_nc500.pkl \
                /scratch/$USER/pps_qj/ladder_nc800.pkl \
        --lowL  /Users/.../clone_aggregate_dense_partial.pkl \
        --lowL-Ls 32,64 --out /scratch/$USER/pps_qj/ladder_fss_ready.pkl
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def _bl(rec):
    for k in ("B_L_mean", "B_L"):
        if k in rec and np.isfinite(rec[k]):
            return float(rec[k])
    return np.nan


def extrapolate(rung_paths, field="B_L_mean", errfield="B_L_err"):
    """-> {(L,lam,zeta): rec with B_inf, B_inf_err, n_rungs, resid_frac}."""
    # gather per key: list of (N_c, value, err)
    pts = defaultdict(list)
    meta = {}
    for p in rung_paths:
        for key, rec in _load(p).items():
            v = rec.get(field, np.nan)
            if not np.isfinite(v):
                continue
            nc = float(rec.get("N_c", np.nan))
            e = float(rec.get(errfield, np.nan))
            if not np.isfinite(e) or e <= 0:
                e = abs(v) * 0.1 + 1e-6
            pts[key].append((nc, float(v), e))
            meta[key] = rec
    out = {}
    for key, rows in pts.items():
        rows = sorted(rows)
        ncs = np.array([r[0] for r in rows]); vs = np.array([r[1] for r in rows])
        es = np.array([r[2] for r in rows])
        rec = dict(meta[key]); rec["n_rungs"] = len(rows)
        if len(rows) == 1:
            rec["B_L_mean"] = float(vs[0]); rec["B_L_err"] = float(es[0])
            rec["resid_frac"] = np.nan
        else:
            x = 1.0 / ncs; w = 1.0 / es**2
            # weighted linear fit v = a + b x ; B_inf = a (x->0)
            X = np.vstack([np.ones_like(x), x]).T
            W = np.diag(w)
            cov = np.linalg.inv(X.T @ W @ X)
            beta = cov @ (X.T @ W @ vs)
            a, b = beta
            a_err = float(np.sqrt(cov[0, 0]))
            resid = vs - (a + b * x)
            rec["B_L_mean"] = float(a)
            rec["B_L_err"] = float(max(a_err, np.std(resid) / np.sqrt(len(rows))))
            rec["resid_frac"] = float(np.max(np.abs(resid)) / (abs(a) + 1e-9))
        rec["N_c"] = "inf"
        out[key] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="1/N_c extrapolation + FSS merge")
    ap.add_argument("--rungs", nargs="+", required=True, help="ladder_nc*.pkl")
    ap.add_argument("--lowL", required=True, help="dense aggregate with clean low-L")
    ap.add_argument("--lowL-Ls", default="32,64")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    lowLs = {int(x) for x in a.lowL_Ls.split(",")}

    binf = extrapolate(a.rungs)
    Lset = sorted({k[0] for k in binf})
    print(f"extrapolated {len(binf)} points at L={Lset} to N_c->inf")
    rf = [r["resid_frac"] for r in binf.values() if np.isfinite(r.get("resid_frac", np.nan))]
    if rf:
        print(f"  1/N_c fit residual (frac of B_inf): median={np.median(rf):.3f} max={np.max(rf):.3f}")
        print("  (large residual => bias is NOT clean 1/N_c; ESS-collapse curvature)")

    merged = dict(binf)
    low = _load(a.lowL)
    nlow = 0
    for key, rec in low.items():
        if key[0] in lowLs and np.isfinite(_bl(rec)):
            merged.setdefault(key, rec); nlow += 1
    print(f"  merged {nlow} clean low-L points (L in {sorted(lowLs)})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(merged, f, protocol=4)
    print(f"  FSS-ready set: L={sorted({k[0] for k in merged})}  zeta={sorted({k[2] for k in merged})}")
    print(f"-> {a.out}\nNext: python analysis/scaling_form.py --data {a.out} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
