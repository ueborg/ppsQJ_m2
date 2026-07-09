#!/usr/bin/env python3
"""Pool chi2_worker JSONs by (L,u,c): weighted-pool components, form the connected
chi2, jackknife over shards (error + bias correction). Then fit chi2 ~ L^p per
(u,c) slice (p~1 => x_J=1 / linear-corner mechanism; p~2 => x_J=1/2).
  python analysis/chi2_aggregate.py --outdir DIR"""
import os, sys, json, glob, argparse
import numpy as np
from collections import defaultdict

def chi2_from(Zh1, Ah1, Zh2, Ah2, O0):
    return 2.0 * (Ah2 - Ah1 * Zh1 + O0 * Zh1**2 - O0 * Zh2)

def pool(shards):
    N1 = np.array([s["N1"] for s in shards], float); N2 = np.array([s["N2"] for s in shards], float)
    Zh1 = np.array([s["Zh1"] for s in shards]); Ah1 = np.array([s["Ah1"] for s in shards])
    Zh2 = np.array([s["Zh2"] for s in shards]); Ah2 = np.array([s["Ah2"] for s in shards])
    O0 = float(np.mean([s["O0"] for s in shards]))
    def wp(x, N, mask=None):
        m = np.ones(len(x), bool) if mask is None else mask
        return np.sum(x[m] * N[m]) / np.sum(N[m])
    full = chi2_from(wp(Zh1, N1), wp(Ah1, N1), wp(Zh2, N2), wp(Ah2, N2), O0)
    n = len(shards)
    if n < 2:
        return full, float("nan")
    jk = np.array([chi2_from(wp(Zh1, N1, np.arange(n) != i), wp(Ah1, N1, np.arange(n) != i),
                             wp(Zh2, N2, np.arange(n) != i), wp(Ah2, N2, np.arange(n) != i), O0)
                   for i in range(n)])
    mean_jk = jk.mean(); bias_corr = full - (n - 1) * (mean_jk - full)
    err = np.sqrt((n - 1) / n * np.sum((jk - mean_jk)**2))
    return bias_corr, err

def fit_scaling(pts):
    """pts = [(L, chi2, err)] -> (p, perr, n) for chi2 ~ L^p, weighted log-log."""
    L = np.array([p[0] for p in pts], float); ch = np.array([p[1] for p in pts]); er = np.array([p[2] for p in pts])
    ok = ch > 0
    L, ch, er = L[ok], ch[ok], er[ok]
    if len(L) < 3:
        return None
    w = (ch / er)**2; x = np.log(L); y = np.log(ch)
    W = w.sum(); Sx = (w * x).sum(); Sy = (w * y).sum(); Sxx = (w * x * x).sum(); Sxy = (w * x * y).sum()
    D = W * Sxx - Sx * Sx; p = (W * Sxy - Sx * Sy) / D; b = (Sy - p * Sx) / W
    s2 = np.sum(w * (y - (p * x + b))**2) / (len(L) - 2)
    return p, float(np.sqrt(s2 * W / D)), len(L)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True); a = ap.parse_args()
    groups = {}
    for f in glob.glob(os.path.join(a.outdir, "chi2_L*.json")):
        s = json.load(open(f)); groups.setdefault((int(s["L"]), s["u"], s["c"]), []).append(s)
    print(f"# {'L':>4} {'u':>5} {'c':>4} {'nshard':>6} {'chi2':>10} {'+-err':>9} {'chi2/L':>9} {'chi2/L^2':>10}")
    res = {}
    for (L, u, c) in sorted(groups):
        ch, er = pool(groups[(L, u, c)]); res[(L, u, c)] = (ch, er)
        print(f"  {L:4d} {u:5.2f} {c:4.1f} {len(groups[(L,u,c)]):6d} {ch:10.4f} {er:9.4f} {ch/L:9.4f} {ch/L**2:10.5f}")
    byuc = defaultdict(list)
    for (L, u, c), (ch, er) in res.items():
        byuc[(u, c)].append((L, ch, er))
    print("\n# chi2 ~ L^p per (u,c) slice   [p~1 => x_J=1 ; p~2 => x_J=1/2]")
    for (u, c) in sorted(byuc):
        r = fit_scaling(byuc[(u, c)])
        if r:
            p, pe, n = r
            flag = "  <-- consistent with x_J=1" if abs(p - 1) < 2 * pe else ""
            print(f"  u={u:.2f} c={c:.1f} ({n} L): p = {p:.2f} +/- {pe:.2f}{flag}")
        else:
            print(f"  u={u:.2f} c={c:.1f}: <3 L values, need more")
