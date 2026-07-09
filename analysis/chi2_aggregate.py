#!/usr/bin/env python3
"""Pool chi2_worker JSONs by (L,u,c): weighted-pool the components, form the
connected chi2, jackknife over shards (seeds) for error + bias correction.
Prints the chi2/L vs L scaling diagnostic (x_J=1 -> flat; x_J=1/2 -> chi2/L^2 flat).
  python analysis/chi2_aggregate.py --outdir DIR"""
import os, sys, json, glob, argparse
import numpy as np

def chi2_from(Zh1, Ah1, Zh2, Ah2, O0):
    return 2.0 * (Ah2 - Ah1 * Zh1 + O0 * Zh1**2 - O0 * Zh2)

def pool(shards):
    N1 = np.array([s["N1"] for s in shards], float); N2 = np.array([s["N2"] for s in shards], float)
    Zh1 = np.array([s["Zh1"] for s in shards]); Ah1 = np.array([s["Ah1"] for s in shards])
    Zh2 = np.array([s["Zh2"] for s in shards]); Ah2 = np.array([s["Ah2"] for s in shards])
    O0 = float(np.mean([s["O0"] for s in shards]))
    def wp(x, N, mask=None):
        if mask is None: return np.sum(x * N) / np.sum(N)
        return np.sum(x[mask] * N[mask]) / np.sum(N[mask])
    full = chi2_from(wp(Zh1, N1), wp(Ah1, N1), wp(Zh2, N2), wp(Ah2, N2), O0)
    n = len(shards)
    if n < 2: return full, float("nan")
    jk = np.array([chi2_from(wp(Zh1, N1, np.arange(n) != i), wp(Ah1, N1, np.arange(n) != i),
                             wp(Zh2, N2, np.arange(n) != i), wp(Ah2, N2, np.arange(n) != i), O0)
                   for i in range(n)])
    mean_jk = jk.mean(); bias_corr = full - (n - 1) * (mean_jk - full)
    err = np.sqrt((n - 1) / n * np.sum((jk - mean_jk)**2))
    return bias_corr, err

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True); a = ap.parse_args()
    groups = {}
    for f in glob.glob(os.path.join(a.outdir, "chi2_L*.json")):
        s = json.load(open(f)); groups.setdefault((int(s["L"]), s["u"], s["c"]), []).append(s)
    print(f"# {'L':>4} {'u':>5} {'c':>4} {'nshard':>6} {'chi2':>10} {'+-err':>9} {'chi2/L':>9} {'chi2/L^2':>10}")
    for (L, u, c) in sorted(groups):
        sh = groups[(L, u, c)]; ch, er = pool(sh)
        print(f"  {L:4d} {u:5.2f} {c:4.1f} {len(sh):6d} {ch:10.4f} {er:9.4f} {ch/L:9.4f} {ch/L**2:10.5f}")
