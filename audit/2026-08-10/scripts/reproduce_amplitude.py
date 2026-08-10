"""Audit-grade independent reproduction of the lambda_c(zeta) amplitude.
READ-ONLY. Binder B_L crossings from the guided combined Case B aggregate.
NOT a production estimator: no bootstrap, no drift errors. Purpose is to
check which parameterization the amplitude 0.96 vs 0.51 belongs to."""
import pickle, os, itertools
import numpy as np
from collections import defaultdict

P = os.path.expanduser('~/Downloads/01_M1_Internship/Data/pps_aggregates/agg_caseB_combined.pkl')
recs = pickle.load(open(P,'rb'))
by = defaultdict(list)
for r in recs:
    if r.get('B_L_mean') is not None and np.isfinite(r['B_L_mean']):
        by[(round(float(r['zeta']),4), int(r['L']))].append((float(r['lam']), float(r['B_L_mean'])))

def cross(c1, c2):
    """crossing of two B_L(lambda) curves on their overlap, linear interp"""
    c1 = np.array(sorted(c1)); c2 = np.array(sorted(c2))
    lo, hi = max(c1[0,0], c2[0,0]), min(c1[-1,0], c2[-1,0])
    if hi - lo < 1e-9: return None
    g = np.linspace(lo, hi, 400)
    d = np.interp(g, c1[:,0], c1[:,1]) - np.interp(g, c2[:,0], c2[:,1])
    s = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if len(s) == 0: return None
    i = s[-1]
    return float(g[i] - d[i]*(g[i+1]-g[i])/(d[i+1]-d[i]))

zetas = sorted({z for z,_ in by})
print(f"{'zeta':>7} {'pairs':>28} {'lam_c(med)':>11} {'r_c':>8}")
lc = {}
for z in zetas:
    Ls = sorted(L for zz,L in by if zz==z)
    xs = []
    for L1,L2 in itertools.combinations(Ls,2):
        if L2 < 2*L1: continue                      # wide pairs only
        x = cross(by[(z,L1)], by[(z,L2)])
        if x is not None and x > 0.005: xs.append(x)
    if xs:
        m = float(np.median(xs)); lc[z] = m
        print(f"{z:>7} {str(Ls):>28} {m:>11.4f} {m/(1-m):>8.4f}")

zs = np.array(sorted(lc)); ls = np.array([lc[z] for z in zs]); rs = ls/(1-ls)
def fit(y, label):
    A_sqrt = float(np.mean(y/np.sqrt(zs)))
    p = np.polyfit(np.log(zs), np.log(y), 1)
    print(f"  {label:>6}:  A(fixed phi=1/2) = {A_sqrt:.3f}   |   free power: phi={p[0]:.3f}, A={np.exp(p[1]):.3f}")
print(f"\nFits over zeta in [{zs.min()}, {zs.max()}], n={len(zs)}:")
fit(ls, 'lam_c'); fit(rs, 'r_c')

print("\n=== zeta=1 coverage check across ALL aggregates ===")
import glob
for p in sorted(glob.glob(os.path.expanduser('~/Downloads/01_M1_Internship/Data/**/*.pkl'), recursive=True)):
    try: o = pickle.load(open(p,'rb'))
    except Exception: continue
    o = list(o.values()) if isinstance(o,dict) else o
    if not (isinstance(o,list) and o and isinstance(o[0],dict)): continue
    zs2 = {round(float(r['zeta']),3) for r in o if 'zeta' in r}
    print(f"  {os.path.basename(p):<38} n={len(o):>5}  zeta_max={max(zs2) if zs2 else None}  has_zeta1={1.0 in zs2}")
