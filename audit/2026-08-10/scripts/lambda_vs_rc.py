"""Is the lambda_c-vs-r_c exponent gap physics or the Jacobian of a nonlinear map?
d ln r / d ln lam = 1/(1-lam). Over a finite range where lam_c is NOT small,
phi_r = phi_lam * <1/(1-lam)>_weighted, with equality of exponents only as lam->0.
Also: extract the zeta=1 Born crossing from the LEGACY (non-guided) aggregates."""
import pickle, os, itertools
import numpy as np
from collections import defaultdict

def load(p):
    o = pickle.load(open(os.path.expanduser(p),'rb'))
    return list(o.values()) if isinstance(o,dict) else o

def curves(recs, key='B_L_mean'):
    by = defaultdict(list)
    for r in recs:
        v = r.get(key)
        if v is None or not np.isfinite(v): continue
        by[(round(float(r['zeta']),4), int(r['L']))].append((float(r['lam']), float(v)))
    return by

def cross(c1,c2):
    c1=np.array(sorted(c1)); c2=np.array(sorted(c2))
    lo,hi = max(c1[0,0],c2[0,0]), min(c1[-1,0],c2[-1,0])
    if hi-lo < 1e-9: return None
    g=np.linspace(lo,hi,400)
    d=np.interp(g,c1[:,0],c1[:,1])-np.interp(g,c2[:,0],c2[:,1])
    s=np.where(np.sign(d[:-1])*np.sign(d[1:])<0)[0]
    if not len(s): return None
    i=s[-1]
    return float(g[i]-d[i]*(g[i+1]-g[i])/(d[i+1]-d[i]))

def lc_of(by, wide=True):
    out={}
    for z in sorted({zz for zz,_ in by}):
        Ls=sorted(L for zz,L in by if zz==z)
        xs=[]
        for L1,L2 in itertools.combinations(Ls,2):
            if wide and L2 < 2*L1: continue
            x=cross(by[(z,L1)],by[(z,L2)])
            if x is not None and 0.005<x<0.9: xs.append(x)
        if xs: out[z]=float(np.median(xs))
    return out

def phi(zs, ys):
    p=np.polyfit(np.log(zs), np.log(ys), 1); return p[0]

print("### A. Guided combined (Case B) -- identical crossings, identical procedure")
lc = lc_of(curves(load('~/Downloads/01_M1_Internship/Data/pps_aggregates/agg_caseB_combined.pkl')))
zs=np.array(sorted(lc)); ls=np.array([lc[z] for z in zs]); rs=ls/(1-ls)
print(f"{'zeta window':>18} {'n':>3} {'phi(lam_c)':>11} {'phi(r_c)':>9} {'ratio':>7} {'<1/(1-lam)>':>12}")
for lo in (0.05, 0.05, 0.05, 0.10, 0.15):
    pass
for lo,hi in [(0.05,0.85),(0.05,0.40),(0.05,0.20),(0.05,0.125),(0.15,0.85),(0.25,0.85)]:
    m=(zs>=lo-1e-9)&(zs<=hi+1e-9)
    if m.sum()<3: continue
    pl, pr = phi(zs[m],ls[m]), phi(zs[m],rs[m])
    jac = float(np.mean(1/(1-ls[m])))
    print(f"  [{lo:.3f},{hi:.2f}] {m.sum():>3} {pl:>11.3f} {pr:>9.3f} {pr/pl:>7.3f} {jac:>12.3f}")
print("\n  PREDICTION: ratio phi_r/phi_lam should track <1/(1-lam_c)> and -> 1 as zeta -> 0.")

print("\n### B. zeta=1 Born endpoint from LEGACY (pre-guided) aggregates")
for f in ['clone_aggregate_dense_full.pkl','clone_aggregate(2).pkl','ladder_fss_ready.pkl']:
    try: recs = load(f'~/Downloads/01_M1_Internship/Data/old_cloning_data/{f}')
    except Exception as e: print(f"  {f}: {e}"); continue
    by = curves(recs)
    Ls = sorted({L for zz,L in by if abs(zz-1.0)<1e-6})
    if not Ls: print(f"  {f}: no zeta=1"); continue
    xs=[]
    for L1,L2 in itertools.combinations(Ls,2):
        if L2 < 2*L1: continue
        x=cross(by[(1.0,L1)],by[(1.0,L2)])
        if x is not None and 0.2<x<0.8: xs.append((L1,L2,round(x,4)))
    print(f"  {f}: L at zeta=1 = {Ls}")
    print(f"     wide-pair crossings: {xs}")
    if xs: print(f"     median lambda_c(zeta=1) = {np.median([x[2] for x in xs]):.4f}")
