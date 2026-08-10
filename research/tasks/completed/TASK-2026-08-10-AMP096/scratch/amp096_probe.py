"""T0 READ-ONLY probe: where could A = 0.96 have come from?

Tests, on data already on disk:
  (1) baseline reproduction of the audit amplitude (OBS-BLPROD-001 locator)
  (2) window scan of A and phi in lambda_c and r_c parameterizations
  (3) L-extrapolation of the crossings (the historical 0.96 was reported as a
      1/sqrt(L) extrapolation) -- does extrapolation move A toward 0.96?
  (4) locator dependence: OBS-BLPROD-001 vs OBS-BLKMR-001 (<CMI>*<S_AB>)
  (5) independent dataset: results/boundary_aggregate.csv (July campaign)

Writes nothing outside this scratch dir. No RNG used; fully deterministic.
"""
import pickle, os, itertools, json, csv, sys
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/catlover1337/Documents/ppsQJ_m2"          # from data_roots.local.yaml: REPO
DATA = "/Users/catlover1337/Downloads/01_M1_Internship"  # DATA_INTERNSHIP

def cross(c1, c2):
    c1 = np.array(sorted(c1)); c2 = np.array(sorted(c2))
    lo, hi = max(c1[0,0], c2[0,0]), min(c1[-1,0], c2[-1,0])
    if hi - lo < 1e-9: return None
    g = np.linspace(lo, hi, 400)
    d = np.interp(g, c1[:,0], c1[:,1]) - np.interp(g, c2[:,0], c2[:,1])
    s = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if len(s) == 0: return None
    i = s[-1]
    return float(g[i] - d[i]*(g[i+1]-g[i])/(d[i+1]-d[i]))

def curves(recs, field):
    by = defaultdict(list)
    for r in recs:
        v = r.get(field)
        if v is None: continue
        try: v = float(v)
        except Exception: continue
        if not np.isfinite(v): continue
        by[(round(float(r['zeta']),4), int(r['L']))].append((float(r['lam']), v))
    return by

def lamc_wide(by, min_pts=3):
    """median wide-pair crossing per zeta, plus the per-pair table"""
    zetas = sorted({z for z,_ in by})
    out, pairs = {}, []
    for z in zetas:
        Ls = sorted(L for zz,L in by if zz==z and len(by[(zz,L)])>=min_pts)
        xs = []
        for L1,L2 in itertools.combinations(Ls,2):
            if L2 < 2*L1: continue
            x = cross(by[(z,L1)], by[(z,L2)])
            if x is not None and x > 0.005:
                xs.append(x); pairs.append((z,L1,L2,x))
        if xs: out[z] = float(np.median(xs))
    return out, pairs

def lamc_allpairs(by, min_pts=3):
    zetas = sorted({z for z,_ in by})
    pairs = []
    for z in zetas:
        Ls = sorted(L for zz,L in by if zz==z and len(by[(zz,L)])>=min_pts)
        for L1,L2 in itertools.combinations(Ls,2):
            x = cross(by[(z,L1)], by[(z,L2)])
            if x is not None and x > 0.005:
                pairs.append((z,L1,L2,x))
    return pairs

def fit(zs, y):
    zs = np.asarray(zs); y = np.asarray(y)
    A_half = float(np.mean(y/np.sqrt(zs)))
    p = np.polyfit(np.log(zs), np.log(y), 1)
    return dict(A_fixed_half=A_half, phi_free=float(p[0]), A_free=float(np.exp(p[1])), n=len(zs))

WINDOWS = [(0.05,0.125),(0.05,0.2),(0.05,0.40),(0.05,0.85),(0.10,0.85),(0.25,0.85),(0.05,1.01)]

def window_scan(lc, tag, out):
    zs_all = np.array(sorted(lc)); ls_all = np.array([lc[z] for z in zs_all])
    for lo,hi in WINDOWS:
        m = (zs_all>=lo-1e-9)&(zs_all<=hi+1e-9)
        if m.sum() < 3: continue
        zs = zs_all[m]; ls = ls_all[m]; rs = ls/(1-ls)
        rec = dict(tag=tag, window=[lo,hi], n=int(m.sum()),
                   lam=fit(zs,ls), rc=fit(zs,rs))
        out.append(rec)
        print(f"  {tag:<22} win=[{lo},{hi}] n={m.sum():2d} | "
              f"lam: A(1/2)={rec['lam']['A_fixed_half']:.3f} phi={rec['lam']['phi_free']:.3f} "
              f"A_free={rec['lam']['A_free']:.3f} | "
              f"rc: A(1/2)={rec['rc']['A_fixed_half']:.3f} phi={rec['rc']['phi_free']:.3f} "
              f"A_free={rec['rc']['A_free']:.3f}")

results = {}
scan = []

# ---------------------------------------------------------------- (1)(2)
P = os.path.join(DATA,'Data/pps_aggregates/agg_caseB_combined.pkl')
recs = pickle.load(open(P,'rb'))
by = curves(recs,'B_L_mean')
lc, wpairs = lamc_wide(by)
print("== agg_caseB_combined, OBS-BLPROD-001 (B_L_mean), wide pairs, median")
for z in sorted(lc): print(f"   zeta={z:<6} lam_c={lc[z]:.4f}  r_c={lc[z]/(1-lc[z]):.4f}")
window_scan(lc,'BLPROD_widepair',scan)
results['blprod_widepair_lamc'] = lc

# ---------------------------------------------------------------- (3) L-extrap
print("\n== L-extrapolation test (all pairs, x vs 1/sqrt(L_eff)) ==")
allp = lamc_allpairs(by)
ext = {}
for z in sorted({p[0] for p in allp}):
    sel = [p for p in allp if p[0]==z]
    if len(sel) < 3: continue
    # L_eff = geometric mean of the pair
    x = np.array([1.0/np.sqrt(np.sqrt(p[1]*p[2])) for p in sel])
    y = np.array([p[3] for p in sel])
    a,b = np.polyfit(x,y,1)   # y = a*x + b ; b = L->inf limit
    ext[z] = float(b)
    print(f"   zeta={z:<6} n_pairs={len(sel):2d}  median={np.median(y):.4f}  Linf={b:.4f}")
ext_pos = {z:v for z,v in ext.items() if v>0.002}
if len(ext_pos)>=3:
    window_scan(ext_pos,'BLPROD_Lextrap',scan)
results['blprod_Lextrap_lamc'] = ext

# also: single widest pair only, and narrow pairs only
print("\n== estimator sensitivity: pair selection ==")
for name, pred in [('all_pairs', lambda L1,L2: True),
                   ('narrow_only', lambda L1,L2: L2 < 2*L1),
                   ('widest_pair', None)]:
    lc2 = {}
    for z in sorted({p[0] for p in allp}):
        sel=[p for p in allp if p[0]==z]
        if pred is None:
            sel = sorted(sel, key=lambda p: p[2]/p[1])[-1:]
        else:
            sel=[p for p in sel if pred(p[1],p[2])]
        if sel: lc2[z]=float(np.median([p[3] for p in sel]))
    if len(lc2)>=3: window_scan(lc2,f'BLPROD_{name}',scan)

# ---------------------------------------------------------------- (4) KMR locator
print("\n== locator dependence: OBS-BLPROD-001 vs OBS-BLKMR-001 ==")
P2 = os.path.join(DATA,'Data/pps_aggregates/agg_pps_clone_guided_prod.pkl')
r2 = pickle.load(open(P2,'rb'))
print(f"   agg_pps_clone_guided_prod.pkl n={len(r2)}  has S_AB_mean={'S_AB_mean' in r2[0]}")
for rr in r2:
    try:
        rr['B_KMR'] = float(rr['CMI_mean'])*float(rr['S_AB_mean'])
    except Exception:
        rr['B_KMR'] = None
for field,tag in [('B_L_mean','BLPROD_prodagg'),('B_KMR','BLKMR_prodagg'),('CMI_mean','CMI_prodagg')]:
    byf = curves(r2, field)
    lcf,_ = lamc_wide(byf)
    if len(lcf)>=3:
        print(f"  -- {tag}: zetas={sorted(lcf)}")
        window_scan(lcf,tag,scan)
        results[tag]=lcf

# ---------------------------------------------------------------- (5) boundary csv
print("\n== independent dataset: results/boundary_aggregate.csv ==")
rows=[]
with open(os.path.join(REPO,'results/boundary_aggregate.csv')) as f:
    for d in csv.DictReader(f):
        rows.append(dict(L=int(d['L']), zeta=float(d['zeta']), lam=float(d['lambda']),
                         B_L_mean=float(d['B_L']), CMI_mean=float(d['CMI'])))
for field,tag in [('B_L_mean','BLPROD_boundarycsv'),('CMI_mean','CMI_boundarycsv')]:
    byb = curves(rows, field)
    npts = {k:len(v) for k,v in byb.items()}
    lcb, wp = lamc_wide(byb)
    print(f"  -- {tag}: cells={len(byb)} cells_with_ge3_lam={sum(1 for v in npts.values() if v>=3)} "
          f"zetas_with_crossing={sorted(lcb)}")
    for z in sorted(lcb): print(f"       zeta={z:<5} lam_c={lcb[z]:.4f}")
    if len(lcb)>=3: window_scan(lcb,tag,scan)
    results[tag]=lcb
    # relaxed: any pair (not just wide)
    lcb2={}
    for z in sorted({z for z,_ in byb}):
        Ls=sorted(L for zz,L in byb if zz==z and len(byb[(zz,L)])>=3)
        xs=[cross(byb[(z,a)],byb[(z,b)]) for a,b in itertools.combinations(Ls,2)]
        xs=[x for x in xs if x is not None and x>0.005]
        if xs: lcb2[z]=float(np.median(xs))
    print(f"     relaxed(any pair) zetas={sorted(lcb2)}")
    if len(lcb2)>=3: window_scan(lcb2,tag+'_anypair',scan)
    results[tag+'_anypair']=lcb2

json.dump({'scan':scan,'lamc':{k:{str(z):v for z,v in d.items()} for k,d in results.items()}},
          open(os.path.join(HERE,'amp096_probe_out.json'),'w'), indent=1)
print("\nwrote", os.path.join(HERE,'amp096_probe_out.json'))
