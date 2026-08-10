"""T0 READ-ONLY probe 2: uncertainty, proper KMR locator, boundary-CSV L-extrap.
Deterministic: bootstrap uses a fixed seed (12345).
"""
import pickle, os, itertools, json, csv
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/catlover1337/Documents/ppsQJ_m2"
DATA = "/Users/catlover1337/Downloads/01_M1_Internship"
RNG  = np.random.default_rng(12345)

def cross(c1, c2):
    c1 = np.array(sorted(c1)); c2 = np.array(sorted(c2))
    lo, hi = max(c1[0,0], c2[0,0]), min(c1[-1,0], c2[-1,0])
    if hi - lo < 1e-9: return None
    g = np.linspace(lo, hi, 400)
    d = np.interp(g, c1[:,0], c1[:,1]) - np.interp(g, c2[:,0], c2[:,1])
    s = np.where(np.sign(d[:-1])*np.sign(d[1:]) < 0)[0]
    if len(s)==0: return None
    i = s[-1]
    return float(g[i] - d[i]*(g[i+1]-g[i])/(d[i+1]-d[i]))

def curves(recs, field):
    by = defaultdict(list)
    for r in recs:
        v = r.get(field)
        if v is None: continue
        try: v=float(v)
        except Exception: continue
        if not np.isfinite(v): continue
        by[(round(float(r['zeta']),4), int(r['L']))].append((float(r['lam']), v))
    return by

def pairs_of(by, wide=True, min_pts=3):
    out=defaultdict(list)
    for z in sorted({z for z,_ in by}):
        Ls=sorted(L for zz,L in by if zz==z and len(by[(zz,L)])>=min_pts)
        for L1,L2 in itertools.combinations(Ls,2):
            if wide and L2 < 2*L1: continue
            x=cross(by[(z,L1)],by[(z,L2)])
            if x is not None and x>0.005: out[z].append((L1,L2,x))
    return out

def amp(zs, lc, window):
    zs=np.asarray(zs); lc=np.asarray(lc)
    m=(zs>=window[0]-1e-9)&(zs<=window[1]+1e-9)
    if m.sum()<3: return None
    z=zs[m]; l=lc[m]; r=l/(1-l)
    p=np.polyfit(np.log(z),np.log(l),1); q=np.polyfit(np.log(z),np.log(r),1)
    return dict(n=int(m.sum()),
                A_lam_half=float(np.mean(l/np.sqrt(z))), phi_lam=float(p[0]), A_lam_free=float(np.exp(p[1])),
                A_rc_half=float(np.mean(r/np.sqrt(z))),  phi_rc=float(q[0]),  A_rc_free=float(np.exp(q[1])))

def boot(pz, window, B=400):
    """bootstrap over the pair set within each zeta"""
    zs=sorted(pz)
    vals=defaultdict(list)
    for _ in range(B):
        lc=[]
        for z in zs:
            xs=[p[2] for p in pz[z]]
            idx=RNG.integers(0,len(xs),len(xs))
            lc.append(float(np.median([xs[i] for i in idx])))
        a=amp(zs,lc,window)
        if a:
            for k,v in a.items():
                if k!='n': vals[k].append(v)
    return {k:(float(np.mean(v)),float(np.std(v))) for k,v in vals.items()}

print("### A. bootstrap uncertainty, agg_caseB_combined, wide pairs")
recs=pickle.load(open(os.path.join(DATA,'Data/pps_aggregates/agg_caseB_combined.pkl'),'rb'))
pz=pairs_of(curves(recs,'B_L_mean'))
zs=sorted(pz); lc=[float(np.median([p[2] for p in pz[z]])) for z in zs]
for w in [(0.05,0.4),(0.05,0.85),(0.25,0.85)]:
    pt=amp(zs,lc,w); bs=boot(pz,w)
    print(f"  win={w} n={pt['n']}")
    for k in ('A_lam_half','phi_lam','A_lam_free','A_rc_half','phi_rc','A_rc_free'):
        print(f"     {k:<12} point={pt[k]:.3f}  boot={bs[k][0]:.3f} +/- {bs[k][1]:.3f}")

print("\n### B. OBS-BLKMR-001 done per realisation (product of clone-averages, then mean over reals)")
r2=pickle.load(open(os.path.join(DATA,'Data/pps_aggregates/agg_pps_clone_guided_prod.pkl'),'rb'))
print("   keys with means_all:", [k for k in r2[0] if k.endswith('means_all')])
for rr in r2:
    try:
        cm=np.asarray(rr['CMI_means_all'],dtype=float); sm=np.asarray(rr['S_AB_means_all'],dtype=float)
        v=cm*sm; v=v[np.isfinite(v)]
        rr['B_KMR_perreal']=float(np.mean(v)) if v.size else None
    except Exception:
        rr['B_KMR_perreal']=None
    try:
        rr['B_KMR_agg']=float(rr['CMI_mean'])*float(rr['S_AB_mean'])
    except Exception:
        rr['B_KMR_agg']=None
tab={}
for field in ('B_L_mean','B_KMR_perreal','B_KMR_agg','CMI_mean'):
    p=pairs_of(curves(r2,field))
    z2=sorted(p); l2=[float(np.median([q[2] for q in p[z]])) for z in z2]
    tab[field]=dict(zip(z2,l2))
    a=amp(z2,l2,(0.05,0.85))
    print(f"   {field:<15} A_lam(1/2)={a['A_lam_half']:.4f} phi_lam={a['phi_lam']:.4f} "
          f"A_rc(1/2)={a['A_rc_half']:.4f} phi_rc={a['phi_rc']:.4f}")
zc=sorted(set(tab['B_L_mean'])&set(tab['B_KMR_perreal']))
d=[abs(tab['B_L_mean'][z]-tab['B_KMR_perreal'][z])/tab['B_L_mean'][z] for z in zc]
print(f"   max |lam_c(BLPROD)-lam_c(BLKMR)|/lam_c over {len(zc)} zetas = {max(d)*100:.2f}%  median {np.median(d)*100:.2f}%")

print("\n### C. boundary_aggregate.csv: bootstrap + L-extrapolation")
rows=[]
with open(os.path.join(REPO,'results/boundary_aggregate.csv')) as f:
    for dd in csv.DictReader(f):
        rows.append(dict(L=int(dd['L']),zeta=float(dd['zeta']),lam=float(dd['lambda']),
                         B_L_mean=float(dd['B_L']),CMI_mean=float(dd['CMI'])))
byb=curves(rows,'B_L_mean')
pb=pairs_of(byb)
zb=sorted(pb); lb=[float(np.median([p[2] for p in pb[z]])) for z in zb]
for w in [(0.05,0.4),(0.05,0.85),(0.25,0.85)]:
    pt=amp(zb,lb,w); bs=boot(pb,w)
    print(f"  win={w} n={pt['n']}")
    for k in ('A_lam_half','phi_lam','A_rc_half','phi_rc','A_rc_free'):
        print(f"     {k:<12} point={pt[k]:.3f}  boot={bs[k][0]:.3f} +/- {bs[k][1]:.3f}")
pball=pairs_of(byb,wide=False)
print("  L-extrap (all pairs, 1/sqrt(Lgeom)) on boundary csv:")
ex={}
for z in sorted(pball):
    s=pball[z]
    if len(s)<3: continue
    x=np.array([1/np.sqrt(np.sqrt(p[0]*p[1])) for p in s]); y=np.array([p[2] for p in s])
    a,b=np.polyfit(x,y,1); ex[z]=float(b)
    print(f"    zeta={z:<5} n={len(s):2d} median={np.median(y):.4f} Linf={b:.4f}")
exz=[z for z in sorted(ex) if ex[z]>0.002]
for w in [(0.05,0.4),(0.05,0.85),(0.25,0.85)]:
    a=amp(exz,[ex[z] for z in exz],w)
    if a: print(f"    Lextrap win={w}: A_lam(1/2)={a['A_lam_half']:.3f} phi_lam={a['phi_lam']:.3f} "
                f"A_rc(1/2)={a['A_rc_half']:.3f} phi_rc={a['phi_rc']:.3f}")

print("\n### D. small-zeta floor check (lam_c vs L at fixed small zeta), boundary csv")
for z in (0.05,0.1,0.15):
    line=[]
    for (zz,L),c in sorted(byb.items()):
        if zz!=z: continue
        c=np.array(sorted(c))
        line.append((L,len(c)))
    print(f"   zeta={z}: (L,n_lambda)={line}")
    s=pball.get(z,[])
    print(f"      pair crossings: {[(p[0],p[1],round(p[2],4)) for p in s]}")
