"""T0 READ-ONLY probe 5: replicate the ORIGINAL May-2026 recipe exactly as
documented in analysis/lambda_c_phi_analysis.md (provenance, not support):
  L pairs from {32,48,64,96,128}; crossing by linear interpolation;
  extrapolate lambda_c(L_avg) vs 1/L_avg^p for p in {0.5,0.7,1,2};
  log-log fit lambda_c^inf = A zeta^phi restricted to zeta <= 0.3.
Also refits the lambda_c^inf TABLE printed in that document.
Deterministic; no RNG.
"""
import pickle, os, itertools
import numpy as np
from collections import defaultdict
D="/Users/catlover1337/Downloads/01_M1_Internship/Data/old_cloning_data"
LSET={32,48,64,96,128}

def cross(c1,c2):
    c1=np.array(sorted(c1)); c2=np.array(sorted(c2))
    lo,hi=max(c1[0,0],c2[0,0]),min(c1[-1,0],c2[-1,0])
    if hi-lo<1e-9: return None
    g=np.linspace(lo,hi,600)
    d=np.interp(g,c1[:,0],c1[:,1])-np.interp(g,c2[:,0],c2[:,1])
    s=np.where(np.sign(d[:-1])*np.sign(d[1:])<0)[0]
    if len(s)==0: return None
    i=s[-1]; return float(g[i]-d[i]*(g[i+1]-g[i])/(d[i+1]-d[i]))

def loglogfit(z,y):
    z=np.asarray(z,float); y=np.asarray(y,float); m=y>0
    p=np.polyfit(np.log(z[m]),np.log(y[m]),1)
    return float(p[0]),float(np.exp(p[1])),float(np.mean(y[m]/np.sqrt(z[m])))

print("### 1. refit of the lambda_c^inf TABLE printed in analysis/lambda_c_phi_analysis.md")
tab={0.02:0.149,0.05:0.157,0.10:0.251,0.15:0.233,0.20:0.229,0.30:0.594,
     0.50:0.759,0.70:0.487,0.85:0.459,1.00:0.443}
for w in [(0.02,0.2),(0.02,0.3),(0.02,0.5),(0.02,1.01)]:
    ks=[z for z in tab if w[0]-1e-9<=z<=w[1]+1e-9]
    phi,Af,Ah=loglogfit(ks,[tab[z] for z in ks])
    r=[tab[z]/(1-tab[z]) for z in ks]; phir,Afr,Ahr=loglogfit(ks,r)
    print(f"  win={w} n={len(ks)}: lam phi={phi:.3f} A_free={Af:.3f} A(1/2)={Ah:.3f}"
          f" | rc phi={phir:.3f} A_free={Afr:.3f} A(1/2)={Ahr:.3f}")

print("\n### 2. re-run the documented recipe on the old aggregates (L in {32,48,64,96,128})")
for fn in ['clone_aggregate(2).pkl','clone_aggregate_dense_full.pkl']:
    o=pickle.load(open(os.path.join(D,fn),'rb'))
    o=list(o.values()) if isinstance(o,dict) else o
    by=defaultdict(list)
    for r in o:
        v=r.get('B_L_mean')
        if v is None or not np.isfinite(float(v)): continue
        L=int(float(r['L']))
        if L not in LSET: continue
        by[(round(float(r['zeta']),4),L)].append((float(r['lam']),float(v)))
    print(f"\n-- {fn}: L present = {sorted({L for _,L in by})}")
    pts=defaultdict(list)
    for z in sorted({z for z,_ in by}):
        Ls=sorted(L for zz,L in by if zz==z and len(by[(zz,L)])>=3)
        for L1,L2 in itertools.combinations(Ls,2):
            x=cross(by[(z,L1)],by[(z,L2)])
            if x is not None and x>0.005: pts[z].append((0.5*(L1+L2),x))   # L_avg per the doc
    for p in (0.5,0.7,1.0,2.0):
        inf={}
        for z,s in pts.items():
            if len(s)<3: continue
            X=np.array([1.0/(q[0]**p) for q in s]); Y=np.array([q[1] for q in s])
            a,b=np.polyfit(X,Y,1); inf[z]=float(b)
        for w in [(0.02,0.3),(0.02,0.2),(0.02,1.01)]:
            ks=[z for z in sorted(inf) if w[0]-1e-9<=z<=w[1]+1e-9 and inf[z]>0.002]
            if len(ks)<3: continue
            phi,Af,Ah=loglogfit(ks,[inf[z] for z in ks])
            print(f"   p={p:<4} win={w} n={len(ks):2d}: lam phi={phi:.3f} A_free={Af:.3f} A(1/2)={Ah:.3f}")
        if abs(p-0.5)<1e-9:
            print("      lam_c^inf(p=0.5):", {z:round(v,3) for z,v in sorted(inf.items())})
