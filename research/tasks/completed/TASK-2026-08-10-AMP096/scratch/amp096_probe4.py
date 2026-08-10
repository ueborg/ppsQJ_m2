"""T0 READ-ONLY probe 4: can the historical A = 0.96 be regenerated?

Hypothesis under test: the May-2026 number came from a 1/sqrt(L) extrapolation
of Binder crossings on the OLD cloning aggregate, fitted over a SMALL-zeta window,
in the lambda_c parameterization. Test it on the old aggregates on disk.
Deterministic; no RNG.
"""
import pickle, os, itertools
import numpy as np
from collections import defaultdict
D="/Users/catlover1337/Downloads/01_M1_Internship/Data/old_cloning_data"

def cross(c1,c2):
    c1=np.array(sorted(c1)); c2=np.array(sorted(c2))
    lo,hi=max(c1[0,0],c2[0,0]),min(c1[-1,0],c2[-1,0])
    if hi-lo<1e-9: return None
    g=np.linspace(lo,hi,600)
    d=np.interp(g,c1[:,0],c1[:,1])-np.interp(g,c2[:,0],c2[:,1])
    s=np.where(np.sign(d[:-1])*np.sign(d[1:])<0)[0]
    if len(s)==0: return None
    i=s[-1]; return float(g[i]-d[i]*(g[i+1]-g[i])/(d[i+1]-d[i]))

def load(fn):
    o=pickle.load(open(os.path.join(D,fn),'rb'))
    o=list(o.values()) if isinstance(o,dict) else o
    by=defaultdict(list)
    for r in o:
        v=r.get('B_L_mean')
        if v is None or not np.isfinite(float(v)): continue
        by[(round(float(r['zeta']),4),int(float(r['L'])))].append((float(r['lam']),float(v)))
    return by

def fitlog(z,y):
    z=np.asarray(z); y=np.asarray(y); m=(y>0)
    p=np.polyfit(np.log(z[m]),np.log(y[m]),1)
    return float(p[0]),float(np.exp(p[1])),float(np.mean(y[m]/np.sqrt(z[m])))

for fn in ['clone_aggregate(2).pkl','clone_aggregate_dense_full.pkl']:
    by=load(fn)
    print(f"\n===== {fn}")
    zs=sorted({z for z,_ in by})
    tab={}
    for z in zs:
        Ls=sorted(L for zz,L in by if zz==z and len(by[(zz,L)])>=3)
        pts=[]
        for L1,L2 in itertools.combinations(Ls,2):
            x=cross(by[(z,L1)],by[(z,L2)])
            if x is not None and x>0.005: pts.append((np.sqrt(L1*L2),x))
        if len(pts)<3: continue
        Le=np.array([p[0] for p in pts]); x=np.array([p[1] for p in pts])
        a,b=np.polyfit(1/np.sqrt(Le),x,1)      # 1/sqrt(L) extrapolation
        a2,b2=np.polyfit(1/Le,x,1)             # 1/L extrapolation
        big=[p[1] for p in pts if p[0]>=max(Le)*0.8]
        tab[z]=dict(med=float(np.median(x)), inf_sqrt=float(b), inf_lin=float(b2),
                    largest=float(np.median(big)), n=len(pts), Ls=Ls)
        print(f"  zeta={z:<5} n={len(pts):2d} Ls={Ls} med={np.median(x):.4f} "
              f"1/sqrtL->{b:.4f}  1/L->{b2:.4f}")
    for est in ('med','inf_sqrt','inf_lin','largest'):
        for win in [(0.02,0.2),(0.02,0.3),(0.02,0.5),(0.02,1.01),(0.05,0.3)]:
            ks=[z for z in tab if win[0]-1e-9<=z<=win[1]+1e-9 and tab[z][est]>0.002]
            if len(ks)<3: continue
            y=np.array([tab[z][est] for z in ks]); zz=np.array(ks)
            phi,Af,Ah=fitlog(zz,y)
            r=y/(1-y); phir,Afr,Ahr=fitlog(zz,r)
            print(f"   [{est:<9} win={win}] n={len(ks):2d} lam: phi={phi:.3f} A_free={Af:.3f} A(1/2)={Ah:.3f}"
                  f" | rc: phi={phir:.3f} A_free={Afr:.3f} A(1/2)={Ahr:.3f}")
