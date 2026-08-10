"""T0 READ-ONLY probe 3: parametric bootstrap on the stored standard errors.
Deterministic seed 777. Gives a statistical uncertainty on A and phi.
"""
import pickle, os, itertools, csv
import numpy as np
from collections import defaultdict
REPO="/Users/catlover1337/Documents/ppsQJ_m2"; DATA="/Users/catlover1337/Downloads/01_M1_Internship"
RNG=np.random.default_rng(777)

def cross(c1,c2):
    c1=np.array(sorted(c1)); c2=np.array(sorted(c2))
    lo,hi=max(c1[0,0],c2[0,0]),min(c1[-1,0],c2[-1,0])
    if hi-lo<1e-9: return None
    g=np.linspace(lo,hi,400)
    d=np.interp(g,c1[:,0],c1[:,1])-np.interp(g,c2[:,0],c2[:,1])
    s=np.where(np.sign(d[:-1])*np.sign(d[1:])<0)[0]
    if len(s)==0: return None
    i=s[-1]; return float(g[i]-d[i]*(g[i+1]-g[i])/(d[i+1]-d[i]))

def build(recs,f,fe):
    by=defaultdict(list)
    for r in recs:
        v,e=r.get(f),r.get(fe)
        try: v=float(v); e=float(e) if e is not None else 0.0
        except Exception: continue
        if not np.isfinite(v): continue
        if not np.isfinite(e): e=0.0
        by[(round(float(r['zeta']),4),int(r['L']))].append((float(r['lam']),v,e))
    return by

def lamc(by,jitter,wide=True,minp=3):
    out={}
    for z in sorted({z for z,_ in by}):
        Ls=sorted(L for zz,L in by if zz==z and len(by[(zz,L)])>=minp)
        xs=[]
        for L1,L2 in itertools.combinations(Ls,2):
            if wide and L2<2*L1: continue
            c1=[(l,v+(RNG.normal()*e if jitter else 0)) for l,v,e in by[(z,L1)]]
            c2=[(l,v+(RNG.normal()*e if jitter else 0)) for l,v,e in by[(z,L2)]]
            x=cross(c1,c2)
            if x is not None and x>0.005: xs.append(x)
        if xs: out[z]=float(np.median(xs))
    return out

def amp(lc,w):
    z=np.array([k for k in sorted(lc) if w[0]-1e-9<=k<=w[1]+1e-9])
    if len(z)<3: return None
    l=np.array([lc[k] for k in z]); r=l/(1-l)
    p=np.polyfit(np.log(z),np.log(l),1); q=np.polyfit(np.log(z),np.log(r),1)
    return dict(A_lam_half=float(np.mean(l/np.sqrt(z))),phi_lam=float(p[0]),
                A_rc_half=float(np.mean(r/np.sqrt(z))),phi_rc=float(q[0]),
                A_rc_free=float(np.exp(q[1])))

def run(name,by,B=300):
    print(f"\n--- {name}")
    base=lamc(by,False)
    for w in [(0.05,0.4),(0.05,0.85),(0.25,0.85)]:
        pt=amp(base,w)
        if pt is None: continue
        acc=defaultdict(list)
        for _ in range(B):
            a=amp(lamc(by,True),w)
            if a:
                for k,v in a.items(): acc[k].append(v)
        s={k:(float(np.mean(v)),float(np.std(v))) for k,v in acc.items()}
        print(f"  win={w}: "+"  ".join(f"{k}={pt[k]:.3f}+/-{s[k][1]:.3f}" for k in pt))

recs=pickle.load(open(os.path.join(DATA,'Data/pps_aggregates/agg_caseB_combined.pkl'),'rb'))
run("agg_caseB_combined B_L_mean (OBS-BLPROD-001), wide pairs", build(recs,'B_L_mean','B_L_err'))
run("agg_caseB_combined CMI_mean (OBS-CMI-001), wide pairs",    build(recs,'CMI_mean','CMI_err'))

rows=[]
with open(os.path.join(REPO,'results/boundary_aggregate.csv')) as f:
    for d in csv.DictReader(f):
        rows.append(dict(L=int(d['L']),zeta=float(d['zeta']),lam=float(d['lambda']),
                         B_L=float(d['B_L']),B_L_se=float(d['B_L_se']),
                         CMI=float(d['CMI']),CMI_se=float(d['CMI_se'])))
run("boundary_aggregate.csv B_L (OBS-BLPROD-001), wide pairs", build(rows,'B_L','B_L_se'))
run("boundary_aggregate.csv CMI (OBS-CMI-001), wide pairs",    build(rows,'CMI','CMI_se'))
