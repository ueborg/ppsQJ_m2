#!/usr/bin/env python3
"""Score adaptive-resampling outputs by final observables; GESS is diagnostic."""
import os,json,argparse,math
from collections import defaultdict
import numpy as np

def load(root):
    out=[]
    for d,_,fs in os.walk(root):
        for f in fs:
            if not f.endswith('.json'):continue
            try:r=json.load(open(os.path.join(d,f)))
            except Exception:continue
            if r.get('status')=='ok' and 'arm' in r:out.append(r)
    return out

def group(rows,keys):
    g=defaultdict(list)
    for r in rows:g[tuple(r[k] for k in keys)].append(r)
    return g

def key(r):return (r['zeta'],r['L'],r['lambda'],r['real'])

def paired(rows,arm,obs='CMI_mean'):
    b={key(r):r for r in rows if r['arm']=='baseline'};a={key(r):r for r in rows if r['arm']==arm};G=defaultdict(list)
    for k in b.keys()&a.keys():
        x=float(a[k].get(obs,np.nan))-float(b[k].get(obs,np.nan))
        if np.isfinite(x):G[k[:3]].append(x)
    out=[]
    for c,d in G.items():
        d=np.asarray(d);out.append((c,len(d),float(d.mean()),float(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else np.nan))
    return out

def vargain(rows,arm,obs='CMI_mean'):
    g=group(rows,['arm','zeta','L','lambda']);out=defaultdict(list)
    cells=set((r['zeta'],r['L'],r['lambda']) for r in rows)
    for z,L,lam in cells:
        kb=('baseline',z,L,lam);ka=(arm,z,L,lam)
        if kb not in g or ka not in g:continue
        xb=np.asarray([r.get(obs,np.nan) for r in g[kb]],float);xa=np.asarray([r.get(obs,np.nan) for r in g[ka]],float)
        xb=xb[np.isfinite(xb)];xa=xa[np.isfinite(xa)]
        if len(xb)>2 and len(xa)>2 and np.var(xa,ddof=1)>0:out[z].append(float(np.var(xb,ddof=1)/np.var(xa,ddof=1)))
    return out

def sign_changes(y):
    s=np.sign(np.asarray(y,float));s=s[s!=0]
    return int(np.sum(s[:-1]*s[1:]<0)) if len(s)>1 else 0

def crossing(x,y):
    hit=[]
    for i in range(len(y)-1):
        if y[i]*y[i+1]<0:hit.append(i)
    if len(hit)!=1:return None
    i=hit[0];return float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i]))

def bootcross(rows,arm,z,B=300):
    rr=[r for r in rows if r['arm']==arm and abs(r['zeta']-z)<1e-12];Ls=sorted(set(r['L'] for r in rr));lams=sorted(set(r['lambda'] for r in rr))
    if len(Ls)<2 or len(lams)<3:return None
    lo,hi=Ls[0],Ls[-1];g=group(rr,['L','lambda'])
    if any((L,l) not in g for L in (lo,hi) for l in lams):return None
    vals=lambda L,l:np.asarray([r['CMI_mean'] for r in g[(L,l)]],float)
    D=np.asarray([np.mean(vals(hi,l))-np.mean(vals(lo,l)) for l in lams]);c0=crossing(lams,D)
    rng=np.random.default_rng(12345+int(1000*z));cs=[];nu=0
    for _ in range(B):
        db=[]
        for l in lams:
            a=vals(lo,l);b=vals(hi,l);db.append(np.mean(rng.choice(b,len(b)))-np.mean(rng.choice(a,len(a))))
        if sign_changes(db)==1:
            c=crossing(lams,db)
            if c is not None:nu+=1;cs.append(c)
    q=np.percentile(cs,[16,50,84]) if cs else [np.nan]*3
    return c0,nu/B,float(q[2]-q[0])

def main():
    p=argparse.ArgumentParser();p.add_argument('--dir',required=True);p.add_argument('--bootstrap',type=int,default=300);a=p.parse_args();rows=load(a.dir)
    if not rows:raise SystemExit('no records')
    arms=sorted(set(r['arm'] for r in rows));zetas=sorted(set(r['zeta'] for r in rows));print('records',len(rows),'arms',arms,'zetas',zetas)
    print('\nDIAGNOSTICS')
    for arm in arms:
        rr=[r for r in rows if r['arm']==arm];M=lambda k:float(np.nanmean([r.get(k,np.nan) for r in rr]))
        print('%-12s nres=%6.1f GESS/N=%5.3f lag4/N=%5.3f minESS=%5.3f wall=%7.1fs'%(arm,M('n_resampling_events'),M('gen_ess_frac'),M('lagged_gess_frac_4w'),M('min_ess_frac'),M('wall_traj_s')))
    print('\nFINAL-OBSERVABLE CHECKS')
    for arm in arms:
        if arm=='baseline':continue
        sh=paired(rows,arm);bad=sum(np.isfinite(se) and se>0 and abs(mu)>2*se for _,_,mu,se in sh);vg=vargain(rows,arm)
        print('\n',arm,'paired cells',len(sh),'resolved shifts',bad)
        for z in zetas:
            if vg.get(z):print(' z=%.2f CMI variance gain median %.2fx'%(z,np.median(vg[z])))
    print('\nCROSSING CONDITIONING')
    for z in zetas:
        for arm in arms:
            x=bootcross(rows,arm,z,a.bootstrap)
            if x:print(' z=%.2f %-12s lambda=%s P(unique)=%.3f width=%.4f'%(z,arm,'nan' if x[0] is None else '%.4f'%x[0],x[1],x[2]))
    print('\nRule: do not promote an arm from GESS alone. Require final-observable/crossing improvement and no resolved shift.')
if __name__=='__main__':main()
