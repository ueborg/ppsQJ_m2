import pickle, numpy as np
D=pickle.load(open('/Users/catlover1337/Downloads/_guided_master.pkl','rb'))
df,arrs=D['df'],D['arrs']; df=df.reset_index(drop=True)

# (1) Case A parametrisation: is lam=0.5 the self-dual point alpha==gamma?
print('=== Case A rate parametrisation (alpha_rate, gamma_rate vs lam) ===')
sub=df[(df.dset=='A')&(df.L==64)&(np.abs(df.zeta-0.1)<1e-9)].sort_values('lam')
for _,r in sub.iterrows():
    print(f'  lam={r.lam:.3f}  alpha_rate={r.alpha_rate:.4f}  gamma_rate={r.gamma_rate:.4f}  sum={r.alpha_rate+r.gamma_rate:.3f}')
# where is alpha==gamma?
print('  -> self-dual (alpha=gamma) is at lam where alpha_rate=gamma_rate=0.5')

# (2) multi-observable Case A crossing at a few zeta: CMI vs KMR vs B_L vs S
print('\n=== Case A crossings by observable (pair 48-96), test if 0.436 is observable-robust ===')
def curveA(L,zeta,obs):
    m=(df.dset=='A')&(df.L==L)&(np.abs(df.zeta-zeta)<1e-9)
    idx=np.where(m.values)[0]
    if len(idx)==0:return None
    rows=sorted((df.lam.values[i],i) for i in idx)
    lams=np.array([r[0] for r in rows]);mats=[]
    for _,i in rows:
        a=arrs[i]
        if obs=='CMI':v=a['CMI_means_all']
        elif obs=='KMR':v=a['CMI_means_all']*a['S_AB_means_all']
        elif obs=='B_L':v=a['B_L_means_all']
        elif obs=='S':v=a['S_means_all']
        mats.append(np.asarray(v,float))
    return lams,np.vstack(mats)
def cq(lams,m1,m2,tgt=0.5):
    y1=np.nanmean(m1,1);y2=np.nanmean(m2,1);ok=np.isfinite(y1)&np.isfinite(y2)
    if ok.sum()<4:return np.nan
    x=lams[ok];p=np.polyfit(x,y1[ok],2)-np.polyfit(x,y2[ok],2)
    r=np.roots(p);r=[t.real for t in r if abs(t.imag)<1e-9]
    lo,hi=x.min(),x.max();r=[t for t in r if lo-0.02<=t<=hi+0.02]
    return min(r,key=lambda t:abs(t-tgt)) if r else np.nan
def cr(L1,L2,z,obs):
    c1=curveA(L1,z,obs);c2=curveA(L2,z,obs)
    if c1 is None or c2 is None:return np.nan
    l1,M1=c1;l2,M2=c2;com=np.intersect1d(np.round(l1,4),np.round(l2,4))
    if len(com)<4:return np.nan
    i1=[np.where(np.round(l1,4)==x)[0][0] for x in com]
    i2=[np.where(np.round(l2,4)==x)[0][0] for x in com]
    return cq(com,M1[i1],M2[i2])
print(' zeta    CMI     KMR     B_L      S')
for z in [0.05,0.1,0.15,0.2]:
    print(f' {z:5.3f}  '+'  '.join(f'{cr(48,96,z,o):.3f}' if np.isfinite(cr(48,96,z,o)) else '  --  '
                                   for o in ['CMI','KMR','B_L','S']))

# (3) Case A: is CMI(lam) symmetric about 0.5?  print CMI at lam and (1-lam) pairs, L=96 z=0.1
print('\n=== Case A CMI(lam) symmetry check about 0.5 (L=96, zeta=0.1) ===')
c=curveA(96,0.1,'CMI'); lams,M=c; y=np.nanmean(M,1)
for i,(L_,yv) in enumerate(zip(lams,y)):
    mirror=0.5+(0.5-L_)
    j=np.argmin(np.abs(lams-mirror))
    print(f'  lam={L_:.3f}: CMI={yv:.3f}   mirror lam={lams[j]:.3f}: CMI={y[j]:.3f}   diff={yv-y[j]:+.3f}')
