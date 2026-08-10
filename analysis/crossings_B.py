import pickle, numpy as np
rng = np.random.default_rng(0)

D = pickle.load(open('/Users/catlover1337/Downloads/_guided_master.pkl','rb'))
df, arrs = D['df'], D['arrs']
df = df.reset_index(drop=True)

def curve(dset, L, zeta, obs):
    m = (df.dset==dset)&(df.L==L)&(np.abs(df.zeta-zeta)<1e-9)
    idx = np.where(m.values)[0]
    if len(idx)==0: return None
    rows = sorted((df.lam.values[i], i) for i in idx)
    lams = np.array([r[0] for r in rows])
    mats = []
    for _,i in rows:
        a = arrs[i]
        if   obs=='CMI': v=a['CMI_means_all']
        elif obs=='KMR': v=a['CMI_means_all']*a['S_AB_means_all']
        elif obs=='B_L': v=a['B_L_means_all']
        elif obs=='S':   v=a['S_means_all']
        mats.append(np.asarray(v,float))
    return lams, np.vstack(mats)            # lams[n], M[n,nreal]

def cross_quad(lams, m1, m2):
    """crossing of two quadratic fits within the lam window; nan if none."""
    y1=np.nanmean(m1,1); y2=np.nanmean(m2,1)
    ok=np.isfinite(y1)&np.isfinite(y2)
    if ok.sum()<4: return np.nan
    x=lams[ok]
    p=np.polyfit(x,y1[ok],2)-np.polyfit(x,y2[ok],2)
    r=np.roots(p); r=[t.real for t in r if abs(t.imag)<1e-9]
    lo,hi=x.min(),x.max(); c=0.5*(lo+hi)
    r=[t for t in r if lo-0.015<=t<=hi+0.015]
    return min(r,key=lambda t:abs(t-c)) if r else np.nan

def cross_boot(dsetL1,L1,dsetL2,L2,zeta,obs,B=300):
    c1=curve(dsetL1,L1,zeta,obs); c2=curve(dsetL2,L2,zeta,obs)
    if c1 is None or c2 is None: return (np.nan,np.nan,np.nan,0)
    lam1,M1=c1; lam2,M2=c2
    common=np.intersect1d(np.round(lam1,4),np.round(lam2,4))
    if len(common)<4: return (np.nan,np.nan,np.nan,len(common))
    i1=[np.where(np.round(lam1,4)==x)[0][0] for x in common]
    i2=[np.where(np.round(lam2,4)==x)[0][0] for x in common]
    M1=M1[i1]; M2=M2[i2]; lams=common
    point=cross_quad(lams,M1,M2)
    nr1=M1.shape[1]; nr2=M2.shape[1]
    vals=[]
    for _ in range(B):
        b1=M1[:,rng.integers(0,nr1,nr1)]
        b2=M2[:,rng.integers(0,nr2,nr2)]
        v=cross_quad(lams,b1,b2)
        if np.isfinite(v): vals.append(v)
    if len(vals)<10: return (point,np.nan,np.nan,len(common))
    lo,hi=np.percentile(vals,[16,84])
    return (point,(hi-lo)/2,np.median(vals),len(common))

ZB=[0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.85]
def dset_of(L): return 'B_highL' if L==160 else 'B_prod'
pairs=[(32,64),(48,96),(64,96),(96,128),(128,160),(96,160),(64,128)]

print('=== CASE B: <CMI> crossings  lam_c(zeta) per L-pair  (point [boot_err]) ===')
hdr='zeta  '+'  '.join(f'{a}-{b}'.center(13) for a,b in pairs)
print(hdr)
results={}
for z in ZB:
    line=f'{z:5.3f} '
    for (a,b) in pairs:
        pt,err,med,nc=cross_boot(dset_of(a),a,dset_of(b),b,z,'CMI')
        results[(z,a,b)]=(pt,err,med,nc)
        if np.isfinite(pt):
            line+=f' {pt:.3f}±{(err if np.isfinite(err) else 0):.3f}'.ljust(13)
        else:
            line+=' '+'--'.center(12)
    print(line)

pickle.dump(results, open('/Users/catlover1337/Downloads/_crossings_B_CMI.pkl','wb'))
print('\nsaved _crossings_B_CMI.pkl')
