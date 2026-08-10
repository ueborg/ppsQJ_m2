import pickle, numpy as np
from scipy.optimize import curve_fit
rng=np.random.default_rng(1)

D=pickle.load(open('/Users/catlover1337/Downloads/_guided_master.pkl','rb'))
df,arrs=D['df'],D['arrs']; df=df.reset_index(drop=True)
res=pickle.load(open('/Users/catlover1337/Downloads/_crossings_B_CMI.pkl','rb'))

ZB=[0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.85]

# ---- Case B: build lam_c(zeta) from the 3 fully-populated mid-L pairs ----
good_pairs=[(32,64),(48,96),(64,96)]
zc,lc,le=[],[],[]
print('=== CASE B lam_c(zeta): mean of pairs',good_pairs,', err = pair spread ===')
print(' zeta   lam_c   err     A=lc/sqrt(z)   r_c=lc/(1-lc)')
for z in ZB:
    vals=[res[(z,a,b)][0] for (a,b) in good_pairs if np.isfinite(res[(z,a,b)][0])]
    if len(vals)<2: continue
    v=np.array(vals); m=v.mean(); s=v.std(ddof=1)
    zc.append(z); lc.append(m); le.append(max(s,0.003))
    print(f' {z:5.3f}  {m:.3f}  {s:.3f}     {m/np.sqrt(z):.3f}        {m/(1-m):.3f}')
zc=np.array(zc); lc=np.array(lc); le=np.array(le)

def pl(z,A,phi): return A*z**phi
# fit lam_c = A z^phi
p,cov=curve_fit(pl,zc,lc,p0=[0.5,0.5],sigma=le,absolute_sigma=True,maxfev=20000)
perr=np.sqrt(np.diag(cov))
chi2=np.sum(((lc-pl(zc,*p))/le)**2)/(len(zc)-2)
print(f'\n FIT lam_c = A*zeta^phi :  A={p[0]:.3f}±{perr[0]:.3f}  phi={p[1]:.3f}±{perr[1]:.3f}  chi2/dof={chi2:.2f}')
# fixed phi=1/2
pf,covf=curve_fit(lambda z,A:A*np.sqrt(z),zc,lc,p0=[0.5],sigma=le,absolute_sigma=True)
chi2f=np.sum(((lc-pf[0]*np.sqrt(zc))/le)**2)/(len(zc)-1)
print(f' FIT lam_c = A*sqrt(zeta):  A={pf[0]:.3f}±{np.sqrt(covf[0,0]):.3f}            chi2/dof={chi2f:.2f}')
# r_c fit
rc=lc/(1-lc); rce=le/(1-lc)**2
pr,covr=curve_fit(pl,zc,rc,p0=[0.5,0.6],sigma=rce,absolute_sigma=True,maxfev=20000)
chi2r=np.sum(((rc-pl(zc,*pr))/rce)**2)/(len(zc)-2)
print(f' FIT r_c   = A*zeta^phi :  A={pr[0]:.3f}±{np.sqrt(covr[0,0]):.3f}  phi={pr[1]:.3f}±{np.sqrt(covr[1,1]):.3f}  chi2/dof={chi2r:.2f}')
# extrapolated Born value
print(f' lam_c(zeta=1) from free-power fit = {pl(1.0,*p):.3f}   (Carollo Born = 0.5)')

pickle.dump(dict(zc=zc,lc=lc,le=le,fit_free=p,fit_sqrt=pf,fit_rc=pr),
            open('/Users/catlover1337/Downloads/_fit_B.pkl','wb'))

# =====================  CASE A  =====================
print('\n\n=== CASE A: <CMI> crossings, test lam_c^A = 1/2 ===')
def curveA(L,zeta,obs):
    m=(df.dset=='A')&(df.L==L)&(np.abs(df.zeta-zeta)<1e-9)
    idx=np.where(m.values)[0]
    if len(idx)==0: return None
    rows=sorted((df.lam.values[i],i) for i in idx)
    lams=np.array([r[0] for r in rows]); mats=[]
    for _,i in rows:
        a=arrs[i]
        v=a['CMI_means_all'] if obs=='CMI' else a['CMI_means_all']*a['S_AB_means_all']
        mats.append(np.asarray(v,float))
    return lams,np.vstack(mats)
def cq(lams,m1,m2):
    y1=np.nanmean(m1,1);y2=np.nanmean(m2,1);ok=np.isfinite(y1)&np.isfinite(y2)
    if ok.sum()<4:return np.nan
    x=lams[ok];p=np.polyfit(x,y1[ok],2)-np.polyfit(x,y2[ok],2)
    r=np.roots(p);r=[t.real for t in r if abs(t.imag)<1e-9]
    lo,hi=x.min(),x.max();r=[t for t in r if lo-0.02<=t<=hi+0.02]
    return min(r,key=lambda t:abs(t-0.5)) if r else np.nan
def crossA(L1,L2,z,obs='CMI',B=300):
    c1=curveA(L1,z,obs);c2=curveA(L2,z,obs)
    if c1 is None or c2 is None:return(np.nan,np.nan,0)
    l1,M1=c1;l2,M2=c2;com=np.intersect1d(np.round(l1,4),np.round(l2,4))
    if len(com)<4:return(np.nan,np.nan,len(com))
    i1=[np.where(np.round(l1,4)==x)[0][0] for x in com]
    i2=[np.where(np.round(l2,4)==x)[0][0] for x in com]
    M1=M1[i1];M2=M2[i2]
    pt=cq(com,M1,M2);vals=[]
    for _ in range(B):
        b1=M1[:,rng.integers(0,M1.shape[1],M1.shape[1])]
        b2=M2[:,rng.integers(0,M2.shape[1],M2.shape[1])]
        v=cq(com,b1,b2)
        if np.isfinite(v):vals.append(v)
    e=(np.percentile(vals,84)-np.percentile(vals,16))/2 if len(vals)>10 else np.nan
    return(pt,e,len(com))
pairsA=[(32,64),(48,96),(64,96),(64,128),(96,128)]
print(' zeta  '+'  '.join(f'{a}-{b}'.center(12) for a,b in pairsA)+'   mean(all)')
A_zc,A_lc,A_le=[],[],[]
for z in ZB:
    line=f' {z:5.3f}';vv=[]
    for (a,b) in pairsA:
        pt,e,nc=crossA(a,b,z)
        if np.isfinite(pt):
            line+=f' {pt:.3f}±{(e if np.isfinite(e) else 0):.3f}'.ljust(12); vv.append(pt)
        else: line+=' '+'--'.center(11)
    if len(vv)>=2:
        m=np.mean(vv);s=np.std(vv,ddof=1);line+=f'   {m:.3f}±{s:.3f}'
        A_zc.append(z);A_lc.append(m);A_le.append(max(s,0.003))
    print(line)
A_zc=np.array(A_zc);A_lc=np.array(A_lc);A_le=np.array(A_le)
# weighted mean vs 0.5
w=1/A_le**2; mbar=np.sum(w*A_lc)/np.sum(w); mbar_e=1/np.sqrt(np.sum(w))
print(f'\n CASE A weighted-mean lam_c over all zeta = {mbar:.4f} ± {mbar_e:.4f}  (predicted 0.5)')
# slope of lam_c^A vs zeta (should be ~0)
sl=np.polyfit(A_zc,A_lc,1)
print(f' linear slope d lam_c^A / d zeta = {sl[0]:+.4f} (predicted 0)')
pickle.dump(dict(zc=A_zc,lc=A_lc,le=A_le),open('/Users/catlover1337/Downloads/_fit_A.pkl','wb'))
print('\nsaved _fit_B.pkl, _fit_A.pkl')
