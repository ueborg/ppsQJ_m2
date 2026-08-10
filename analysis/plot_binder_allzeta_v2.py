"""
FSS collapse B_L/L for all zeta, with proper initialization from pairwise crossings.
"""
import pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.optimize import minimize

OLD = pickle.load(open('/Users/catlover1337/Downloads/clone_aggregate(1).pkl','rb'))
AC  = pickle.load(open('/Users/catlover1337/Downloads/aggregate_runAC.pkl','rb'))
B   = pickle.load(open('/Users/catlover1337/Downloads/aggregate_B.pkl','rb'))
merged = {}
for src in (OLD, AC, B):
    for k,e in src.items(): merged[k] = e

by_zL = defaultdict(lambda: defaultdict(list))
for (L,lam,z),e in merged.items():
    bm=e.get('B_L_mean',np.nan); be=e.get('B_L_err',np.nan)
    if not (np.isnan(bm) or bm<=0 or np.isnan(be)):
        by_zL[round(z,3)][L].append((lam,bm,be))
for z in by_zL:
    for L in by_zL[z]: by_zL[z][L].sort(key=lambda t:t[0])

def get_arrays(z,L):
    pts=by_zL[z].get(L,[])
    if len(pts)<4: return None,None,None
    return (np.array([t[0] for t in pts]),
            np.array([t[1] for t in pts]),
            np.array([t[2] for t in pts]))

def raw_crossing(z, L1=64, L2=128):
    """Quick pairwise crossing estimate as starting point."""
    for l1,l2 in [(96,128),(64,128),(48,128),(32,64)]:
        la1,b1,_=get_arrays(z,l1) if (get_arrays(z,l1)[0] is not None) else (None,None,None)
        la2,b2,_=get_arrays(z,l2) if (get_arrays(z,l2)[0] is not None) else (None,None,None)
        if la1 is None or la2 is None: continue
        d1={l:v for l,v in zip(la1,b1)}; d2={l:v for l,v in zip(la2,b2)}
        common=sorted(set(d1)&set(d2))
        if len(common)<4: continue
        lams=np.array(common); diff=np.array([d2[l]-d1[l] for l in lams])
        zcs=np.where(np.diff(np.sign(diff)))[0]
        if len(zcs)==0: continue
        i=zcs[0]; t=-diff[i]/(diff[i+1]-diff[i]+1e-12)
        return float(lams[i]+t*(lams[i+1]-lams[i]))
    return 0.3  # fallback

def collapse_res(params, z, Ls, log_err=0.35):
    lc,nu=params
    if nu<0.3 or nu>5.5 or lc<0.005 or lc>0.97: return 1e9
    curves=[]
    for L in Ls:
        lams,bm,be=get_arrays(z,L)
        if lams is None: continue
        y=bm/L; ey=be/L
        x=(lams-lc)*L**(1.0/nu)
        o=np.argsort(x); curves.append((x[o],y[o],ey[o]))
    if len(curves)<3: return 1e9
    tot=0; n=0
    for i in range(len(curves)):
        for j in range(i+1,len(curves)):
            x1,y1,e1=curves[i]; x2,y2,e2=curves[j]
            xlo,xhi=max(x1.min(),x2.min()),min(x1.max(),x2.max())
            if xhi<=xlo: continue
            mask=(x1>=xlo)&(x1<=xhi)
            if mask.sum()<3: continue
            f2=interp1d(x2,y2,kind='linear',bounds_error=False,fill_value='extrapolate')
            y2i=f2(x1[mask])
            ly1=np.log(np.maximum(y1[mask],1e-9))
            ly2=np.log(np.maximum(y2i,1e-9))
            tot+=np.sum(((ly1-ly2)/log_err)**2); n+=mask.sum()
    return tot/max(n,1)

def best_collapse(z, Ls):
    # seed from pairwise crossing
    lc0 = raw_crossing(z)
    # fine grid around the seed
    seeds = []
    for dlc in [-0.05,-0.02,0,0.02,0.05]:
        for nu0 in [1.5,2.0,2.5,3.0]:
            lc_try = max(0.01, lc0+dlc)
            seeds.append((collapse_res((lc_try,nu0),z,Ls),(lc_try,nu0)))
    seeds.sort()
    best_params = seeds[0][1]
    res=minimize(collapse_res, best_params, args=(z,Ls),
                 method='Nelder-Mead',options={'xatol':1e-5,'maxiter':800})
    return float(res.x[0]),float(res.x[1]),float(res.fun)

all_zetas = sorted(by_zL.keys())
Ls_use = [32,48,64,96,128]
colors_L = {32:'#4363d8',48:'#f58231',64:'#3cb44b',96:'#e6194b',128:'#911eb4'}
ncols=4; nrows=int(np.ceil(len(all_zetas)/ncols))

fig,axes=plt.subplots(nrows,ncols,figsize=(18,4.5*nrows))
axes=axes.flatten(); fig.subplots_adjust(hspace=0.55,wspace=0.28)

results={}
for idx,z in enumerate(all_zetas):
    ax=axes[idx]
    lc,nu,q=best_collapse(z,Ls_use)
    results[z]=(lc,nu,q)
    print(f"ζ={z:.3f}: λ_c={lc:.4f}, ν={nu:.3f}, q={q:.2f}")

    plotted=False
    for L in Ls_use:
        lams,bm,be=get_arrays(z,L)
        if lams is None: continue
        x=(lams-lc)*L**(1.0/nu)
        y=bm/L; ey=be/L
        mask=np.abs(x)<6.0
        if mask.sum()<3: continue
        ax.errorbar(x[mask],y[mask],yerr=ey[mask],
                    fmt='o',ms=5.5,lw=1.4,capsize=2,
                    label=f"L={L}",color=colors_L[L],zorder=3)
        plotted=True

    ax.axvline(0,color='crimson',lw=2.0,ls='--',zorder=5,alpha=0.85)
    ax.axvspan(-6,0,color='steelblue',alpha=0.04)
    ax.axvspan(0,6,color='firebrick',alpha=0.04)
    ax.set_xlim(-6,6)
    ax.set_xlabel(r'$(\lambda-\lambda_c)\,L^{1/\nu}$',fontsize=10)
    ax.set_ylabel(r'$B_L\,/\,L$',fontsize=10)

    quality_str = "good" if q<4 else ("ok" if q<8 else "poor — need larger L")
    quality_col = 'green' if q<4 else ('orange' if q<8 else 'red')
    ax.set_title(
        f'ζ = {z:.3f}\n'
        f'$\\lambda_c={lc:.3f}$,  $\\nu={nu:.2f}$',
        fontsize=10.5, fontweight='bold')
    ax.text(0.98,0.97,f'quality: {quality_str}',
            transform=ax.transAxes,ha='right',va='top',
            fontsize=8,color=quality_col,
            bbox=dict(boxstyle='round,pad=0.2',facecolor='white',alpha=0.8))
    ax.legend(fontsize=8,ncol=2,loc='lower left',framealpha=0.7,handlelength=1)
    ax.grid(alpha=0.18)

for idx in range(len(all_zetas),len(axes)): axes[idx].set_visible(False)

# Explanation box
fig.text(0.50,1.005,
    r"Each panel: $B_L/L$ vs rescaled $\lambda$ at one $\zeta$.  "
    r"Good collapse = all 5 colours lie on top of each other (same $y$ for same $x$).  "
    r"Dashed line = $\lambda_c$.  Blue region = log phase, red = area law.",
    ha='center',va='bottom',fontsize=10,
    bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.8))

plt.suptitle(
    r"FSS collapse $B_L/L$ — all $\zeta$ values, $L\in\{32,48,64,96,128\}$",
    fontsize=13,y=1.03)

out='/Users/catlover1337/Documents/ppsQJ_m2/analysis/binder_allzeta_collapse.png'
plt.savefig(out,dpi=130,bbox_inches='tight')
print(f"\nSaved: {out}")
