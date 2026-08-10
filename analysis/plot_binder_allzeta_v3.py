"""
FSS collapse B_L/L using the KNOWN (lambda_c, nu) from global_fss_merged_v2.json.
Shows visually whether the collapse works at each zeta.
"""
import pickle, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

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

# Load known lambda_c, nu
r = json.load(open('/Users/catlover1337/Documents/ppsQJ_m2/analysis/global_fss_merged_v2.json'))
known = {round(float(z),3): (d['lc'],d['nu']) for z,d in r['per_zeta'].items()}

all_zetas = sorted(known.keys())
Ls_use = [32,48,64,96,128]
colors_L = {32:'#4363d8',48:'#f58231',64:'#3cb44b',96:'#e6194b',128:'#911eb4'}

ncols=4; nrows=int(np.ceil(len(all_zetas)/ncols))
fig,axes=plt.subplots(nrows,ncols,figsize=(19,4.8*nrows))
axes=axes.flatten(); fig.subplots_adjust(hspace=0.60,wspace=0.30)

for idx,z in enumerate(all_zetas):
    ax=axes[idx]
    lc,nu=known[z]

    for L in Ls_use:
        pts=by_zL[z].get(L,[])
        if len(pts)<4: continue
        lams=np.array([t[0] for t in pts])
        bm  =np.array([t[1] for t in pts])
        be  =np.array([t[2] for t in pts])
        x  = (lams-lc)*L**(1.0/nu)
        y  = bm/L
        ey = be/L
        mask=np.abs(x)<7.0
        if mask.sum()<3: continue
        ax.errorbar(x[mask],y[mask],yerr=ey[mask],
                    fmt='o',ms=5.5,lw=1.5,capsize=2,
                    label=f"L={L}",color=colors_L[L],zorder=3)

    ax.axvline(0,color='crimson',lw=2.0,ls='--',zorder=5)
    ax.axvspan(-7,0,color='steelblue',alpha=0.05)
    ax.axvspan(0, 7,color='firebrick',alpha=0.05)

    # Annotation: what "good collapse" means
    ax.text(-6.5,0.02,'← LOG\nphase',fontsize=8,color='steelblue',va='bottom',ha='left')
    ax.text( 6.5,0.02,'AREA →\nphase',fontsize=8,color='firebrick',va='bottom',ha='right')

    ax.set_xlim(-7,7)
    ax.set_xlabel(r'$(\lambda-\lambda_c)\,L^{1/\nu}$',fontsize=11)
    ax.set_ylabel(r'$B_L\,/\,L$',fontsize=11)
    ax.set_title(
        f'ζ = {z:.3f}   '
        f'($\\lambda_c={lc:.3f}$, $\\nu={nu:.2f}$)',
        fontsize=11,fontweight='bold')
    ax.legend(fontsize=8.5,ncol=2,loc='upper right',
              framealpha=0.8,handlelength=1.2)
    ax.grid(alpha=0.18)

for idx in range(len(all_zetas),len(axes)): axes[idx].set_visible(False)

# Master explanation
expl = (
    "How to read each panel:\n"
    "• x-axis = (λ − λc) × L^{1/ν} : rescales λ so all L sit on one curve\n"
    "• y-axis = B_L / L : normalised so both phases have finite values\n"
    "• Good collapse: all 5 colours lie on top of each other\n"
    "• x = 0 (dashed) is exactly λc — the MIPT critical point\n"
    "• Left (blue): log phase — B_L/L is large and roughly equal for all L\n"
    "• Right (red): area phase — B_L/L → 0 for all L"
)
fig.text(0.50,-0.01,expl,ha='center',va='top',fontsize=9.5,
         bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.85,edgecolor='goldenrod'))

plt.suptitle(
    r"FSS collapse of $B_L / L$  at all $\zeta$ values"
    "\n"
    r"$\lambda_c$, $\nu$ from global FSS (Binder collapse minimisation on merged dataset)",
    fontsize=13,y=1.01)

out='/Users/catlover1337/Documents/ppsQJ_m2/analysis/binder_allzeta_collapse.png'
plt.savefig(out,dpi=130,bbox_inches='tight')
print(f"Saved: {out}")
