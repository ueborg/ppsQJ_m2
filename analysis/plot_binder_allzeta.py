"""
Full FSS collapse B_L/L for all available zeta values.
2 rows x 5 cols = 10 panels, one per zeta.
Normalisation: y = B_L / L so both phases have O(1) values and all L overlap.
Best (lambda_c, nu) found by minimising collapse residual.
"""
import pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# ── load ──────────────────────────────────────────────────────────────────────
OLD = pickle.load(open('/Users/catlover1337/Downloads/clone_aggregate(1).pkl','rb'))
AC  = pickle.load(open('/Users/catlover1337/Downloads/aggregate_runAC.pkl','rb'))
B   = pickle.load(open('/Users/catlover1337/Downloads/aggregate_B.pkl','rb'))
merged = {}
for src in (OLD, AC, B):
    for k,e in src.items(): merged[k] = e

by_zL = defaultdict(lambda: defaultdict(list))
for (L,lam,z),e in merged.items():
    bm = e.get('B_L_mean',np.nan); be = e.get('B_L_err',np.nan)
    if not (np.isnan(bm) or bm<=0 or np.isnan(be)):
        by_zL[round(z,3)][L].append((lam,bm,be))
for z in by_zL:
    for L in by_zL[z]:
        by_zL[z][L].sort(key=lambda t: t[0])

def get_arrays(z,L):
    pts = by_zL[z].get(L,[])
    if len(pts)<4: return None,None,None
    return (np.array([t[0] for t in pts]),
            np.array([t[1] for t in pts]),
            np.array([t[2] for t in pts]))

def collapse_res(params, z, Ls, log_err=0.3):
    lc,nu = params
    if nu<0.3 or nu>5 or lc<0.01 or lc>0.98: return 1e9
    curves = []
    for L in Ls:
        lams,bm,be = get_arrays(z,L)
        if lams is None: continue
        y  = bm/L;  ey = be/L
        x  = (lams-lc)*L**(1.0/nu)
        o  = np.argsort(x)
        curves.append((x[o],y[o],ey[o]))
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
    best=(np.inf,None)
    for lc0 in np.linspace(0.03,0.70,28):
        for nu0 in [1.5,2.0,2.5,3.0]:
            q=collapse_res((lc0,nu0),z,Ls)
            if q<best[0]: best=(q,(lc0,nu0))
    res=minimize(collapse_res, best[1], args=(z,Ls),
                 method='Nelder-Mead',options={'xatol':1e-5,'maxiter':600})
    return float(res.x[0]), float(res.x[1]), float(res.fun)

# ── all zeta values we have, sorted ──────────────────────────────────────────
all_zetas = sorted(by_zL.keys())
print(f"Zeta values in data: {all_zetas}")
Ls_use = [32,48,64,96,128]
colors_L = {32:'#4363d8',48:'#f58231',64:'#3cb44b',96:'#e6194b',128:'#911eb4'}
n_z = len(all_zetas)

# ── layout: dynamic rows ──────────────────────────────────────────────────────
ncols = 5
nrows = int(np.ceil(n_z / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.2*nrows))
axes = axes.flatten()
fig.subplots_adjust(hspace=0.52, wspace=0.30)

results = {}
for idx, z in enumerate(all_zetas):
    ax = axes[idx]

    # fit collapse
    lc, nu, q = best_collapse(z, Ls_use)
    results[z] = (lc, nu, q)
    print(f"ζ={z:.3f}: λ_c={lc:.4f}, ν={nu:.3f}, q={q:.2f}")

    # plot each L
    plotted = False
    for L in Ls_use:
        lams, bm, be = get_arrays(z, L)
        if lams is None: continue
        x  = (lams - lc) * L**(1.0/nu)
        y  = bm / L
        ey = be / L
        mask = np.abs(x) < 6.0
        if mask.sum() < 3: continue
        ax.errorbar(x[mask], y[mask], yerr=ey[mask],
                    fmt='o', ms=5, lw=1.4, capsize=2,
                    label=f"L={L}", color=colors_L[L], zorder=3)
        plotted = True

    ax.axvline(0, color='crimson', lw=2.0, ls='--', zorder=5, alpha=0.8)

    # shading
    ylims = ax.get_ylim() if plotted else (0, 1)
    ax.axvspan(-6, 0, color='steelblue', alpha=0.04)
    ax.axvspan(0,  6, color='firebrick', alpha=0.04)

    ax.set_xlim(-6, 6)
    ax.set_xlabel(r'$(\lambda-\lambda_c)\,L^{1/\nu}$', fontsize=10)
    ax.set_ylabel(r'$B_L\,/\,L$', fontsize=10)
    ax.set_title(
        f'ζ = {z:.3f}\n'
        f'$\\lambda_c={lc:.3f}$, $\\nu={nu:.2f}$, $q={q:.1f}$',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, ncol=2, loc='upper right',
              framealpha=0.7, handlelength=1.2)
    ax.grid(alpha=0.18)

# hide unused panels
for idx in range(len(all_zetas), len(axes)):
    axes[idx].set_visible(False)

# single shared annotation
fig.text(0.01, 0.5,
         'Good collapse: all colours trace the same curve.\n'
         'Left of dashed line = log phase (B/L finite).\n'
         'Right = area phase (B/L → 0).',
         va='center', ha='left', fontsize=9, rotation=90,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.suptitle(
    r"FSS collapse of $B_L/L$ for all $\zeta$ values — $L\in\{32,48,64,96,128\}$""\n"
    r"When collapse is good: all 5 colours trace ONE master curve; $\lambda_c$ is where curves cross ($x=0$)",
    fontsize=12, y=1.01)

out = '/Users/catlover1337/Documents/ppsQJ_m2/analysis/binder_allzeta_collapse.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\nSaved: {out}")

# Summary table
print("\n=== Summary: λ_c and ν across all ζ ===\n")
print(f"{'zeta':>7}  {'lambda_c':>9}  {'nu':>7}  {'quality':>9}")
print("-"*38)
for z,(lc,nu,q) in sorted(results.items()):
    flag = "  ← poor" if q > 8 else ("  ← ok" if q > 4 else "  ← good")
    print(f"{z:>7.3f}  {lc:>9.4f}  {nu:>7.3f}  {q:>9.2f}{flag}")
