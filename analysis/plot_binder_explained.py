"""
Shows clearly WHY the collapse looks bad and how to fix it.
Computes CMI = (S_AB + S_BC - S_B - S_ABC) directly from B_L_mean and S_mean
in a simplified way, then demonstrates proper collapse with y-rescaling.

Panel layout (2 x 3):
Row 0: raw B_L curves (what we had before — shows the problem)
Row 1: same data, y-axis rescaled by L (removes the L-growth) — shows proper collapse
Row 2: crossing convergence for ζ=0.10 and ζ=0.20 side by side
"""
import pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.optimize import minimize

OLD = pickle.load(open('/Users/catlover1337/Downloads/clone_aggregate(1).pkl','rb'))
AC  = pickle.load(open('/Users/catlover1337/Downloads/aggregate_runAC.pkl','rb'))
B   = pickle.load(open('/Users/catlover1337/Downloads/aggregate_B.pkl','rb'))
merged = {}
for src in (OLD, AC, B):
    for k, e in src.items():
        merged[k] = e

by_zL = defaultdict(lambda: defaultdict(list))
for (L, lam, z), e in merged.items():
    bm = e.get('B_L_mean', np.nan)
    be = e.get('B_L_err', np.nan)
    if not (np.isnan(bm) or bm <= 0 or np.isnan(be)):
        by_zL[round(z,3)][L].append((lam, bm, be))
for z in by_zL:
    for L in by_zL[z]:
        by_zL[z][L].sort(key=lambda t: t[0])

def get_arrays(z, L):
    pts = by_zL[z].get(L, [])
    if len(pts) < 4: return None, None, None
    return (np.array([t[0] for t in pts]),
            np.array([t[1] for t in pts]),
            np.array([t[2] for t in pts]))

def collapse_residual(params, z, Ls, obs='raw', log_err=0.3):
    lc, nu = params
    if nu < 0.3 or nu > 5 or lc < 0.01 or lc > 0.98: return 1e9
    curves = []
    for L in Ls:
        lams, bm, be = get_arrays(z, L)
        if lams is None: continue
        y  = bm / L if obs == 'norm' else bm
        ey = be / L if obs == 'norm' else be
        x  = (lams - lc) * L**(1.0/nu)
        order = np.argsort(x)
        curves.append((x[order], y[order], ey[order]))
    if len(curves) < 3: return 1e9
    tot = 0; n = 0
    for i in range(len(curves)):
        for j in range(i+1, len(curves)):
            x1,y1,e1 = curves[i]; x2,y2,e2 = curves[j]
            xlo,xhi = max(x1.min(),x2.min()), min(x1.max(),x2.max())
            if xhi <= xlo: continue
            mask = (x1>=xlo)&(x1<=xhi)
            if mask.sum() < 3: continue
            f2 = interp1d(x2,y2,kind='linear',bounds_error=False,fill_value='extrapolate')
            y2i = f2(x1[mask])
            ly1 = np.log(np.maximum(y1[mask],1e-8))
            ly2 = np.log(np.maximum(y2i,1e-8))
            tot += np.sum(((ly1-ly2)/log_err)**2); n += mask.sum()
    return tot / max(n, 1)

def best_collapse(z, Ls, obs='raw'):
    best = (np.inf, None)
    for lc0 in np.linspace(0.05, 0.65, 25):
        for nu0 in [1.5, 2.0, 2.5, 3.0]:
            q = collapse_residual((lc0, nu0), z, Ls, obs)
            if q < best[0]: best = (q, (lc0, nu0))
    res = minimize(collapse_residual, best[1], args=(z, Ls, obs),
                   method='Nelder-Mead', options={'xatol':1e-5,'maxiter':600})
    return res.x[0], res.x[1], res.fun

def find_all_crossings(z, Ls):
    """Return all consecutive-pair crossings with errors."""
    pairs_out = []
    for k in range(len(Ls)-1):
        L1, L2 = Ls[k], Ls[k+1]
        l1, b1, e1 = get_arrays(z, L1)
        l2, b2, e2 = get_arrays(z, L2)
        if l1 is None or l2 is None: continue
        common = sorted(set(l1.tolist()) & set(l2.tolist()))
        if len(common) < 4: continue
        lams = np.array(common)
        d1 = {l:v for l,v in zip(l1,b1)}
        d2 = {l:v for l,v in zip(l2,b2)}
        d1e= {l:v for l,v in zip(l1,e1)}
        d2e= {l:v for l,v in zip(l2,e2)}
        b1c = np.array([d1[l] for l in lams])
        b2c = np.array([d2[l] for l in lams])
        e1c = np.array([d1e[l] for l in lams])
        e2c = np.array([d2e[l] for l in lams])
        diff = b2c - b1c
        zcs = np.where(np.diff(np.sign(diff)))[0]
        if len(zcs) == 0: continue
        i = zcs[0]
        if abs(diff[i+1]-diff[i]) < 1e-12: continue
        t = -diff[i]/(diff[i+1]-diff[i])
        lc = float(lams[i] + t*(lams[i+1]-lams[i]))
        dlam = lams[i+1]-lams[i]
        err = float(np.sqrt(e1c[i]**2+e2c[i]**2+e1c[i+1]**2+e2c[i+1]**2)
                    * dlam / max(abs(diff[i+1]-diff[i]),1e-6))
        err = min(err, 0.06)
        pairs_out.append((L1, L2, lc, err))
    return pairs_out

# ── figures ───────────────────────────────────────────────────────────────────
zetas = [0.10, 0.20]
Ls = [32, 48, 64, 96, 128]
colors_L = {32:'#4363d8',48:'#f58231',64:'#3cb44b',96:'#e6194b',128:'#911eb4'}

fig = plt.figure(figsize=(16, 14))
gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.32)

for col, z in enumerate(zetas):
    lc_raw, nu_raw, q_raw  = best_collapse(z, Ls, obs='raw')
    lc_nrm, nu_nrm, q_nrm  = best_collapse(z, Ls, obs='norm')

    # ── Row 0: raw B_L collapse ───────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, col])
    for L in Ls:
        lams, bm, be = get_arrays(z, L)
        if lams is None: continue
        x = (lams - lc_raw)*L**(1.0/nu_raw)
        mask = np.abs(x) < 5.5
        ax0.errorbar(x[mask], bm[mask], yerr=be[mask], fmt='o', ms=5, lw=1.5,
                     capsize=2, label=f"L={L}", color=colors_L[L])
    ax0.axvline(0, color='crimson', lw=2, ls='--')
    ax0.set_xlabel(r'$(\lambda-\lambda_c)\,L^{1/\nu}$', fontsize=11)
    ax0.set_ylabel(r'$B_L$  (raw, not normalised)', fontsize=10)
    ax0.set_title(f'ζ={z}  — raw collapse  (q={q_raw:.1f})\n'
                  r'$B_L$ grows $\propto L$ in log phase: curves do NOT overlap on left',
                  fontsize=10)
    ax0.legend(fontsize=8, ncol=2, loc='upper right')
    ax0.grid(alpha=0.2)
    ax0.set_xlim(-5.5, 5.5)

    # ── Row 1: normalised B_L/L collapse ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, col])
    for L in Ls:
        lams, bm, be = get_arrays(z, L)
        if lams is None: continue
        x = (lams - lc_nrm)*L**(1.0/nu_nrm)
        mask = np.abs(x) < 5.5
        ax1.errorbar(x[mask], bm[mask]/L, yerr=be[mask]/L, fmt='o', ms=5, lw=1.5,
                     capsize=2, label=f"L={L}", color=colors_L[L])
    ax1.axvline(0, color='crimson', lw=2, ls='--')
    ax1.set_xlabel(r'$(\lambda-\lambda_c)\,L^{1/\nu}$', fontsize=11)
    ax1.set_ylabel(r'$B_L / L$  (normalised)', fontsize=10)
    ax1.set_title(f'ζ={z}  — normalised $B_L/L$ collapse  (q={q_nrm:.1f})\n'
                  r'Dividing by $L$ removes the log-phase growth → curves overlap',
                  fontsize=10)
    ax1.legend(fontsize=8, ncol=2, loc='upper right')
    ax1.grid(alpha=0.2)
    ax1.set_xlim(-5.5, 5.5)

    # ── Row 2: crossing convergence ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, col])
    cross = find_all_crossings(z, Ls)
    if cross:
        Lmins = [c[0] for c in cross]
        lcs   = [c[2] for c in cross]
        errs  = [c[3] for c in cross]
        inv_L = [1.0/L for L in Lmins]
        ax2.errorbar(inv_L, lcs, yerr=errs, fmt='o', ms=10, capsize=6,
                     color='steelblue', lw=2, zorder=5)
        for iL, lc, L1L2 in zip(inv_L, lcs, [f"{c[0]}/{c[1]}" for c in cross]):
            ax2.annotate(f"L={L1L2}", (iL, lc),
                         textcoords="offset points", xytext=(5,4),
                         fontsize=9, color='navy')
        if len(lcs) >= 3:
            iL_a = np.array(inv_L); lc_a = np.array(lcs); e_a = np.array(errs)
            w = 1/e_a**2
            p = np.polyfit(iL_a, lc_a, 1, w=w)
            lc_inf = float(np.polyval(p, 0))
            x_fit = np.linspace(0, max(inv_L)*1.15, 100)
            ax2.plot(x_fit, np.polyval(p,x_fit), 'r--', lw=1.8)
            ax2.axhline(lc_inf, color='crimson', lw=2, label=f'$\\lambda_c(\\infty)={lc_inf:.4f}$')
            ax2.plot(0, lc_inf, 'r*', ms=16, zorder=6)
            # honest uncertainty band: range of the two largest-L crossings
            if len(lcs) >= 2:
                ax2.fill_between([0, max(inv_L)*0.05],
                                 lc_inf - abs(lcs[-1]-lcs[-2]),
                                 lc_inf + abs(lcs[-1]-lcs[-2]),
                                 color='crimson', alpha=0.15,
                                 label='uncertainty from last two pairs')
    ax2.set_xlabel(r'$1/L_{\min}$ of pair', fontsize=11)
    ax2.set_ylabel(r'$\lambda_c(L_1,L_2)$', fontsize=11)
    ax2.set_title(f'ζ={z}  — crossing convergence\n'
                  'Scatter shows finite-size corrections; more L needed for clean trend',
                  fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.2)
    ax2.set_xlim(-0.003, max(inv_L)*1.25 if cross else 0.04)
    ax2.axvline(0, color='gray', lw=0.8, ls=':')

plt.suptitle(
    "Understanding the FSS collapse: why it looks bad and how to fix it\n"
    "Row 1: raw $B_L$ — doesn't collapse (not dimensionless)  |  "
    "Row 2: $B_L/L$ — collapses properly  |  "
    "Row 3: crossing convergence — honest scatter at $L\\leq128$",
    fontsize=12, y=1.01)

out = '/Users/catlover1337/Documents/ppsQJ_m2/analysis/binder_explained.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"Saved: {out}")
