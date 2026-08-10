"""
Two proper Binder crossing methods:
  Panel row 1: Crossing convergence (λ_c(L1,L2) vs 1/L_min → λ_c(∞))
  Panel row 2: FSS data collapse (all L curves vs scaled variable)
Shown for zeta = 0.10 and zeta = 0.20 (cleanest data).
"""
import pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# ── load ─────────────────────────────────────────────────────────────────────
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

# ── helpers ───────────────────────────────────────────────────────────────────
def get_arrays(z, L):
    pts = by_zL[z].get(L, [])
    if len(pts) < 4: return None, None, None
    lams = np.array([t[0] for t in pts])
    bm   = np.array([t[1] for t in pts])
    be   = np.array([t[2] for t in pts])
    return lams, bm, be

def pairwise_crossing(z, L1, L2):
    """Precise crossing of B_{L1} and B_{L2} at given zeta."""
    l1, b1, e1 = get_arrays(z, L1)
    l2, b2, e2 = get_arrays(z, L2)
    if l1 is None or l2 is None: return None, None, None
    common = sorted(set(l1.tolist()) & set(l2.tolist()))
    if len(common) < 4: return None, None, None
    lams = np.array(common)
    d1 = {lam: bv for lam, bv in zip(l1, b1)}
    d2 = {lam: bv for lam, bv in zip(l2, b2)}
    d1e = {lam: ev for lam, ev in zip(l1, e1)}
    d2e = {lam: ev for lam, ev in zip(l2, e2)}
    b1c = np.array([d1[l] for l in lams])
    b2c = np.array([d2[l] for l in lams])
    e1c = np.array([d1e[l] for l in lams])
    e2c = np.array([d2e[l] for l in lams])
    diff = b2c - b1c
    zcs = np.where(np.diff(np.sign(diff)))[0]
    if len(zcs) == 0: return None, None, None
    i = zcs[0]
    if abs(diff[i+1] - diff[i]) < 1e-12: return None, None, None
    t = -diff[i] / (diff[i+1] - diff[i])
    lc = float(lams[i] + t * (lams[i+1] - lams[i]))
    # Error by propagation through linear interpolation
    # err(lc) ≈ |dlc/d(diff)| * err(diff) at the interpolation point
    err_diff = float(np.sqrt(e1c[i]**2 + e2c[i]**2 + e1c[i+1]**2 + e2c[i+1]**2))
    dlam = lams[i+1] - lams[i]
    dlc_err = float(err_diff * dlam / max(abs(diff[i+1] - diff[i]), 1e-6))
    bc = float(0.5 * (b1c[i] + t*(b1c[i+1]-b1c[i]) + b2c[i] + t*(b2c[i+1]-b2c[i])))
    return lc, min(dlc_err, 0.05), bc

def collapse_residual(params, z, Ls, log_err=0.3):
    """Collapse residual for FSS: minimize scatter of B_L vs (λ-λc)*L^(1/ν)."""
    lc, nu = params
    if nu < 0.3 or nu > 5 or lc < 0.01 or lc > 0.98: return 1e9
    curves = []
    for L in Ls:
        lams, bm, be = get_arrays(z, L)
        if lams is None: continue
        x = (lams - lc) * L**(1.0/nu)
        order = np.argsort(x)
        curves.append((x[order], bm[order], be[order]))
    if len(curves) < 3: return 1e9
    tot = 0; n = 0
    for i in range(len(curves)):
        for j in range(i+1, len(curves)):
            x1, y1, e1 = curves[i]; x2, y2, e2 = curves[j]
            xlo, xhi = max(x1.min(), x2.min()), min(x1.max(), x2.max())
            if xhi <= xlo: continue
            mask = (x1 >= xlo) & (x1 <= xhi)
            if mask.sum() < 3: continue
            f2 = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value='extrapolate')
            y2i = f2(x1[mask])
            ly1 = np.log(np.maximum(y1[mask], 1e-8))
            ly2 = np.log(np.maximum(y2i, 1e-8))
            tot += np.sum(((ly1-ly2)/log_err)**2); n += mask.sum()
    return tot / max(n, 1)

def best_collapse(z, Ls):
    best = (np.inf, None)
    for lc0 in np.linspace(0.05, 0.65, 25):
        for nu0 in [1.5, 2.0, 2.5, 3.0]:
            q = collapse_residual((lc0, nu0), z, Ls)
            if q < best[0]: best = (q, (lc0, nu0))
    res = minimize(collapse_residual, best[1], args=(z, Ls),
                   method='Nelder-Mead', options={'xatol':1e-5, 'maxiter':600})
    return res.x[0], res.x[1], res.fun

# ─────────────────────────────────────────────────────────────────────────────
zetas = [0.10, 0.20]
Ls    = [32, 48, 64, 96, 128]
pairs = [(32,48),(48,64),(64,96),(96,128)]
colors_L = {32:'#4363d8', 48:'#f58231', 64:'#3cb44b', 96:'#e6194b', 128:'#911eb4'}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.subplots_adjust(hspace=0.38, wspace=0.32)

for col, z in enumerate(zetas):
    # ── ROW 0: crossing convergence ──────────────────────────────────────────
    ax = axes[0, col]
    lc_pairs, err_pairs, Lmins = [], [], []
    for L1, L2 in pairs:
        lc, err, bc = pairwise_crossing(z, L1, L2)
        if lc is None: continue
        lc_pairs.append(lc)
        err_pairs.append(err)
        Lmins.append(L1)
        print(f"ζ={z}, L={L1}/{L2}: λ_c = {lc:.4f} ± {err:.4f}")

    if len(lc_pairs) >= 2:
        inv_L = [1.0/L for L in Lmins]
        ax.errorbar(inv_L, lc_pairs, yerr=err_pairs, fmt='o', ms=9, capsize=5,
                    color='steelblue', ecolor='steelblue', lw=2, zorder=5)
        for i, (iL, lc, L1L2) in enumerate(zip(inv_L, lc_pairs,
                                                 [f"{L1}/{L2}" for L1,L2 in pairs[:len(lc_pairs)]])):
            ax.annotate(f"L={L1L2}", (iL, lc),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=9, color='steelblue')
        # extrapolate to 1/L → 0
        if len(lc_pairs) >= 3:
            iL_arr = np.array(inv_L); lc_arr = np.array(lc_pairs)
            w = 1.0 / np.array(err_pairs)**2
            p = np.polyfit(iL_arr, lc_arr, 1, w=w)
            lc_inf = float(np.polyval(p, 0))
            xfit = np.linspace(0, max(inv_L)*1.1, 100)
            ax.plot(xfit, np.polyval(p, xfit), 'r--', lw=1.8, label=f'linear extrap.')
            ax.axhline(lc_inf, color='crimson', lw=2.0, ls='-',
                       label=f'$\\lambda_c(\\infty) = {lc_inf:.4f}$')
            ax.plot(0, lc_inf, 'r*', ms=16, zorder=6)

    ax.set_xlabel(r'$1 / L_{\min}$  of pair', fontsize=12)
    ax.set_ylabel(r'$\lambda_c(L_1, L_2)$  pairwise crossing', fontsize=11)
    ax.set_title(f'ζ = {z}  —  crossing convergence', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.003, max([1.0/L for L in Lmins], default=0.05)*1.25)
    ax.legend(fontsize=10); ax.grid(alpha=0.25)
    ax.axvline(0, color='gray', lw=0.8, ls=':')

    # ── ROW 1: FSS collapse ───────────────────────────────────────────────────
    ax2 = axes[1, col]
    lc_fss, nu_fss, q = best_collapse(z, Ls)
    print(f"\nζ={z}: FSS collapse → λ_c={lc_fss:.4f}, ν={nu_fss:.3f}, q={q:.3f}")

    for L in Ls:
        lams, bm, be = get_arrays(z, L)
        if lams is None: continue
        x = (lams - lc_fss) * L**(1.0/nu_fss)
        # only plot around the crossing (|x| < some cutoff)
        mask = np.abs(x) < 5.0
        if mask.sum() < 3: continue
        ax2.errorbar(x[mask], bm[mask], yerr=be[mask], fmt='o', ms=6, lw=1.5,
                     capsize=2, label=f"L = {L}", color=colors_L[L], zorder=3)

    ax2.axvline(0, color='crimson', lw=2.0, ls='--', label='$\\lambda = \\lambda_c$', zorder=5)
    ax2.set_xlabel(r'$(\lambda - \lambda_c)\,L^{1/\nu}$', fontsize=13)
    ax2.set_ylabel(r'$B_L$', fontsize=13)
    ax2.set_title(
        f'ζ = {z}  —  FSS collapse\n'
        f'$\\lambda_c = {lc_fss:.4f}$, $\\nu = {nu_fss:.2f}$, quality = {q:.2f}',
        fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left'); ax2.grid(alpha=0.25)
    ax2.text(0.55, 0.95,
             'If collapse is good:\nall colours lie\non ONE curve',
             transform=ax2.transAxes, va='top', ha='left', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle(
    "Rigorous $\\lambda_c$ extraction — two methods\n"
    "Top: pairwise crossings converge to $\\lambda_c(\\infty)$ as $L\\to\\infty$\n"
    "Bottom: FSS collapse — all $L$ curves overlap on a single master curve at the correct $(\\lambda_c, \\nu)$",
    fontsize=12, y=1.02)

out = '/Users/catlover1337/Documents/ppsQJ_m2/analysis/binder_fss_proper.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f"\nSaved: {out}")
