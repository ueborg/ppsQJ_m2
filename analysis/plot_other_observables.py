"""
Plot observables OTHER than B_L that we already have data for:
  S_var (entropy variance) — SRN diagnostic
  chi_k  (activity susceptibility) — dynamical PT
  covar_Sk (entropy-activity covariance) — sign change at λ_c
  theta_hat (saddle activity)
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

# Known lambda_c from global FSS for marking
known = json.load(open('/Users/catlover1337/Documents/ppsQJ_m2/analysis/global_fss_merged_v2.json'))
known_lc = {round(float(z),3): d['lc'] for z,d in known['per_zeta'].items()}

# Pick observables and organize per (zeta, L)
def get_obs(z, L, field):
    pts = []
    for (LL,lam,zz),e in merged.items():
        if LL==L and abs(zz-z)<0.005 and field in e:
            v = e[field]
            ev = e.get(field.replace('_mean','_err'), 0)
            if v is not None and not np.isnan(v):
                pts.append((lam, v, ev))
    pts.sort()
    if not pts: return None, None, None
    return (np.array([p[0] for p in pts]),
            np.array([p[1] for p in pts]),
            np.array([p[2] for p in pts]))

zetas_show = [0.05, 0.10, 0.20, 0.50]
Ls_show    = [32, 64, 96, 128]
colors_L = {32:'#4363d8', 64:'#3cb44b', 96:'#e6194b', 128:'#911eb4'}

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.subplots_adjust(hspace=0.55, wspace=0.35)

observables = [
    ('S_var_mean',    'Entropy variance Var($S_{L/2}$)',  'Skinner-Ruhman-Nahum: PEAKS at $\\lambda_c$'),
    ('chi_k_mean',    'Activity susceptibility $\\chi_k$', 'Click-count fluctuations: peak at dynamical PT'),
    ('covar_Sk_mean', 'Entropy–activity cov $C_{S,k}$',    'Changes sign at $\\lambda_c$'),
    ('S_mean',        'Half-chain entropy $S_{L/2}$',      'Smooth crossover; check L-scaling at $\\lambda_c$'),
]

for row, (field, fname, hint) in enumerate(observables):
    for col, z in enumerate(zetas_show):
        ax = axes[row, col]
        lc_ref = known_lc.get(round(z,3), None)

        ymax = -np.inf; ymin = np.inf
        for L in Ls_show:
            lams, vals, errs = get_obs(z, L, field)
            if lams is None: continue
            # Sanity: clip ridiculous outliers (NaN/inf safeguarded)
            mask = np.isfinite(vals)
            if mask.sum()<3: continue
            ax.errorbar(lams[mask], vals[mask],
                        yerr=np.minimum(errs[mask], np.abs(vals[mask])*0.5),
                        fmt='o-', ms=4, lw=1.2, capsize=2,
                        label=f'L={L}', color=colors_L[L])
            ymax = max(ymax, np.max(vals[mask]))
            ymin = min(ymin, np.min(vals[mask]))

        if lc_ref is not None and np.isfinite(ymax):
            ax.axvline(lc_ref, color='crimson', lw=1.6, ls='--', alpha=0.7,
                       label=f'$\\lambda_c^{{Binder}}={lc_ref:.3f}$')
        if field == 'covar_Sk_mean':
            ax.axhline(0, color='black', lw=0.8, ls=':', alpha=0.6)

        ax.set_xlabel(r'$\lambda$', fontsize=10)
        ax.set_ylabel(fname, fontsize=9)
        if row==0:
            ax.set_title(f'ζ = {z}', fontsize=12, fontweight='bold')
        if col==0:
            ax.text(-0.30, 0.5, hint, transform=ax.transAxes,
                    rotation=90, ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.2)

plt.suptitle(
    "Other transition diagnostics already in your data (besides $B_L$)\n"
    "Red dashed = $\\lambda_c$ from Binder collapse. Do the peaks/zeros agree?",
    fontsize=13, y=1.005)

out = '/Users/catlover1337/Documents/ppsQJ_m2/analysis/other_observables.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"Saved: {out}")
