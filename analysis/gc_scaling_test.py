"""
g_c = alpha_c/w_c = lambda_c/(1-lambda_c) scaling test.

For each (L1,L2) pair and each zeta, extract g_c from the Binder crossing,
then plot g_c/zeta and g_c/sqrt(zeta) vs zeta.

If g_c/zeta flattens -> linear law (naive NLSM).
If g_c/sqrt(zeta) flattens -> square-root law (matched NLSM prediction).

Also plots the effective exponent phi_eff = d log g_c / d log zeta
for each L pair to show whether it is drifting toward 1 or staying near 0.5.
"""
import pickle, numpy as np, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import interp1d

# ── load ──────────────────────────────────────────────────────────────────────
OLD = pickle.load(open('/Users/catlover1337/Downloads/clone_aggregate(1).pkl','rb'))
AC  = pickle.load(open('/Users/catlover1337/Downloads/aggregate_runAC.pkl','rb'))
B   = pickle.load(open('/Users/catlover1337/Downloads/aggregate_B.pkl','rb'))
merged = {}
for src in (OLD, AC, B):
    for k,e in src.items(): merged[k] = e

by_zL = defaultdict(lambda: defaultdict(list))
for (L,lam,z),e in merged.items():
    bm = e.get('B_L_mean', np.nan)
    be = e.get('B_L_err',  np.nan)
    if not (np.isnan(bm) or bm <= 0 or np.isnan(be)):
        by_zL[round(z,3)][L].append((lam, bm, be))
for z in by_zL:
    for L in by_zL[z]:
        by_zL[z][L].sort(key=lambda t: t[0])

# ── pairwise crossing ─────────────────────────────────────────────────────────
def pairwise_crossing(z, L1, L2):
    """Return (lam_c, err) from B_L1 / B_L2 crossing at given zeta."""
    d1 = {t[0]:t[1] for t in by_zL[z].get(L1,[])}
    d2 = {t[0]:t[1] for t in by_zL[z].get(L2,[])}
    e1 = {t[0]:t[2] for t in by_zL[z].get(L1,[])}
    e2 = {t[0]:t[2] for t in by_zL[z].get(L2,[])}
    common = sorted(set(d1) & set(d2))
    if len(common) < 4:
        return None, None
    lams = np.array(common)
    diff = np.array([d2[l] - d1[l] for l in lams])
    zcs = np.where(np.diff(np.sign(diff)))[0]
    if len(zcs) == 0:
        return None, None
    i = zcs[0]
    denom = diff[i+1] - diff[i]
    if abs(denom) < 1e-12:
        return None, None
    t = -diff[i] / denom
    lc = float(lams[i] + t*(lams[i+1]-lams[i]))
    # propagate error
    dlam = lams[i+1] - lams[i]
    err_d = np.sqrt(e1[lams[i]]**2 + e2[lams[i]]**2 + e1[lams[i+1]]**2 + e2[lams[i+1]]**2)
    err = float(min(err_d * dlam / max(abs(denom), 1e-6), 0.04))
    return lc, err

# ── collect g_c for each (L-pair, zeta) ──────────────────────────────────────
# Use the decisive zeta window recommended in the collaborator note
zetas_target = sorted([z for z in by_zL if 0.05 <= z <= 1.0])
L_pairs = [(32,64), (48,96), (64,128), (96,128)]
pair_labels = {(32,64):'L=32/64', (48,96):'L=48/96',
               (64,128):'L=64/128', (96,128):'L=96/128'}
pair_colors = {(32,64):'#aec7e8', (48,96):'#ffbb78',
               (64,128):'#98df8a', (96,128):'#ff9896'}
pair_marker = {(32,64):'s', (48,96):'^', (64,128):'o', (96,128):'D'}

results = {pair: {} for pair in L_pairs}
for pair in L_pairs:
    L1, L2 = pair
    for z in zetas_target:
        lc, err = pairwise_crossing(z, L1, L2)
        if lc is not None and 0.01 < lc < 0.99:
            gc = lc / (1 - lc)
            gc_err = err / (1-lc)**2
            results[pair][z] = (gc, gc_err)

print("g_c values by L-pair:\n")
print(f"{'zeta':>7}", end="")
for pair in L_pairs:
    print(f"  {pair_labels[pair]:>12}", end="")
print()
print("-"*75)
for z in zetas_target:
    print(f"{z:>7.3f}", end="")
    for pair in L_pairs:
        if z in results[pair]:
            gc, ge = results[pair][z]
            print(f"  {gc:>6.4f}±{ge:.4f}", end="")
        else:
            print(f"  {'--':>12}", end="")
    print()

# ── Figure 1: g_c/zeta and g_c/sqrt(zeta) vs zeta ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.subplots_adjust(wspace=0.32)

# Panel A: raw g_c vs zeta
ax = axes[0]
for pair in L_pairs:
    zs  = sorted(results[pair])
    gcs = np.array([results[pair][z][0] for z in zs])
    ges = np.array([results[pair][z][1] for z in zs])
    ax.errorbar(zs, gcs, yerr=ges, fmt=pair_marker[pair]+'-',
                ms=7, lw=1.8, capsize=3,
                label=pair_labels[pair], color=pair_colors[pair])

# Reference curves
zfit = np.linspace(0.04, 1.05, 200)
ax.plot(zfit, np.sqrt(zfit), 'r-',  lw=2.5, label=r'$g_c=\sqrt{\zeta}$ (matched NLSM)', zorder=5)
ax.plot(zfit, zfit,          'b--', lw=2.0, label=r'$g_c=\zeta$ (naive NLSM)', zorder=5)
ax.set_xlabel(r'$\zeta$', fontsize=13)
ax.set_ylabel(r'$g_c = \lambda_c/(1-\lambda_c)$', fontsize=12)
ax.set_title('Raw critical rate ratio', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.2)
ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.4)

# Panel B: g_c/sqrt(zeta) vs zeta  (should be FLAT ~C if sqrt law)
ax = axes[1]
for pair in L_pairs:
    zs  = np.array(sorted(results[pair]))
    gcs = np.array([results[pair][z][0] for z in zs])
    ges = np.array([results[pair][z][1] for z in zs])
    ratio = gcs / np.sqrt(zs)
    ratio_err = ges / np.sqrt(zs)
    ax.errorbar(zs, ratio, yerr=ratio_err, fmt=pair_marker[pair]+'-',
                ms=7, lw=1.8, capsize=3,
                label=pair_labels[pair], color=pair_colors[pair])
ax.axhline(1.0, color='red', lw=2.5, ls='-',
           label=r'$C=1$ (Möbius prediction)', zorder=5)
ax.axhline(0.91, color='red', lw=1.5, ls='--',
           label=r'$C=0.91$ (best fit)', zorder=4, alpha=0.7)
ax.set_xlabel(r'$\zeta$', fontsize=13)
ax.set_ylabel(r'$g_c / \sqrt{\zeta}$', fontsize=12)
ax.set_title(r'$g_c/\sqrt{\zeta}$: flat $\Rightarrow$ square-root law confirmed',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.2)
ax.set_xlim(0, 1.05); ax.set_ylim(0, 3.0)
ax.text(0.5, 0.97, 'FLAT here → square-root law', transform=ax.transAxes,
        ha='center', va='top', fontsize=10, color='red',
        bbox=dict(boxstyle='round', facecolor='#ffe8e8', alpha=0.8))

# Panel C: g_c/zeta vs zeta  (should be FLAT ~C if linear law)
ax = axes[2]
for pair in L_pairs:
    zs  = np.array(sorted(results[pair]))
    gcs = np.array([results[pair][z][0] for z in zs])
    ges = np.array([results[pair][z][1] for z in zs])
    ratio = gcs / zs
    ratio_err = ges / zs
    ax.errorbar(zs, ratio, yerr=ratio_err, fmt=pair_marker[pair]+'-',
                ms=7, lw=1.8, capsize=3,
                label=pair_labels[pair], color=pair_colors[pair])
ax.axhline(1.0, color='blue', lw=2.5, ls='--',
           label=r'$C=1$ (naive NLSM)', zorder=5)
ax.set_xlabel(r'$\zeta$', fontsize=13)
ax.set_ylabel(r'$g_c / \zeta$', fontsize=12)
ax.set_title(r'$g_c/\zeta$: flat $\Rightarrow$ linear law confirmed',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.2)
ax.set_xlim(0, 1.05)
ax.text(0.5, 0.97, 'FLAT here → linear law', transform=ax.transAxes,
        ha='center', va='top', fontsize=10, color='blue',
        bbox=dict(boxstyle='round', facecolor='#e8e8ff', alpha=0.8))

plt.suptitle(
    r'Scaling test: $g_c = \alpha_c/w_c$ from Binder crossings at $L\leq 128$'
    '\n'
    r'Left: raw data vs two predictions   |   Centre: $g_c/\sqrt{\zeta}$ (flat $\Rightarrow$ matched-NLSM)   |   Right: $g_c/\zeta$ (flat $\Rightarrow$ naive NLSM)',
    fontsize=12, y=1.01)

out1 = '/Users/catlover1337/Documents/ppsQJ_m2/analysis/gc_scaling_test.png'
plt.savefig(out1, dpi=140, bbox_inches='tight')
print(f"\nSaved: {out1}")

# ── Figure 2: effective exponent phi_eff per L-pair ───────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 6))

for pair in L_pairs:
    zs  = np.array(sorted(results[pair]))
    gcs = np.array([results[pair][z][0] for z in zs])
    if len(zs) < 3:
        continue
    # d log g_c / d log zeta by finite differences on log-log
    log_z  = np.log(zs)
    log_gc = np.log(gcs)
    # central differences where possible
    phi = np.gradient(log_gc, log_z)
    # plot at midpoints
    ax2.plot(zs, phi, pair_marker[pair]+'-', ms=7, lw=1.8,
             label=pair_labels[pair], color=pair_colors[pair])

ax2.axhline(0.5, color='red',  lw=2.5, ls='-',  label=r'$\phi=1/2$ (matched NLSM)', zorder=5)
ax2.axhline(1.0, color='blue', lw=2.0, ls='--', label=r'$\phi=1$   (naive NLSM)',   zorder=4)
ax2.set_xlabel(r'$\zeta$', fontsize=13)
ax2.set_ylabel(r'$\phi_{\rm eff} = d\log g_c / d\log\zeta$', fontsize=12)
ax2.set_title(
    r'Effective exponent $\phi_{\rm eff}$ from pairwise crossings at $L\leq128$'
    '\n'
    r'Drifts toward 1 as $L\to\infty$ $\Rightarrow$ linear law.  Stays near 0.5 $\Rightarrow$ square-root law.',
    fontsize=11)
ax2.set_xlim(0.04, 1.05)
ax2.set_ylim(-0.2, 2.0)
ax2.legend(fontsize=10); ax2.grid(alpha=0.2)
ax2.fill_between([0.04,1.05], 0.4, 0.6, color='red',  alpha=0.07, label='_noleg')
ax2.fill_between([0.04,1.05], 0.9, 1.1, color='blue', alpha=0.07, label='_noleg')

out2 = '/Users/catlover1337/Documents/ppsQJ_m2/analysis/gc_phi_eff.png'
plt.savefig(out2, dpi=140, bbox_inches='tight')
print(f"Saved: {out2}")
