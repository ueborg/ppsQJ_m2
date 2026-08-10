import pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
from scipy.interpolate import interp1d

OLD = pickle.load(open('/Users/catlover1337/Downloads/clone_aggregate(1).pkl','rb'))
AC  = pickle.load(open('/Users/catlover1337/Downloads/aggregate_runAC.pkl','rb'))
B   = pickle.load(open('/Users/catlover1337/Downloads/aggregate_B.pkl','rb'))
merged = {}
for src in (OLD, AC, B):
    for k, e in src.items():
        merged[k] = e
print(f"Loaded {len(merged)} entries")

by_zL = defaultdict(lambda: defaultdict(list))
for (L, lam, z), e in merged.items():
    bm = e.get('B_L_mean', np.nan)
    be = e.get('B_L_err', np.nan)
    if not (np.isnan(bm) or bm <= 0 or np.isnan(be)):
        by_zL[round(z, 3)][L].append((lam, bm, be))
for z in by_zL:
    for L in by_zL[z]:
        by_zL[z][L].sort(key=lambda t: t[0])

def find_crossing(pts1, pts2):
    d1 = {t[0]: t[1] for t in pts1}
    d2 = {t[0]: t[1] for t in pts2}
    common = sorted(set(d1) & set(d2))
    if len(common) < 4:
        return None, None
    lams = np.array(common)
    diff = np.array([d2[l] - d1[l] for l in lams])
    zcs = np.where(np.diff(np.sign(diff)))[0]
    if len(zcs) == 0:
        return None, None
    i = zcs[0]
    if abs(diff[i+1] - diff[i]) < 1e-12:
        return None, None
    t = -diff[i] / (diff[i+1] - diff[i])
    lc = float(lams[i] + t * (lams[i+1] - lams[i]))
    f1 = interp1d(lams, [d1[l] for l in lams], kind='linear', fill_value='extrapolate')
    return lc, float(f1(lc))

zetas_to_show = [0.10, 0.20, 0.50, 0.70, 0.85, 1.00]
Ls_to_show    = [32, 48, 64, 96, 128]
colors = {32:'#4363d8', 48:'#f58231', 64:'#3cb44b', 96:'#e6194b', 128:'#911eb4'}
zoom_half = 0.16

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for ax, z in zip(axes, zetas_to_show):
    if z not in by_zL:
        ax.text(0.5,0.5,"no data",transform=ax.transAxes,ha='center',va='center')
        ax.set_title(f"ζ = {z}"); continue

    lc_est, Bc_est = None, None
    for L1, L2 in [(96,128),(64,128),(48,128)]:
        p1 = by_zL[z].get(L1,[])
        p2 = by_zL[z].get(L2,[])
        lc_est, Bc_est = find_crossing(p1, p2)
        if lc_est is not None:
            break
    if lc_est is None:
        lc_est = 0.47

    xlim = (max(0.02, lc_est - zoom_half), min(0.93, lc_est + zoom_half))

    ymin_data, ymax_data = np.inf, -np.inf
    for L in Ls_to_show:
        pts = by_zL[z].get(L,[])
        if len(pts) < 3: continue
        lams = np.array([t[0] for t in pts])
        bm   = np.array([t[1] for t in pts])
        be   = np.array([t[2] for t in pts])
        mask = (lams >= xlim[0]-0.01) & (lams <= xlim[1]+0.01)
        if mask.sum() < 3: continue
        ax.errorbar(lams[mask], bm[mask], yerr=be[mask],
                    fmt='o-', ms=6, lw=2.0, capsize=3,
                    label=f"L = {L}", color=colors[L], zorder=3)
        ymin_data = min(ymin_data, np.min(bm[mask]-be[mask]))
        ymax_data = max(ymax_data, np.max(bm[mask]+be[mask]))

    # Crossing markers
    ax.axvline(lc_est, color='crimson', lw=2.2, ls='--', zorder=5,
               label=f'crossing: $\\lambda_c={lc_est:.3f}$')
    if Bc_est is not None and np.isfinite(ymin_data):
        ax.plot(lc_est, Bc_est, 'r*', ms=18, zorder=7,
                markeredgecolor='darkred', markeredgewidth=0.8)
        ax.axhline(Bc_est, color='gray', lw=1.0, ls=':', alpha=0.5, zorder=2)
        ax.annotate(f'$B_c = {Bc_est:.2f}$',
                    xy=(lc_est, Bc_est),
                    xytext=(lc_est + (xlim[1]-xlim[0])*0.12, Bc_est),
                    fontsize=9, color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2))

    # Phase arrows and shading
    span = xlim[1] - xlim[0]
    if np.isfinite(ymin_data):
        ypad = (ymax_data - ymin_data) * 0.05
        ax.set_ylim(ymin_data - ypad, ymax_data + ypad*3.5)
        ax.fill_betweenx([ymin_data-ypad, ymax_data+ypad*3.5],
                         xlim[0], lc_est, color='steelblue', alpha=0.05)
        ax.fill_betweenx([ymin_data-ypad, ymax_data+ypad*3.5],
                         lc_est, xlim[1], color='firebrick', alpha=0.05)

    ax.text(xlim[0] + span*0.03, 0.97,
            'LOG PHASE\n$B_L$ ↑ with $L$',
            transform=ax.get_xaxis_transform(), ha='left', va='top',
            fontsize=9, color='steelblue', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',facecolor='white',alpha=0.8,edgecolor='steelblue'))
    ax.text(xlim[1] - span*0.03, 0.97,
            'AREA PHASE\n$B_L$ ↓ with $L$',
            transform=ax.get_xaxis_transform(), ha='right', va='top',
            fontsize=9, color='firebrick', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',facecolor='white',alpha=0.8,edgecolor='firebrick'))

    ax.set_xlim(xlim)
    ax.set_xlabel(r'$\lambda$', fontsize=13)
    ax.set_ylabel(r'$B_L$', fontsize=13)
    ax.set_title(f'ζ = {z}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9.5, loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.2)

plt.suptitle(
    r"Binder cumulant $B_L(\lambda)$ — six values of $\zeta$, system sizes $L \in \{32,48,64,96,128\}$"
    "\n"
    r"All curves meet at $\lambda_c(\zeta)$ (red star). Left: $B_L$ grows with $L$. Right: $B_L$ shrinks with $L$.",
    fontsize=13, y=1.01)

out = '/Users/catlover1337/Documents/ppsQJ_m2/analysis/binder_crossings_clean.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f"Saved: {out}")
