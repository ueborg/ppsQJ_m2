"""Scan lambda at fixed zeta to find MIPT signatures."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pps_lindbladian import compute_gap

zetas = [1.0, 0.5, 0.2, 0.1, 0.05]
lambdas = np.linspace(0.05, 0.95, 19)
L = 6

results = {}
print(f"Scanning L={L}, lambdas={len(lambdas)}, zetas={zetas}")
for zeta in zetas:
    results[zeta] = {'lam':[], 'gap':[]}
    for lam in lambdas:
        scgf, gap, eigs = compute_gap(L=L, alpha=lam, w=1-lam, zeta=zeta)
        results[zeta]['lam'].append(lam)
        results[zeta]['gap'].append(gap)
    print(f"  zeta={zeta}: max gap = {max(results[zeta]['gap']):.4f}")

pickle.dump(results, open('/Users/catlover1337/Documents/ppsQJ_m2/analysis/liouvillian/gap_data.pkl', 'wb'))
print("Saved gap_data.pkl")

# Make the plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(zetas)))

ax = axes[0]
for zeta, color in zip(zetas, colors):
    r = results[zeta]
    ax.plot(r['lam'], r['gap'], 'o-', ms=5, color=color, label=f'$\\zeta={zeta}$')
    lc_emp = np.sqrt(zeta)/(1+np.sqrt(zeta))
    ax.axvline(lc_emp, color=color, ls=':', alpha=0.6)
ax.set_xlabel(r'$\lambda$', fontsize=12)
ax.set_ylabel('Liouvillian gap', fontsize=12)
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.set_title(f'Lindbladian gap (L={L})\nDotted: $\\lambda_c^{{emp}} = \\sqrt{{\\zeta}}/(1+\\sqrt{{\\zeta}})$', fontsize=11)
ax.grid(alpha=0.3, which='both')

ax = axes[1]
gap_at_lc = []; peak_gap = []
for zeta in zetas:
    r = results[zeta]
    lc_emp = np.sqrt(zeta)/(1+np.sqrt(zeta))
    lams = np.array(r['lam']); gaps = np.array(r['gap'])
    idx_c = np.argmin(np.abs(lams - lc_emp))
    gap_at_lc.append(gaps[idx_c])
    peak_gap.append(np.max(gaps))

ax.loglog(zetas, gap_at_lc, 'ko-', ms=8, label='gap at $\\lambda_c^{emp}$')
ax.loglog(zetas, peak_gap, 'rs-', ms=8, label='peak gap')
z_sm = np.logspace(-1.5, 0.05, 50)
ax.loglog(z_sm, gap_at_lc[0]*z_sm, 'g--', alpha=0.5, label=r'$\propto \zeta$')
ax.loglog(z_sm, gap_at_lc[0]*np.sqrt(z_sm), 'b--', alpha=0.5, label=r'$\propto \sqrt{\zeta}$')
ax.set_xlabel(r'$\zeta$', fontsize=12); ax.set_ylabel('gap', fontsize=12)
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
ax.set_title(r'Gap scaling with $\zeta$', fontsize=11)

plt.tight_layout()
plt.savefig('/Users/catlover1337/Documents/ppsQJ_m2/analysis/liouvillian/gap_scan.png', dpi=130, bbox_inches='tight')

print(f"\n{'zeta':>6} {'lc_emp':>9} {'gap@lc':>10} {'peak gap':>10}")
for zeta, gat, pg in zip(zetas, gap_at_lc, peak_gap):
    lc_emp = np.sqrt(zeta)/(1+np.sqrt(zeta))
    print(f"{zeta:>6.2f} {lc_emp:>9.4f} {gat:>10.5f} {pg:>10.5f}")

slope_lc = np.polyfit(np.log(zetas), np.log(gap_at_lc), 1)[0]
slope_peak = np.polyfit(np.log(zetas), np.log(peak_gap), 1)[0]
print(f"\nPower fits:  gap@lc ~ zeta^{slope_lc:.3f},  peak ~ zeta^{slope_peak:.3f}")
