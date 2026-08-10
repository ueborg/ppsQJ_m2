"""Scan lambda for the two-replica Lindbladian and locate the MIPT signature."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
from two_replica import compute_two_replica_gap

L = 4
zetas = [1.0, 0.5, 0.2, 0.1, 0.05]
lambdas = np.linspace(0.10, 0.90, 17)

results = {}
print(f"Scanning two-replica gap at L={L}")
for zeta in zetas:
    print(f"\nzeta = {zeta}:")
    results[zeta] = {'lam':[], 'top':[], 'second':[], 'gap':[]}
    t0 = time.time()
    for lam in lambdas:
        eigs = compute_two_replica_gap(L, alpha=lam, w=1-lam, zeta=zeta, n_eigs=4)
        top = max(e.real for e in eigs)
        # Second largest distinct
        sec_candidates = [e.real for e in eigs if e.real < top - 1e-6]
        second = max(sec_candidates) if sec_candidates else top
        gap = top - second
        results[zeta]['lam'].append(lam)
        results[zeta]['top'].append(top)
        results[zeta]['second'].append(second)
        results[zeta]['gap'].append(gap)
        print(f"  lambda={lam:.2f}: top={top:+.4f}, gap={gap:.5f}")
    print(f"  ({time.time()-t0:.1f}s)")

pickle.dump(results, open('/Users/catlover1337/Documents/ppsQJ_m2/analysis/liouvillian/two_rep_data.pkl', 'wb'))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(zetas)))

ax = axes[0]
for zeta, color in zip(zetas, colors):
    r = results[zeta]
    ax.plot(r['lam'], r['gap'], 'o-', ms=6, color=color, label=f'$\\zeta={zeta}$')
    lc_emp = np.sqrt(zeta)/(1+np.sqrt(zeta))
    ax.axvline(lc_emp, color=color, ls=':', alpha=0.6)
ax.set_xlabel(r'$\lambda$', fontsize=12)
ax.set_ylabel('two-replica Lindbladian gap', fontsize=12)
ax.set_yscale('log')
ax.legend(fontsize=10, loc='upper right')
ax.set_title(f'Two-replica gap vs $\\lambda$ at L={L}', fontsize=11)
ax.grid(alpha=0.3, which='both')

# Look for gap minimum location
ax = axes[1]
min_lams = []
min_gaps = []
for zeta in zetas:
    r = results[zeta]
    gaps = np.array(r['gap']); lams = np.array(r['lam'])
    idx = np.argmin(gaps)
    min_lams.append(lams[idx])
    min_gaps.append(gaps[idx])

ax.semilogx(zetas, min_lams, 'ko-', ms=8, label='$\\lambda$ of gap minimum')
z_sm = np.logspace(-1.5, 0.05, 50)
ax.semilogx(z_sm, np.sqrt(z_sm)/(1+np.sqrt(z_sm)), 'b--', alpha=0.6,
            label=r'$\sqrt{\zeta}/(1+\sqrt{\zeta})$ (empirical)')
ax.semilogx(z_sm, 0.5*np.ones_like(z_sm), 'r:', alpha=0.5,
            label=r'$\lambda=1/2$ (Carollo at $\zeta=1$)')
ax.set_xlabel(r'$\zeta$', fontsize=12)
ax.set_ylabel(r'$\lambda$ at gap minimum', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')
ax.set_ylim(0, 1)
ax.set_title('Location of two-replica gap minimum', fontsize=11)

plt.tight_layout()
plt.savefig('/Users/catlover1337/Documents/ppsQJ_m2/analysis/liouvillian/two_rep_scan.png', dpi=130, bbox_inches='tight')
print("\nSaved two_rep_scan.png")

print(f"\n{'zeta':>6} {'lc_emp':>10} {'lambda_at_minimum':>20} {'min_gap':>10}")
for zeta, ml, mg in zip(zetas, min_lams, min_gaps):
    lc_emp = np.sqrt(zeta)/(1+np.sqrt(zeta))
    print(f"{zeta:>6.2f} {lc_emp:>10.4f} {ml:>20.4f} {mg:>10.5f}")
