"""Plot the scan results and analyze gap scaling."""
import pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = pickle.load(open('/Users/catlover1337/Documents/ppsQJ_m2/analysis/liouvillian/gap_data.pkl','rb'))
zetas = sorted(results.keys(), reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: gap vs lambda for each zeta
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(zetas)))
for zeta, color in zip(zetas, colors):
    r = results[zeta]
    ax.plot(r['lam'], r['gap'], 'o-', ms=5, color=color, label=f'$\\zeta={zeta}$')
    lc_emp = np.sqrt(zeta)/(1+np.sqrt(zeta))
    ax.axvline(lc_emp, color=color, ls=':', alpha=0.6)
ax.set_xlabel(r'$\lambda = \alpha/(\alpha+w)$', fontsize=12)
ax.set_ylabel('Liouvillian gap above SCGF', fontsize=12)
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.set_title('Lindbladian gap vs $\\lambda$ at L=6\nDotted lines: empirical $\\lambda_c$', fontsize=11)
ax.grid(alpha=0.3, which='both')

# Panel 2: gap at empirical critical line vs zeta
ax = axes[1]
# Extract gap at lambda closest to empirical lambda_c
gap_at_lc = []
peak_gap = []
peak_lam = []
for zeta in zetas:
    r = results[zeta]
    lc_emp = np.sqrt(zeta)/(1+np.sqrt(zeta))
    lams = np.array(r['lam'])
    gaps = np.array(r['gap'])
    idx_c = np.argmin(np.abs(lams - lc_emp))
    gap_at_lc.append(gaps[idx_c])
    idx_peak = np.argmax(gaps)
    peak_gap.append(gaps[idx_peak])
    peak_lam.append(lams[idx_peak])

ax.loglog(zetas, gap_at_lc, 'ko-', ms=8, label='gap at empirical $\\lambda_c$')
ax.loglog(zetas, peak_gap, 'rs-', ms=8, label='peak gap (any $\\lambda$)')
z_sm = np.logspace(-1.5, 0.05, 50)
# Reference scaling lines
ref_at_1 = gap_at_lc[0]
ax.loglog(z_sm, ref_at_1*z_sm, 'g--', alpha=0.6, label=r'$\propto \zeta$')
ax.loglog(z_sm, ref_at_1*np.sqrt(z_sm), 'b--', alpha=0.6, label=r'$\propto \sqrt{\zeta}$')
ax.set_xlabel(r'$\zeta$', fontsize=12); ax.set_ylabel('gap', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, which='both')
ax.set_title(r'Gap scaling with $\zeta$', fontsize=11)

plt.tight_layout()
plt.savefig('/Users/catlover1337/Documents/ppsQJ_m2/analysis/liouvillian/gap_scan.png', dpi=130, bbox_inches='tight')
print("Saved gap_scan.png")

# Print summary
print(f"\n{'zeta':>6} {'lambda_c_emp':>13} {'peak lam':>10} {'gap@lc':>10} {'peak gap':>10}")
print("-"*60)
for zeta, gat, pg, pl in zip(zetas, gap_at_lc, peak_gap, peak_lam):
    lc_emp = np.sqrt(zeta)/(1+np.sqrt(zeta))
    print(f"{zeta:>6.2f} {lc_emp:>13.4f} {pl:>10.2f} {gat:>10.5f} {pg:>10.5f}")

# Fit gap_at_lc ~ zeta^p
log_z = np.log(zetas)
log_g = np.log(gap_at_lc)
slope_lc = np.polyfit(log_z, log_g, 1)[0]
slope_peak = np.polyfit(log_z, np.log(peak_gap), 1)[0]
print(f"\nPower-law fits:")
print(f"  gap at empirical lambda_c:  ~ zeta^{slope_lc:.3f}")
print(f"  peak gap (any lambda):      ~ zeta^{slope_peak:.3f}")
