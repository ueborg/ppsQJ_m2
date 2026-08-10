"""Quick analysis of parity_resolved_data.pkl with proper exponent fits."""
import pickle, numpy as np

with open('/Users/catlover1337/Documents/ppsQJ_m2/analysis/parity_resolved_data.pkl', 'rb') as f:
    data = pickle.load(f)
results = data['results']

print('=' * 78)
print('Odd-L (degenerate parity doublet) data: L=5 vs L=7')
print('=' * 78)
print(f"  {'lambda':>7} {'K_eff(5)':>11} {'K_eff(7)':>11} {'ratio':>7} {'kappa':>7} "
      f"{'|delta|(7)':>11}")
lams = sorted({r['lam'] for r in results})
for lam in lams:
    r5 = next((r for r in results if r['L']==5 and r['lam']==lam), None)
    r7 = next((r for r in results if r['L']==7 and r['lam']==lam), None)
    if r5 and r7:
        ratio = r7['K_eff'] / r5['K_eff']
        kap = np.log(ratio) / np.log(7/5)
        print(f"  {lam:>7.3f} {r5['K_eff']:>11.4e} {r7['K_eff']:>11.4e} "
              f"{ratio:>7.3f} {kap:>7.3f} {abs(r7['delta_par'].real):>11.2e}")

print()
print('=' * 78)
print('All-L fit excluding L=4 (which is structurally anomalous, K_eff~0)')
print('=' * 78)
print(f"  {'lambda':>7} {'kappa':>8} {'A':>11}   "
      f"{'K_eff(5)':>11} {'K_eff(6)':>11} {'K_eff(7)':>11}")
for lam in lams:
    sub = sorted([r for r in results if r['L'] >= 5 and r['lam'] == lam],
                 key=lambda r: r['L'])
    Ls = np.array([r['L'] for r in sub], dtype=float)
    Ks = np.array([r['K_eff'] for r in sub])
    if all(Ks > 0):
        slope, intercept = np.polyfit(np.log(Ls), np.log(Ks), 1)
        A = np.exp(intercept)
    else:
        slope, A = np.nan, np.nan
    K5 = sub[0]['K_eff'] if len(sub) > 0 else float('nan')
    K6 = sub[1]['K_eff'] if len(sub) > 1 else float('nan')
    K7 = sub[2]['K_eff'] if len(sub) > 2 else float('nan')
    print(f"  {lam:>7.3f} {slope:>8.3f} {A:>11.4e}   "
          f"{K5:>11.4e} {K6:>11.4e} {K7:>11.4e}")

print()
print('=' * 78)
print('theta_+, theta_- L-scaling (should be ~ L for extensive decay rate)')
print('=' * 78)
print(f"  {'lambda':>7} {'|theta_+|/L (L=5)':>20} {'|theta_+|/L (L=6)':>20} "
      f"{'|theta_+|/L (L=7)':>20}")
for lam in lams:
    r5 = next((r for r in results if r['L']==5 and r['lam']==lam), None)
    r6 = next((r for r in results if r['L']==6 and r['lam']==lam), None)
    r7 = next((r for r in results if r['L']==7 and r['lam']==lam), None)
    t5 = abs(r5['theta_plus'].real)/5 if r5 else float('nan')
    t6 = abs(r6['theta_plus'].real)/6 if r6 else float('nan')
    t7 = abs(r7['theta_plus'].real)/7 if r7 else float('nan')
    print(f"  {lam:>7.3f} {t5:>20.5f} {t6:>20.5f} {t7:>20.5f}")

print()
print('=' * 78)
print('Effective theta_2 = K_+- * K_-+ / |delta|  (quadratic correction)')
print('  L=6 only (even-L, non-degenerate); L=5,7 in degenerate regime')
print('=' * 78)
print(f"  {'lambda':>7} {'L=6 theta_2_eff':>20}")
for lam in lams:
    r6 = next((r for r in results if r['L']==6 and r['lam']==lam), None)
    if r6:
        K_pm = r6['K_plus_minus']
        K_mp = r6['K_minus_plus']
        d = abs(r6['delta_par'].real)
        t2 = (K_pm * K_mp).real / d if d > 1e-12 else float('nan')
        print(f"  {lam:>7.3f} {t2:>20.4e}")
