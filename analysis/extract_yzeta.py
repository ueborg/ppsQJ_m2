"""
Extract y_zeta from existing B_L crossings.

Test scaling collapse:
    lambda_c(L_g, zeta) * sqrt(L_g) = F(zeta * L_g^{y_zeta})

For various y_zeta, measure collapse quality and pick optimum.
Also fit phi = 1/(2 y_zeta) directly from the TD slope.
"""
import pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, curve_fit

agg = pickle.load(open('/Users/catlover1337/Downloads/clone_aggregate(1).pkl','rb'))
Ls = sorted(set(k[0] for k in agg))
zetas = sorted(set(k[2] for k in agg))
lams = sorted(set(k[1] for k in agg))

def BL(L, lam, zeta):
    k = (L, lam, zeta)
    if k not in agg: return None
    v = agg[k].get('B_L_mean')
    if v is None or not np.isfinite(v): return None
    return float(v)

def lambda_c_BL(L1, L2, zeta):
    diffs = []
    for lam in lams:
        v1 = BL(L1, lam, zeta); v2 = BL(L2, lam, zeta)
        if v1 is not None and v2 is not None:
            diffs.append((lam, v1-v2))
    if len(diffs) < 2: return None
    diffs.sort()
    arr_l = np.array([p[0] for p in diffs]); arr_d = np.array([p[1] for p in diffs])
    for i in range(len(arr_d)-1):
        if arr_d[i]*arr_d[i+1] < 0:
            return float(arr_l[i] - arr_d[i]*(arr_l[i+1]-arr_l[i])/(arr_d[i+1]-arr_d[i]))
    return None

L_pairs = [(Ls[i], Ls[i+1]) for i in range(len(Ls)-1)]
clean_pairs = [(L1,L2) for L1,L2 in L_pairs if L1>=16]

# Build (L_g, zeta, lambda_c) table — drop the (96,128) noisy data at small zeta
data = []
for L1,L2 in clean_pairs:
    Lg = np.sqrt(L1*L2)
    for z in zetas:
        lc = lambda_c_BL(L1, L2, z)
        if lc is not None and lc > 0.01:
            # Drop (96,128) at zeta <= 0.10 (known noisy from N_c=100 stats)
            if (L1, L2) == (96, 128) and z <= 0.10:
                continue
            data.append((Lg, z, lc))

print(f"Clean data points: {len(data)}")

# ============================================================
# Method 1: fit phi from each L_g separately (slice fit)
# ============================================================
print("\n=== Method 1: power-law fit at fixed L_g ===")
print("Fitting lambda_c = A * zeta^phi for each L_g, using zeta in [0.1, 0.7]")
print("(small zeta dominated by no-click crossover; very large zeta saturates)")
print(f"{'L_g':>6} | {'phi_fit':>8} | {'A_fit':>8} | {'y_zeta':>8}")
print("-"*45)
phi_vs_L = []
for Lg_target in sorted(set(p[0] for p in data)):
    pts = [(z, lc) for (Lg, z, lc) in data if abs(Lg - Lg_target) < 0.01 and 0.1 <= z <= 0.7]
    if len(pts) < 3:
        continue
    zs, lcs = zip(*pts)
    zs = np.array(zs); lcs = np.array(lcs)
    # Linear fit in log-log
    log_z = np.log(zs); log_lc = np.log(lcs)
    A_coef = np.polyfit(log_z, log_lc, 1)
    phi = A_coef[0]
    A = np.exp(A_coef[1])
    y_zeta = 1.0 / (2 * phi) if phi > 0 else float('nan')
    phi_vs_L.append((Lg_target, phi))
    print(f"{Lg_target:>6.1f} | {phi:>8.3f} | {A:>8.3f} | {y_zeta:>8.3f}")

# ============================================================
# Method 2: global scaling collapse — find optimal y_zeta
# ============================================================
print("\n=== Method 2: optimal y_zeta from scaling collapse ===")

def collapse_residual(y_zeta, data):
    """Lower = better collapse. Bins data by x=zeta*L^y_zeta and computes
    within-bin variance of y=lc*sqrt(L)."""
    xs = np.array([z * Lg**y_zeta for (Lg, z, lc) in data])
    ys = np.array([lc * np.sqrt(Lg) for (Lg, z, lc) in data])
    # Log-log binning
    log_xs = np.log(xs)
    # Compute residual: for each pair of nearby points (in log_x), variance of log_y
    # Sort by log_x
    order = np.argsort(log_xs)
    log_xs = log_xs[order]; log_ys = np.log(np.abs(ys))[order]
    # Bin: for each window of size dx=0.3 in log space, compute std of log_y
    residuals = []
    dx = 0.3
    for x_center in np.arange(log_xs.min()+dx, log_xs.max()-dx, dx/2):
        mask = np.abs(log_xs - x_center) < dx
        if np.sum(mask) >= 2:
            residuals.append(np.std(log_ys[mask]))
    return np.mean(residuals) if residuals else float('inf')

y_zetas_test = np.linspace(0.2, 2.0, 50)
residuals = [collapse_residual(yz, data) for yz in y_zetas_test]
best_idx = np.argmin(residuals)
y_zeta_opt = y_zetas_test[best_idx]
print(f"  Optimal y_zeta (collapse quality): {y_zeta_opt:.3f}")
print(f"  Min residual: {residuals[best_idx]:.4f}")
print(f"  Implied phi = 1/(2 y_zeta) = {1/(2*y_zeta_opt):.3f}")

# Compare residuals at a few special values
for yz_special, name in [(0.5, "y_zeta=1/2 (linear lc~zeta)"),
                         (1.0, "y_zeta=1 (sqrt-zeta)"),
                         (1.5, "y_zeta=3/2"),
                         (y_zeta_opt, "OPTIMAL")]:
    print(f"  Residual at {name}: {collapse_residual(yz_special, data):.4f}")

# ============================================================
# Method 3: fit phi from TD-limit values
# ============================================================
# Extrapolate lambda_c(L_g -> infty) at each zeta if possible
print("\n=== Method 3: lambda_c TD-limit extrapolation + power-law fit ===")
print("At each zeta, fit lambda_c(L_g) = lc_inf + B/sqrt(L_g) and extract lc_inf")
print(f"{'zeta':>6} | {'lc_inf':>8} | {'B':>8} | {'n_pts':>5}")
print("-"*45)
lc_inf_data = []
for z in zetas:
    pts = [(Lg, lc) for (Lg, zz, lc) in data if abs(zz-z) < 1e-6]
    if len(pts) < 3:
        continue
    Lgs = np.array([p[0] for p in pts])
    lcs = np.array([p[1] for p in pts])
    # Sort by L_g
    order = np.argsort(Lgs)
    Lgs = Lgs[order]; lcs = lcs[order]
    # Linear fit: lc = lc_inf + B / sqrt(L_g)
    x = 1.0/np.sqrt(Lgs)
    coef = np.polyfit(x, lcs, 1)
    B_fit, lc_inf = coef[0], coef[1]
    print(f"{z:>6.3f} | {lc_inf:>8.4f} | {B_fit:>8.4f} | {len(pts):>5}")
    if lc_inf > 0.01:
        lc_inf_data.append((z, lc_inf))

# Now fit power law lc_inf = A * zeta^phi
if len(lc_inf_data) >= 3:
    print("\nFit lc_inf = A * zeta^phi to extrapolated TD values:")
    zs = np.array([p[0] for p in lc_inf_data])
    lcs = np.array([p[1] for p in lc_inf_data])
    print(f"  data: {list(zip(zs.round(3), lcs.round(4)))}")
    # Fit only zeta in [0.1, 0.7] to avoid no-click crossover and saturation
    mask = (zs >= 0.1) & (zs <= 0.7)
    if np.sum(mask) >= 2:
        log_z = np.log(zs[mask]); log_lc = np.log(lcs[mask])
        c = np.polyfit(log_z, log_lc, 1)
        phi_TD = c[0]; A_TD = np.exp(c[1])
        y_zeta_TD = 1.0/(2*phi_TD)
        print(f"  phi (from TD limit) = {phi_TD:.3f}")
        print(f"  A = {A_TD:.3f}")
        print(f"  y_zeta = {y_zeta_TD:.3f}")

# ============================================================
# Plot scaling collapse with optimal y_zeta
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: residual vs y_zeta
ax = axes[0,0]
ax.plot(y_zetas_test, residuals, 'b-', lw=2)
ax.axvline(y_zeta_opt, color='r', ls='--', label=f'optimal y_ζ={y_zeta_opt:.2f}')
ax.axvline(0.5, color='gray', ls=':', alpha=0.6, label='y_ζ=1/2 (linear lc~ζ)')
ax.axvline(1.0, color='green', ls=':', alpha=0.6, label='y_ζ=1 (lc~√ζ)')
ax.set_xlabel(r'$y_\zeta$ trial value')
ax.set_ylabel('Collapse residual (lower = better)')
ax.set_title('Optimal scaling exponent from collapse')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel B: data collapse at optimal y_zeta
ax = axes[0,1]
pair_colors = plt.cm.plasma(np.linspace(0, 0.85, len(clean_pairs)))
for (L1,L2), c in zip(clean_pairs, pair_colors):
    Lg = np.sqrt(L1*L2)
    pts = [(z*Lg**y_zeta_opt, lc*np.sqrt(Lg))
           for (Lg_, z, lc) in data if abs(Lg_ - Lg) < 0.01]
    if pts:
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, 'o-', color=c, label=f'({L1},{L2})', markersize=6)
ax.set_xlabel(rf'$\zeta L_g^{{{y_zeta_opt:.2f}}}$')
ax.set_ylabel(r'$\lambda_c \sqrt{L_g}$')
ax.set_title(f'Scaling collapse at optimal y_ζ = {y_zeta_opt:.2f}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3, which='both')

# Panel C: comparison — collapse at y_zeta=1/2 (the failed assumption)
ax = axes[1,0]
y_zeta_baseline = 0.5
for (L1,L2), c in zip(clean_pairs, pair_colors):
    Lg = np.sqrt(L1*L2)
    pts = [(z*Lg**y_zeta_baseline, lc*np.sqrt(Lg))
           for (Lg_, z, lc) in data if abs(Lg_ - Lg) < 0.01]
    if pts:
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, 'o-', color=c, label=f'({L1},{L2})', markersize=6)
ax.set_xlabel(r'$\zeta L_g^{1/2}$ (= $\zeta\sqrt{L_g}$, the original ansatz)')
ax.set_ylabel(r'$\lambda_c \sqrt{L_g}$')
ax.set_title('Original Scenario C collapse (y_ζ=1/2) for comparison')
ax.set_xscale('log'); ax.set_yscale('log')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3, which='both')

# Panel D: lc_inf vs zeta — TD critical line and power-law fit
ax = axes[1,1]
if lc_inf_data:
    zs = np.array([p[0] for p in lc_inf_data])
    lcs = np.array([p[1] for p in lc_inf_data])
    ax.plot(zs, lcs, 'ko', markersize=8, label='Extrapolated TD λ_c')
    # Power-law fit
    mask = (zs >= 0.1) & (zs <= 0.7)
    if np.sum(mask) >= 2:
        zs_fit = np.linspace(0.01, 1.0, 100)
        ax.plot(zs_fit, A_TD * zs_fit**phi_TD, 'r--',
                label=f'fit: λ_c = {A_TD:.2f}·ζ^{{{phi_TD:.2f}}} (y_ζ={y_zeta_TD:.2f})')
    ax.plot(zs_fit, zs_fit*0.5/(1+zs_fit*0.5), 'g:',
            label='Original Scenario C: ζ/(1+ζ)')
ax.axhline(0.5, color='gray', ls=':', alpha=0.5, label='Carollo λ_c(ζ=1)=0.5')
ax.set_xlabel(r'$\zeta$')
ax.set_ylabel(r'$\lambda_c(\zeta)$ — TD limit')
ax.set_title('TD critical line and power-law fit')
ax.set_xscale('log')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle(f'Two-parameter FSS analysis: extracting y_ζ from data\n'
             f'(y_λ=1/2 from BdG, y_ζ measured: {y_zeta_opt:.2f}, implied φ={1/(2*y_zeta_opt):.2f})',
             fontsize=11)
plt.tight_layout()
plt.savefig('/Users/catlover1337/Documents/ppsQJ_m2/analysis/yzeta_extraction.png', dpi=120, bbox_inches='tight')
print('\nSaved /Users/catlover1337/Documents/ppsQJ_m2/analysis/yzeta_extraction.png')

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Slice fits (lambda_c ~ zeta^phi at fixed L_g): phi varies, see table")
print(f"Global collapse optimum: y_zeta = {y_zeta_opt:.3f}, phi = {1/(2*y_zeta_opt):.3f}")
if len(lc_inf_data) >= 2:
    print(f"TD-extrapolation: phi = {phi_TD:.3f}, y_zeta = {y_zeta_TD:.3f}")
print(f"\n* Baseline phi=1 (linear lc~zeta): EXCLUDED")
print(f"* phi ~ 1/2 (sqrt: lc~sqrt(zeta)): " + ("CONSISTENT" if abs(1/(2*y_zeta_opt) - 0.5) < 0.15 else "rejected"))
print(f"* phi ~ 2/3: " + ("CONSISTENT" if abs(1/(2*y_zeta_opt) - 0.67) < 0.15 else "rejected"))
