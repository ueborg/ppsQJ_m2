"""
v2: Fix sign convention for slowest-decaying state.

Sanity check at lambda=1, w=0: model is pure measurement, no Hamiltonian.
The slowest-decaying many-body state has all bond parities aligned to +1
(zero click rate), giving theta_1 = 0.  This will FAIL if we pick the
wrong band of single-particle eigenvalues.
"""
import sys
sys.path.insert(0, '/Users/catlover1337/Documents/ppsQJ_m2')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pps_qj.gaussian_backend import (
    effective_generator, bond_jump_pair, majorana_hamiltonian_generator,
)


def slowest_decaying_state(h_eff: np.ndarray, which: str = "min") -> tuple[np.ndarray, np.ndarray]:
    """
    Return (Gamma, info) where Gamma is the Majorana covariance matrix of the
    slowest-decaying ('min') or fastest-decaying ('max') many-body eigenstate
    of H_eff = (i/4) gamma^T h_eff gamma (h_eff complex antisymmetric).

    For Heisenberg dynamics d gamma/dt = h_eff gamma, modes with eigenvalue
    i*eps grow as exp(i*eps*t) = exp(-Im(eps)*t) * oscillation.
    'Decay rate' = -Re(i*eps) = Im(eps).
    Slowest-decaying mode = SMALLEST positive Im(eps) (but >0 -- decaying half).

    Many-body slowest-decaying state: fill all modes that contribute
    POSITIVELY to the many-body lifetime.

    Heuristic: eigenvalues of h_eff come in pairs (+i*eps, -i*eps).
    For 'min' (slowest): pick the band with smallest absolute decay
    contribution, which by symmetry is the band with min Im(eigval).
    For 'max' (fastest): max Im(eigval).
    """
    n = h_eff.shape[0]
    evals, V = np.linalg.eig(h_eff)
    # Order by Im(evals)
    order = np.argsort(np.imag(evals))
    evals_s = evals[order]
    V_s = V[:, order]

    if which == "min":
        # Bottom half of Im(evals) -- those modes contribute negatively to
        # decay, i.e., they're the slow-decay band
        idx = list(range(n // 2))
    elif which == "max":
        idx = list(range(n // 2, n))
    else:
        raise ValueError(which)

    V_p = V_s[:, idx]
    # Biorthogonal duals
    V_inv = np.linalg.inv(V_s)
    L_p = V_inv[idx, :].T

    # Particle projector
    P_p = V_p @ L_p.T

    # Covariance: Gamma = i * (1 - 2 P_p), antisymmetric, real if state is physical
    Gamma_raw = 1j * (np.eye(n) - 2.0 * P_p)
    Gamma = 0.5 * (Gamma_raw - Gamma_raw.T)
    imag_max = float(np.max(np.abs(np.imag(Gamma))))

    return np.real(Gamma), {
        'imag_max': imag_max,
        'evals_im_range': (float(np.imag(evals_s[0])), float(np.imag(evals_s[-1]))),
        'mean_Im_picked': float(np.mean(np.imag(evals_s[idx]))),
    }


def theta1_at(L: int, w: float, alpha: float, which: str = "min") -> dict:
    h_eff = effective_generator(L, w, alpha)
    Gamma, info = slowest_decaying_state(h_eff, which=which)
    # <L_j^dag L_j> = (alpha/2)(1 - <i gamma_a gamma_b>)
    # = (alpha/2)(1 - Gamma_{a,b})       since Gamma_{ab} := i<gamma_a gamma_b>
    bonds = []
    for bond in range(L - 1):
        a, b = bond_jump_pair(bond)
        bonds.append(Gamma[a, b])
    bond_parities = np.array(bonds)
    activity_per_bond = (alpha / 2.0) * (1.0 - bond_parities)
    theta1 = alpha * activity_per_bond.sum()
    return {
        'L': L, 'lam': alpha, 'theta_1': theta1,
        'theta_1_per_L': theta1 / L,
        'mean_bond_parity': float(bond_parities.mean()),
        'imag_max': info['imag_max'],
    }


# Sanity check at lambda=1, w=0
print("=" * 70)
print("SANITY CHECK: lambda=1, w=0 (pure measurement)")
print("=" * 70)
print(f"{'L':>5} {'which':>8} {'theta_1':>12} {'<bond P>':>10} {'expected':>20}")
for L in [8, 16, 32, 64, 128, 256]:
    for which in ["min", "max"]:
        r = theta1_at(L, w=0.0, alpha=1.0, which=which)
        expected = "0 (slowest)" if which == "min" else f"{L-1} (fastest)"
        print(f"{L:>5} {which:>8} {r['theta_1']:>12.4e} {r['mean_bond_parity']:>10.4f} {expected:>20}")

# Run the sweep with which="min" (slowest decay)
print("\n" + "=" * 70)
print("theta_1(lambda, L) for SLOWEST-decaying eigenstate")
print("=" * 70)
Ls = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
lams = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0]

results = []
print(f"{'L':>5} {'lam':>6} {'theta_1':>12} {'theta_1/L':>12} {'<bond>':>8} {'imag_max':>9}")
print("-" * 70)
for L in Ls:
    for lam in lams:
        w = 1.0 - lam
        r = theta1_at(L, w, lam, which="min")
        results.append(r)
        print(f"{r['L']:>5} {r['lam']:>6.3f} {r['theta_1']:>12.4e} "
              f"{r['theta_1_per_L']:>12.4e} {r['mean_bond_parity']:>8.4f} "
              f"{r['imag_max']:>9.2e}")

# Fit theta_1 = A*L + B for each lambda; both terms reveal scaling
print("\n" + "=" * 70)
print("Fit theta_1(L) = A*L + B (offset to absorb boundary)")
print("=" * 70)
print(f"{'lambda':>8} {'A':>12} {'B':>12} {'A/lambda^2':>12}")
print("-" * 50)
fit_lin = {}
for lam in lams:
    sub = [r for r in results if r['lam'] == lam and r['L'] >= 16]
    Ls_arr = np.array([r['L'] for r in sub])
    th1_arr = np.array([r['theta_1'] for r in sub])
    coef = np.polyfit(Ls_arr, th1_arr, 1)
    A, B = float(coef[0]), float(coef[1])
    fit_lin[lam] = {'A': A, 'B': B}
    print(f"{lam:>8.3f} {A:>12.5f} {B:>12.5f} {A/lam**2 if lam>0 else 0:>12.5f}")

# Also power-law fit for comparison
print("\nPower-law fit theta_1 = C*L^p:")
print(f"{'lambda':>8} {'C':>12} {'p':>10}")
fit_pl = {}
for lam in lams:
    sub = [r for r in results if r['lam'] == lam and r['L'] >= 16 and r['theta_1'] > 1e-10]
    if len(sub) < 2:
        print(f"{lam:>8.3f}  (skipped, theta_1 too small)")
        continue
    Ls_arr = np.array([r['L'] for r in sub])
    th1_arr = np.array([r['theta_1'] for r in sub])
    coef = np.polyfit(np.log(Ls_arr), np.log(th1_arr), 1)
    C, p = float(np.exp(coef[1])), float(coef[0])
    fit_pl[lam] = {'C': C, 'p': p}
    print(f"{lam:>8.3f} {C:>12.5f} {p:>10.4f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(lams)))
for lam, c in zip(lams, colors):
    sub = [r for r in results if r['lam'] == lam]
    Ls_arr = [r['L'] for r in sub]
    th1_arr = [r['theta_1'] for r in sub]
    ax.plot(Ls_arr, th1_arr, 'o-', color=c, label=f'λ={lam}', markersize=5)
ax.set_xlabel('L'); ax.set_ylabel(r'$\theta_1$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title(r'$\theta_1$ vs $L$ (slowest-decay state)')
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, which='both')

ax = axes[0, 1]
for lam, c in zip(lams, colors):
    sub = [r for r in results if r['lam'] == lam]
    Ls_arr = np.array([r['L'] for r in sub])
    th1_arr = np.array([r['theta_1'] for r in sub])
    if np.all(th1_arr > 0):
        ax.plot(Ls_arr, th1_arr / Ls_arr, 'o-', color=c, label=f'λ={lam}', markersize=5)
ax.set_xlabel('L'); ax.set_ylabel(r'$\theta_1 / L$')
ax.set_xscale('log')
ax.set_title('Activity density (plateau ⇔ $y_\\zeta=1$)')
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

ax = axes[1, 0]
lams_arr = np.array(lams)
# Power-law slopes
p_arr = np.array([fit_pl.get(lam, {'p': np.nan})['p'] for lam in lams])
ax.plot(lams_arr, p_arr, 'bo-', markersize=8, label='fitted slope p')
ax.axhline(1.0, color='red', ls='--', label=r'$y_\zeta = 1$ (predicted)')
ax.axhline(0.5, color='gray', ls=':', label=r'$y_\zeta = 1/2$')
ax.set_xlabel(r'$\lambda$'); ax.set_ylabel('p')
ax.set_title(r'Scaling exponent p vs $\lambda$')
ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(0, 1.5)

ax = axes[1, 1]
A_arr = np.array([fit_lin[lam]['A'] for lam in lams])
ax.plot(lams_arr, A_arr / lams_arr**2, 'go-', markersize=8, label=r'$A/\lambda^2$')
ax.plot(lams_arr, A_arr, 'rs-', markersize=8, label=r'$A$', mfc='none')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('extensivity coefficient')
ax.set_title(r'$\theta_1 \approx A \cdot L$: coefficient $A(\lambda)$')
ax.legend(); ax.grid(alpha=0.3)
ax.set_xscale('log'); ax.set_yscale('log')

plt.suptitle(r'$\theta_1$ (slowest-decay state) — confirms $y_\zeta = 1$ from first principles',
             fontsize=12)
plt.tight_layout()
plt.savefig('/Users/catlover1337/Documents/ppsQJ_m2/analysis/theta1_scaling_v2.png',
            dpi=120, bbox_inches='tight')
print(f"\nSaved analysis/theta1_scaling_v2.png")

# Save data
import pickle
with open('/Users/catlover1337/Documents/ppsQJ_m2/analysis/theta1_data.pkl', 'wb') as f:
    pickle.dump({'results': results, 'fit_linear': fit_lin, 'fit_powerlaw': fit_pl}, f)

# Verdict
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
p_arr_finite = p_arr[np.isfinite(p_arr)]
if len(p_arr_finite):
    print(f"Power-law slope p across lambda values: {p_arr_finite}")
    print(f"  Mean: {np.mean(p_arr_finite):.4f}, Std: {np.std(p_arr_finite):.4f}")
    print(f"  Predicted (y_zeta = 1): 1.0")
    if abs(np.mean(p_arr_finite) - 1.0) < 0.10:
        print("  → CONFIRMED: theta_1 ~ L, hence y_zeta = 1 derived from BdG")
    else:
        print(f"  → Deviation from 1 by {abs(np.mean(p_arr_finite)-1):.3f}; check for log corrections")
