"""
analysis/compute_theta1_exact.py

Exact Liouville-space computation of theta_1 = d theta / d zeta |_{zeta=0}
for the QJ-PPS model at small L. Closes Open Question A from
theory/HANDOFF.md: removes the BdG biorthogonal-covariance construction
(which gives unphysical Gamma for lambda > 0.3) and replaces it with the
rigorous SCGF derivative on the full 4^L-dim Liouville space.

Definition (rigorous, SCGF):
    L_zeta[rho] = -i(H_eff rho - rho H_eff^dag)
                  + zeta * sum_j J_j rho J_j^dag
                = L_0 + zeta * J
    theta(zeta) = leading eigenvalue of L_zeta (largest Re part)
                = theta_0 + zeta * theta_1 + O(zeta^2)
    theta_1     = <ell_0 | J | r_0> / <ell_0 | r_0>
where (r_0, ell_0) are right/left eigenmatrices of L_0 at theta_0.

Relation to the BdG-doc formula. The previous BdG result
    theta_1^BdG = alpha * sum_j <J_j^dag J_j>_slow
                = (alpha^2 / 2) * sum_j (1 - Gamma_{2j, 2j+3})
is NOT the same object. With r_0 = |R_0><L_0| factorized,
    <ell|J|r>/<ell|r> = sum_j <L_0|J_j|R_0> <L_0|J_j^dag|R_0> / <L_0|R_0>^2
involves two distinct biorthogonal matrix elements. By the biorthogonal
resolution of identity sum_k |R_k><L_k| = 1,
    <L_0|J_j^dag J_j|R_0> = sum_k <L_0|J_j^dag|R_k><L_k|J_j|R_0>
so the BdG 'activity' contains the k=0 diagonal piece plus k != 0
cross-state pieces. Both objects are extensive in L for a locally
correlated slow state, so the y_zeta = 1 scaling claim is robust to
this distinction -- absolute values need not match.

For L <= 6, dim(L_super) = 4^L <= 4096; dense scipy.linalg.eig
handles it in ~10 s/instance on a laptop.

Convention: column-stacking vec, so vec(A rho B) = (B^T (x) A) vec(rho).
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/Users/catlover1337/Documents/ppsQJ_m2')

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from time import time

from pps_qj.gaussian_backend import effective_generator, bond_jump_pair


# ============================================================
# BdG-style theta_1 (copied here to avoid importing the
# side-effect-heavy analysis/compute_theta1.py script)
# ============================================================
def _slowest_decay_bdg(h_eff: np.ndarray) -> np.ndarray:
    """Single-particle slowest-decay band -> Majorana covariance Gamma.
    Same construction as analysis/compute_theta1.py.

    Returns real antisymmetric Gamma (possibly unphysical for large lambda --
    that's the whole point of the BdG breakdown we are now bypassing).
    """
    n = h_eff.shape[0]
    evals, V = np.linalg.eig(h_eff)
    order = np.argsort(np.imag(evals))
    V_s = V[:, order]
    idx = list(range(n // 2))           # bottom half = slow band
    V_p = V_s[:, idx]
    V_inv = np.linalg.inv(V_s)
    L_p = V_inv[idx, :].T
    P_p = V_p @ L_p.T                   # biorthogonal projector (non-unitary)
    Gamma_raw = 1j * (np.eye(n) - 2.0 * P_p)
    Gamma = 0.5 * (Gamma_raw - Gamma_raw.T)
    return np.real(Gamma)


def theta1_at_bdg(L: int, w: float, alpha: float, which: str = "min") -> dict:
    """BdG reproduction of analysis/compute_theta1.py: theta1 = alpha * sum_j <J_j^dag J_j>."""
    h_eff = effective_generator(L, w, alpha)
    Gamma = _slowest_decay_bdg(h_eff)
    bond_parities = np.array([Gamma[bond_jump_pair(b)[0], bond_jump_pair(b)[1]]
                              for b in range(L - 1)])
    activity = (alpha / 2.0) * (1.0 - bond_parities)
    theta_1 = alpha * float(activity.sum())
    return {'theta_1': theta_1, 'mean_bond_parity': float(bond_parities.mean())}


# ============================================================
# Jordan-Wigner construction (dense, basis-ordered as binary)
# ============================================================
# Basis: |s_0 s_1 ... s_{L-1}>, with s_0 in the first tensor slot.
# c|1> = |0>, c^dag|0> = |1>, so c_loc = [[0,1],[0,0]], cdag_loc = c_loc.T.

I2 = np.eye(2, dtype=np.complex128)
ZZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
C_LOC = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
CD_LOC = C_LOC.conj().T


def _embed(L: int, j: int, op2: np.ndarray, jw: bool) -> np.ndarray:
    out = np.array([[1.0]], dtype=np.complex128)
    for k in range(L):
        if k < j and jw:
            out = np.kron(out, ZZ)
        elif k == j:
            out = np.kron(out, op2)
        else:
            out = np.kron(out, I2)
    return out


def jw_fermions(L: int):
    """Return (c, cdag) -- lists of length L, each a 2^L x 2^L dense matrix."""
    c = [_embed(L, j, C_LOC, jw=True) for j in range(L)]
    cdag = [_embed(L, j, CD_LOC, jw=True) for j in range(L)]
    return c, cdag


def verify_jw(c, cdag, L):
    D = 2 ** L
    I_D = np.eye(D, dtype=np.complex128)
    for j in range(L):
        for k in range(L):
            anti_cc = c[j] @ c[k] + c[k] @ c[j]
            if not np.allclose(anti_cc, 0.0, atol=1e-12):
                raise ValueError(f"{{c_{j},c_{k}}} != 0")
            anti_ccd = c[j] @ cdag[k] + cdag[k] @ c[j]
            target = I_D if j == k else np.zeros_like(I_D)
            if not np.allclose(anti_ccd, target, atol=1e-12):
                raise ValueError(f"{{c_{j},cdag_{k}}} != delta_{{jk}} I")


# ============================================================
# Model: H, H_eff, J_j list
# ============================================================
def build_model(L: int, lam: float):
    """QJ-PPS on a chain of L sites with OBC.

        H     = w * sum_{j=0}^{L-2} (c_j^dag c_{j+1} + h.c.),   w = 1 - lambda
        d_j   = (c_j + c_j^dag - c_{j+1} + c_{j+1}^dag) / 2
              = (gamma_{2j} - i gamma_{2j+3}) / 2
        J_j   = sqrt(alpha) * d_j,     alpha = lambda
        K     = sum_j J_j^dag J_j
        H_eff = H - i/2 K

    Returns: (H, H_eff, [J_0, ..., J_{L-2}], K).
    """
    alpha = float(lam)
    w = 1.0 - alpha
    c, cd = jw_fermions(L)
    D = 2 ** L

    H = np.zeros((D, D), dtype=np.complex128)
    for j in range(L - 1):
        H += w * (cd[j] @ c[j + 1] + cd[j + 1] @ c[j])
    assert np.allclose(H, H.conj().T, atol=1e-12), "H not Hermitian"

    jumps = []
    K = np.zeros((D, D), dtype=np.complex128)
    for j in range(L - 1):
        d_j = 0.5 * (c[j] + cd[j] - c[j + 1] + cd[j + 1])
        J_j = np.sqrt(alpha) * d_j
        jumps.append(J_j)
        K += J_j.conj().T @ J_j
    assert np.allclose(K, K.conj().T, atol=1e-12), "K not Hermitian"

    H_eff = H - 0.5j * K
    return H, H_eff, jumps, K


# ============================================================
# Liouville-space superoperators (column-stacking vectorization)
# ============================================================
def build_L0(H_eff: np.ndarray) -> np.ndarray:
    """L_0 = -i(I (x) H_eff - H_eff^* (x) I).

    With column-stacking vec, vec(H_eff rho)     = (I (x) H_eff) vec(rho)
                              vec(rho H_eff^dag) = (H_eff^* (x) I) vec(rho).
    """
    D = H_eff.shape[0]
    I_D = np.eye(D, dtype=np.complex128)
    return -1j * (np.kron(I_D, H_eff) - np.kron(H_eff.conj(), I_D))


def build_J_super(jumps) -> np.ndarray:
    """J_super = sum_j J_j^* (x) J_j."""
    D = jumps[0].shape[0]
    out = np.zeros((D * D, D * D), dtype=np.complex128)
    for J in jumps:
        out += np.kron(J.conj(), J)
    return out


def vec_to_mat(v: np.ndarray, D: int) -> np.ndarray:
    """Inverse of column-stacking vec."""
    return v.reshape(D, D, order='F')


# ============================================================
# Leading eigenmode of L_0 (biorthonormal)
# ============================================================
def solve_leading(L0_super: np.ndarray):
    """Dense diagonalization; pick the largest-Re eigenvalue. Returns
        (theta_0, r_vec, ell_vec, gap, vals_sorted)
    with biorthonormal <ell|r> = ell^H r = 1. scipy.linalg.eig with
    left=right=True returns paired columns: vecsL[:, k]^H @ A = vals[k] *
    vecsL[:, k]^H, so vecsR[:, k] and vecsL[:, k] correspond to the same
    eigenvalue.
    """
    vals, vecsL, vecsR = la.eig(L0_super, left=True, right=True)
    order = np.argsort(-np.real(vals))
    vals_sorted = vals[order]
    R_sorted = vecsR[:, order]
    L_sorted = vecsL[:, order]

    theta_0 = vals_sorted[0]
    r_vec = R_sorted[:, 0]
    ell_vec = L_sorted[:, 0]
    gap = float(np.real(theta_0) - np.real(vals_sorted[1]))

    overlap = np.vdot(ell_vec, r_vec)  # = ell^H r
    if abs(overlap) < 1e-10:
        raise RuntimeError(
            f"<ell|r> = {overlap:.2e}: leading mode appears degenerate"
        )
    ell_vec = ell_vec / np.conj(overlap)
    assert abs(np.vdot(ell_vec, r_vec) - 1.0) < 1e-7
    return theta_0, r_vec, ell_vec, gap, vals_sorted


# ============================================================
# theta_1 via four methods
# ============================================================
def theta1_perturbation(J_super, r_vec, ell_vec) -> complex:
    """Method 1 (rigorous): <ell|J|r>/<ell|r>, with biorthonormal denom = 1."""
    return complex(np.vdot(ell_vec, J_super @ r_vec))


def theta1_finite_diff(L0_super, J_super, theta_0_ref,
                       zetas=(1e-7, 1e-6, 1e-5, 1e-4)):
    """Method 2 (numerical cross-check): finite-difference (theta(zeta) - theta_0)/zeta.
    Tracks the leading eigenvalue by proximity to theta_0_ref."""
    out = []
    for zeta in zetas:
        vals = la.eigvals(L0_super + zeta * J_super)
        idx = np.argmin(np.abs(vals - theta_0_ref))
        theta_z = vals[idx]
        out.append((zeta, complex((theta_z - theta_0_ref) / zeta)))
    return out


def theta_series_coefficients(L0_super, J_super, theta_0_ref,
                              zetas=(1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)):
    """Extract theta_1 and theta_2 from a polynomial fit of theta(zeta) - theta_0.
    Returns (theta_1_fit, theta_2_fit, raw_data) where raw_data is the list of
    (zeta, theta(zeta) - theta_0) pairs.

    Useful when theta_1 = 0 by symmetry; the quadratic coefficient is the
    actual leading non-trivial term in the SCGF.
    """
    raw = []
    for zeta in zetas:
        vals = la.eigvals(L0_super + zeta * J_super)
        idx = np.argmin(np.abs(vals - theta_0_ref))
        theta_z = vals[idx]
        raw.append((zeta, complex(theta_z - theta_0_ref)))

    z = np.array([r[0] for r in raw], dtype=np.float64)
    f = np.array([r[1].real for r in raw], dtype=np.float64)
    # Fit f(zeta) = a*zeta + b*zeta^2  (no constant -- exact at zeta=0)
    # Solve least-squares for (a, b) given f = a*z + b*z^2
    A = np.column_stack([z, z**2])
    coef, *_ = np.linalg.lstsq(A, f, rcond=None)
    theta_1_fit, theta_2_fit = float(coef[0]), float(coef[1])
    return theta_1_fit, theta_2_fit, raw


def activity_in_r0(r_vec, jumps, D) -> tuple[complex, complex]:
    """Method 3 (BdG-style 'click rate in slow state'):
        <K>_eff = sum_j Tr(r_mat * J_j^dag J_j) / Tr(r_mat)
                = sum_j <L_0|J_j^dag J_j|R_0> / <L_0|R_0>.
    Distinct from theta_1; included for explicit comparison. Returns
    (activity, Tr(r_mat))."""
    r_mat = vec_to_mat(r_vec, D)
    trace_r = complex(np.trace(r_mat))
    JdJ = sum(J.conj().T @ J for J in jumps)
    raw = complex(np.trace(r_mat @ JdJ))
    if abs(trace_r) < 1e-12:
        return raw, trace_r
    return raw / trace_r, trace_r


# ============================================================
# Cross-check via direct H_eff diagonalization
# ============================================================
def slowest_decay_state(H_eff: np.ndarray):
    """Return (z_0, |R_0>, <L_0|) for H_eff's eigenvalue with largest Im(z)
    (smallest decay rate Gamma_0 = -2 Im z_0). Biorthonormal <L_0|R_0> = 1."""
    vals, Ldag, R = la.eig(H_eff, left=True, right=True)
    order = np.argsort(-np.imag(vals))
    z0 = vals[order[0]]
    R0 = R[:, order[0]]
    L0 = Ldag[:, order[0]]
    overlap = np.vdot(L0, R0)
    if abs(overlap) < 1e-10:
        raise RuntimeError(f"<L|R> = {overlap:.2e}")
    L0 = L0 / np.conj(overlap)
    return z0, R0, L0


def theta1_from_heff(jumps, R0, L0) -> complex:
    """Method 4: sum_j <L_0|J_j|R_0> <L_0|J_j^dag|R_0> from H_eff eigenvectors.
    Should match Method 1 exactly when r_0 is rank-1 (= |R_0><L_0|)."""
    t1 = 0.0 + 0.0j
    for J in jumps:
        m1 = np.vdot(L0, J @ R0)               # <L_0|J|R_0>
        m2 = np.vdot(L0, J.conj().T @ R0)      # <L_0|J^dag|R_0>
        t1 += m1 * m2
    return complex(t1)


# ============================================================
# Top-level sweep
# ============================================================
def main():
    Ls = [4, 5, 6]
    lams = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0]

    # --- JW verification (Diagnostic 0) ---
    print("=" * 78)
    print("Step 1: Verify Jordan-Wigner anticommutation relations")
    print("=" * 78)
    for L in Ls:
        t0 = time()
        c, cd = jw_fermions(L)
        verify_jw(c, cd, L)
        print(f"  L = {L}: OK  ({time()-t0:.2f}s)")
    print()

    # --- Sanity check: lambda = 1, w = 0 (pure measurement) ---
    print("=" * 78)
    print("Step 2: Sanity check at lambda = 1, w = 0  (pure measurement, no hopping)")
    print("  Slow state has all bond parities -> +1, so theta_1 is boundary-only.")
    print("=" * 78)
    for L in Ls:
        H, H_eff, jumps, K = build_model(L, lam=1.0)
        L0s = build_L0(H_eff)
        Js = build_J_super(jumps)
        theta_0, r, ell, gap, vals = solve_leading(L0s)
        theta_1 = theta1_perturbation(Js, r, ell)
        fd = theta1_finite_diff(L0s, Js, theta_0, zetas=(1e-5,))
        z0, R0, L0 = slowest_decay_state(H_eff)
        theta_1_heff = theta1_from_heff(jumps, R0, L0)
        print(f"  L = {L}:")
        print(f"    theta_0      = {theta_0.real:+.6f}{theta_0.imag:+.2g}j   "
              f"(expected: -Gamma_0; gap = {gap:.4f})")
        print(f"    theta_1 (pert)  = {theta_1.real:+.6e}{theta_1.imag:+.1e}j")
        print(f"    theta_1 (fd)    = {fd[0][1].real:+.6e}{fd[0][1].imag:+.1e}j")
        print(f"    theta_1 (Heff)  = {theta_1_heff.real:+.6e}{theta_1_heff.imag:+.1e}j")
    print()

    # --- Full sweep (Diagnostics 1-4) ---
    print("=" * 78)
    print("Step 3: Full sweep across (L, lambda)")
    print("=" * 78)
    header = (f"{'L':>3} {'lam':>5} {'Re th_0':>9} {'th_1 pert':>11} "
              f"{'th_1 fit':>11} {'th_2 fit':>11} {'th_1_BdG':>11} "
              f"{'rank-1':>8} {'gap':>7} {'t(s)':>5}")
    print(header)
    print("-" * len(header))

    results = []
    for L in Ls:
        for lam in lams:
            t0 = time()
            H, H_eff, jumps, K = build_model(L, lam)
            L0s = build_L0(H_eff)
            Js = build_J_super(jumps)
            theta_0, r, ell, gap, vals = solve_leading(L0s)
            theta_1 = theta1_perturbation(Js, r, ell)
            # Polynomial fit gives both theta_1 and theta_2
            t1_fit, t2_fit, raw_fd = theta_series_coefficients(L0s, Js, theta_0)

            try:
                bdg = theta1_at_bdg(L, w=1.0 - lam, alpha=lam, which="min")
                theta_1_bdg = bdg['theta_1']
            except Exception:
                theta_1_bdg = float('nan')

            D = 2 ** L
            r_mat = vec_to_mat(r, D)
            S = la.svdvals(r_mat)
            rank_ratio = S[1] / S[0] if S[0] > 0 else float('inf')

            activity_naive, trace_r = activity_in_r0(r, jumps, D)

            try:
                z0, R0, L0 = slowest_decay_state(H_eff)
                theta_1_heff = theta1_from_heff(jumps, R0, L0)
                Gamma_0 = -2.0 * float(np.imag(z0))
            except Exception:
                theta_1_heff = complex('nan')
                Gamma_0 = float('nan')

            elapsed = time() - t0
            row = dict(
                L=L, lam=lam,
                theta_0=complex(theta_0),
                theta_1=complex(theta_1),
                theta_1_fit=float(t1_fit),
                theta_2_fit=float(t2_fit),
                theta_1_heff=complex(theta_1_heff),
                theta_1_bdg=float(theta_1_bdg) if np.isfinite(theta_1_bdg) else float('nan'),
                activity_naive=complex(activity_naive),
                trace_r=complex(trace_r),
                rank_ratio=float(rank_ratio),
                gap=float(gap),
                Gamma_0=float(Gamma_0),
                elapsed=float(elapsed),
            )
            results.append(row)
            print(f"{L:>3} {lam:>5.3f} {theta_0.real:>+9.4f} "
                  f"{theta_1.real:>+11.3e} {t1_fit:>+11.3e} {t2_fit:>+11.3e} "
                  f"{theta_1_bdg:>+11.3e} {rank_ratio:>8.1e} {gap:>7.3f} {elapsed:>5.1f}")

    # --- Persist ---
    out = '/Users/catlover1337/Documents/ppsQJ_m2/analysis/theta1_exact_data.pkl'
    with open(out, 'wb') as f:
        pickle.dump({'results': results, 'Ls': Ls, 'lams': lams}, f)
    print(f"\nSaved: {out}")

    return results


# ============================================================
# Plotting and summary
# ============================================================
def plot_results(results, save_path):
    import collections
    by_lam = collections.defaultdict(list)
    for r in results:
        by_lam[r['lam']].append(r)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    lams_sorted = sorted(by_lam.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lams_sorted)))

    # Panel 1: theta_1 vs L (Liouville)
    ax = axes[0, 0]
    for lam, c in zip(lams_sorted, colors):
        sub = sorted(by_lam[lam], key=lambda r: r['L'])
        Ls_a = np.array([r['L'] for r in sub])
        t1_a = np.array([r['theta_1'].real for r in sub])
        ax.plot(Ls_a, np.abs(t1_a), 'o-', color=c, label=f'λ={lam}', ms=6)
    ax.set_xlabel('L'); ax.set_ylabel(r'$|\theta_1|$ (Liouville exact)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(r'$\theta_1$ vs $L$, Liouville-exact')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, which='both')

    # Panel 2: Liouville vs BdG, ratio
    ax = axes[0, 1]
    for lam, c in zip(lams_sorted, colors):
        sub = sorted(by_lam[lam], key=lambda r: r['L'])
        Ls_a = np.array([r['L'] for r in sub])
        t1_a = np.array([r['theta_1'].real for r in sub])
        bdg_a = np.array([r['theta_1_bdg'] for r in sub])
        mask = np.isfinite(bdg_a) & (np.abs(bdg_a) > 1e-12)
        if mask.any():
            ratio = np.where(mask, t1_a / bdg_a, np.nan)
            ax.plot(Ls_a, ratio, 'o-', color=c, label=f'λ={lam}', ms=6)
    ax.set_xlabel('L'); ax.set_ylabel(r'$\theta_1^{\rm Liou} / \theta_1^{\rm BdG}$')
    ax.axhline(1.0, color='red', ls='--', alpha=0.5)
    ax.set_title('Ratio Liouville / BdG')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # Panel 3: theta_1 vs lambda at L=6
    ax = axes[1, 0]
    L_max = max(r['L'] for r in results)
    sub_L = [r for r in results if r['L'] == L_max]
    lams_a = np.array([r['lam'] for r in sub_L])
    t1_a = np.array([r['theta_1'].real for r in sub_L])
    bdg_a = np.array([r['theta_1_bdg'] for r in sub_L])
    ax.plot(lams_a, np.abs(t1_a), 'o-', label='Liouville', ms=8)
    mask = np.isfinite(bdg_a)
    ax.plot(lams_a[mask], np.abs(bdg_a[mask]), 's--', label='BdG (compute_theta1.py)', ms=6, mfc='none')
    # Add reference lines
    ax.plot(lams_a, 0.18 * lams_a**2 * L_max, ':', color='gray',
            label=r'$0.18\,\lambda^2 L$ (BdG fit at small $\lambda$)')
    ax.plot(lams_a, 0.18 * lams_a**3 * L_max, ':', color='red',
            label=r'$0.18\,\lambda^3 L$ (potential Liou scaling?)')
    ax.set_xlabel(r'$\lambda$'); ax.set_ylabel(r'$|\theta_1|$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(f'Lambda-dependence at L={L_max}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    # Panel 4: rank-1 quality + gap
    ax = axes[1, 1]
    for lam, c in zip(lams_sorted, colors):
        sub = sorted(by_lam[lam], key=lambda r: r['L'])
        Ls_a = np.array([r['L'] for r in sub])
        rank_a = np.array([r['rank_ratio'] for r in sub])
        ax.plot(Ls_a, rank_a, 'o-', color=c, label=f'λ={lam}', ms=6)
    ax.set_xlabel('L'); ax.set_ylabel(r'$S_2 / S_1$ of $r_0$ reshaped')
    ax.set_yscale('log')
    ax.set_title('Rank-1 quality of leading $r_0$ (lower = cleaner factorisation)')
    ax.axhline(1e-10, color='gray', ls='--', alpha=0.5, label='1e-10 threshold')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    plt.suptitle(r'Exact Liouville-space $\theta_1$ vs. BdG biorthogonal-covariance method',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def summarize(results):
    """Tight summary highlighting (a) theta_1 = 0 by parity symmetry, and
    (b) the scaling of the actual leading non-trivial coefficient theta_2."""
    print("\n" + "=" * 78)
    print("SUMMARY 1: theta_1 (linear) vs theta_2 (quadratic) at every (L, lambda)")
    print("=" * 78)
    print(f"  {'L':>3} {'lam':>6} {'theta_1 pert':>15} {'theta_1 fit':>14} "
          f"{'theta_2 fit':>14} {'theta_1_BdG':>14}")
    for r in results:
        print(f"  {r['L']:>3} {r['lam']:>6.3f} "
              f"{r['theta_1'].real:>+15.3e} {r['theta_1_fit']:>+14.3e} "
              f"{r['theta_2_fit']:>+14.3e} {r['theta_1_bdg']:>+14.3e}")

    print("\n" + "=" * 78)
    print("SUMMARY 2: theta_1 should be ZERO by fermion-parity selection rule")
    print("=" * 78)
    max_abs_theta1 = max(abs(r['theta_1'].real) for r in results)
    print(f"  max |Re theta_1| over all (L, lambda) = {max_abs_theta1:.3e}")
    print(f"  (machine epsilon ~1e-16; values <1e-10 confirm theta_1 = 0 exactly)")

    print("\n" + "=" * 78)
    print("SUMMARY 3: scaling of theta_2 with L (at fixed lambda)")
    print("=" * 78)
    import collections
    by_lam = collections.defaultdict(list)
    for r in results:
        by_lam[r['lam']].append(r)
    print(f"  {'lambda':>7} {'L=4':>11} {'L=5':>11} {'L=6':>11} "
          f"{'fit theta_2 ~ L^p':>22}")
    for lam in sorted(by_lam.keys()):
        sub = sorted(by_lam[lam], key=lambda r: r['L'])
        Ls_a = np.array([r['L'] for r in sub], dtype=float)
        t2_a = np.array([r['theta_2_fit'] for r in sub])
        vals_str = "  ".join(f"{v:>+11.3e}" for v in t2_a)
        # Power-law fit
        if all(t2 > 0 for t2 in t2_a):
            slope, intercept = np.polyfit(np.log(Ls_a), np.log(t2_a), 1)
            fit_str = f"{np.exp(intercept):.3e} * L^{slope:.2f}"
        elif all(t2 < 0 for t2 in t2_a):
            slope, intercept = np.polyfit(np.log(Ls_a), np.log(-t2_a), 1)
            fit_str = f"-{np.exp(intercept):.3e} * L^{slope:.2f}"
        else:
            fit_str = "(mixed sign)"
        print(f"  {lam:>7.3f} {vals_str}  {fit_str:>22}")
    print()


if __name__ == "__main__":
    results = main()
    plot_results(results,
                 '/Users/catlover1337/Documents/ppsQJ_m2/analysis/theta1_exact.png')
    summarize(results)
