from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq


def bond_jump_pair(bond: int) -> tuple[int, int]:
    """Return the 0-indexed Majorana pair for bond `bond`."""
    return 2 * bond, 2 * bond + 3


def neel_covariance(L: int) -> np.ndarray:
    """Majorana covariance of the |up,down,up,...> product state."""
    gamma = np.zeros((2 * L, 2 * L), dtype=np.float64)
    for site in range(L):
        a = 2 * site
        b = a + 1
        sign = 1.0 if site % 2 == 0 else -1.0
        gamma[a, b] = sign
        gamma[b, a] = -sign
    return gamma


def covariance_from_orbitals(orbitals: np.ndarray) -> np.ndarray:
    n = orbitals.shape[0]
    gamma = 1j * (2.0 * (orbitals @ orbitals.conj().T) - np.eye(n))
    gamma = np.real_if_close(gamma, tol=1_000.0)
    gamma = np.asarray(gamma.real, dtype=np.float64)
    return 0.5 * (gamma - gamma.T)


def orbitals_from_covariance(covariance: np.ndarray) -> np.ndarray:
    Gamma = np.asarray(covariance, dtype=np.float64)
    n = Gamma.shape[0]
    L = n // 2
    values, vectors = np.linalg.eigh(1j * Gamma)
    order = np.argsort(values.real)
    orbitals = vectors[:, order[:L]]
    orbitals, _ = np.linalg.qr(orbitals, mode="reduced")
    return np.asarray(orbitals, dtype=np.complex128)


def project_to_physical_covariance(covariance: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    C = 0.5 * (np.asarray(covariance, dtype=np.float64) - np.asarray(covariance, dtype=np.float64).T)
    spectrum, vectors = np.linalg.eigh(1j * C)
    clipped = np.clip(spectrum.real, -1.0 + eps, 1.0 - eps)
    hermitian = vectors @ np.diag(clipped) @ vectors.conj().T
    projected = np.real_if_close(-1j * hermitian, tol=1_000.0).real
    return 0.5 * (projected - projected.T)


def majorana_hamiltonian_generator(L: int, w: float) -> np.ndarray:
    h = np.zeros((2 * L, 2 * L), dtype=np.float64)
    for bond in range(L - 1):
        a, b = bond_jump_pair(bond)
        c = 2 * bond + 1
        d = 2 * bond + 2
        h[a, b] = w
        h[b, a] = -w
        h[c, d] = -w
        h[d, c] = w
    return h


def effective_generator(L: int, w: float, alpha: float) -> np.ndarray:
    h_eff = np.asarray(majorana_hamiltonian_generator(L, w), dtype=np.complex128)
    for bond in range(L - 1):
        a, b = bond_jump_pair(bond)
        h_eff[a, b] -= 1j * alpha
        h_eff[b, a] += 1j * alpha
    return h_eff


@dataclass(frozen=True)
class GaussianChainModel:
    L: int
    w: float
    # alpha: measurement rate in Kells et al. (2023), Eq. (1); jump rate = 2*alpha
    alpha: float
    h_hamiltonian: np.ndarray
    h_effective: np.ndarray
    jump_pairs: tuple[tuple[int, int], ...]
    gamma0: np.ndarray
    orbitals0: np.ndarray


def build_gaussian_chain_model(L: int, w: float, alpha: float) -> GaussianChainModel:
    gamma0 = neel_covariance(L)
    return GaussianChainModel(
        L=L,
        w=w,
        alpha=alpha,
        h_hamiltonian=majorana_hamiltonian_generator(L, w),
        h_effective=effective_generator(L, w, alpha),
        jump_pairs=tuple(bond_jump_pair(bond) for bond in range(L - 1)),
        gamma0=gamma0,
        orbitals0=orbitals_from_covariance(gamma0),
    )


def jump_probability(covariance: np.ndarray, jump_pair: tuple[int, int]) -> float:
    a, b = jump_pair
    sigma = float(np.asarray(covariance, dtype=np.float64)[a, b])
    return float(np.clip(0.5 * (1.0 - sigma), 0.0, 1.0))


@dataclass(frozen=True)
class GaussianNoClickEvolution:
    orbitals_unnormalized: np.ndarray
    orbitals_normalized: np.ndarray
    covariance: np.ndarray
    branch_norm: float


def propagate_no_click_orbitals(
    orbitals: np.ndarray,
    h_effective: np.ndarray,
    dt: float,
    *,
    alpha: float = 0.0,
    n_monitored: int = 0,
) -> GaussianNoClickEvolution:
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if dt == 0.0:
        gamma = covariance_from_orbitals(orbitals)
        return GaussianNoClickEvolution(
            orbitals_unnormalized=np.asarray(orbitals, dtype=np.complex128).copy(),
            orbitals_normalized=np.asarray(orbitals, dtype=np.complex128).copy(),
            covariance=gamma,
            branch_norm=1.0,
        )

    M = expm(np.asarray(h_effective, dtype=np.complex128) * dt)
    orbitals_tilde = M @ np.asarray(orbitals, dtype=np.complex128)
    q_matrix, r_matrix = np.linalg.qr(orbitals_tilde, mode="reduced")
    diag_abs = np.abs(np.diag(r_matrix))
    if np.any(diag_abs <= 1e-300):
        branch_norm = 0.0
    else:
        log_abs_det_r = float(np.sum(np.log(diag_abs)))
        branch_norm = float(np.exp(log_abs_det_r - alpha * n_monitored * dt))
    gamma = covariance_from_orbitals(q_matrix)
    return GaussianNoClickEvolution(
        orbitals_unnormalized=orbitals_tilde,
        orbitals_normalized=q_matrix,
        covariance=gamma,
        branch_norm=branch_norm,
    )


def apply_projective_jump(
    covariance: np.ndarray,
    jump_pair: tuple[int, int],
) -> tuple[float, np.ndarray]:
    Gamma = np.asarray(covariance, dtype=np.float64)
    a, b = jump_pair
    sigma = float(Gamma[a, b])
    denom = 1.0 - sigma
    q = 0.5 * denom
    if denom <= 1e-14:
        raise ValueError("Jump probability is numerically zero")

    u = Gamma[:, a].copy()
    v = Gamma[:, b].copy()
    keep_mask = np.ones(Gamma.shape[0], dtype=bool)
    keep_mask[a] = False
    keep_mask[b] = False
    keep = np.flatnonzero(keep_mask)

    gamma_new = Gamma.copy()
    gamma_new[np.ix_(keep, keep)] += (
        np.outer(u[keep], v[keep]) - np.outer(v[keep], u[keep])
    ) / denom
    gamma_new[a, :] = 0.0
    gamma_new[:, a] = 0.0
    gamma_new[b, :] = 0.0
    gamma_new[:, b] = 0.0
    gamma_new[a, b] = -1.0
    gamma_new[b, a] = 1.0
    gamma_new = 0.5 * (gamma_new - gamma_new.T)
    return float(np.clip(q, 0.0, 1.0)), gamma_new


def entanglement_entropy(covariance: np.ndarray, ell: int, base: float = 2.0) -> float:
    Gamma = np.asarray(covariance, dtype=np.float64)
    if ell < 0 or 2 * ell > Gamma.shape[0]:
        raise ValueError("ell must define a valid subsystem")
    if ell == 0:
        return 0.0
    sub = Gamma[: 2 * ell, : 2 * ell]
    eigenvalues = np.linalg.eigvals(sub)
    nus = np.sort(np.abs(np.imag(eigenvalues)))[::2]
    entropy = 0.0
    log_fn = np.log2 if base == 2.0 else lambda x: np.log(x) / np.log(base)
    for nu in nus:
        nu = float(np.clip(nu, 0.0, 1.0))
        p_plus = 0.5 * (1.0 + nu)
        p_minus = 0.5 * (1.0 - nu)
        term = 0.0
        if p_plus > 1e-15:
            term -= p_plus * log_fn(p_plus)
        if p_minus > 1e-15:
            term -= p_minus * log_fn(p_minus)
        entropy += term
    return float(entropy)


@dataclass(frozen=True)
class GaussianTrajectoryResult:
    """Result of a Born-rule quantum-jump trajectory in the Gaussian backend."""
    final_covariance: np.ndarray
    n_jumps: int
    jump_times: list[float]
    jump_channels: list[int]


def gaussian_born_rule_trajectory(
    model: GaussianChainModel,
    T: float,
    rng: np.random.Generator,
    bisection_tol: float = 1e-8,
) -> GaussianTrajectoryResult:
    """Exact Born-rule quantum-jump trajectory.

    Waiting times are sampled by inverting the survival probability
    ``branch_norm(dt) = U`` using Brent's method (brentq), which converges
    in ~5–8 function evaluations vs ~40 for pure bisection.

    Per-jump no-click propagation uses the cached eigendecomposition of
    ``h_effective`` (two matrix multiplies) rather than scipy's Padé expm,
    and reuses ``coeffs = V⁻¹ @ orbitals`` already computed for the branch-norm
    evaluation, saving one extra matrix multiply per jump.
    """
    n_monitored = len(model.jump_pairs)

    # Pre-diagonalise h_effective once for the whole trajectory.
    h_eff = np.asarray(model.h_effective, dtype=np.complex128)
    evals, V = np.linalg.eig(h_eff)
    V_inv = np.linalg.inv(V)
    # Gram matrix V†V — needed because V is not unitary (h_eff non-Hermitian).
    VhV = V.conj().T @ V

    # Precompute jump-pair index arrays for vectorised probability extraction.
    # Replaces an L-1 Python function call loop with a single numpy indexing op.
    _jp = model.jump_pairs
    _ja = np.array([p[0] for p in _jp], dtype=np.intp)  # (L-1,) row indices
    _jb = np.array([p[1] for p in _jp], dtype=np.intp)  # (L-1,) col indices

    orbitals = np.asarray(model.orbitals0, dtype=np.complex128).copy()
    cov = np.asarray(model.gamma0, dtype=np.float64).copy()
    t = 0.0
    jump_times: list[float] = []
    jump_channels: list[int] = []

    while t < T:
        U = float(rng.uniform(0.0, 1.0))
        T_rem = T - t

        # V⁻¹ @ orbitals — reused both in branch-norm and in propagation.
        coeffs = V_inv @ orbitals  # (2L, L)

        def _fast_branch_norm(dt: float) -> float:
            """log|det(M(dt) V₀)| via L×L Gram-matrix slogdet."""
            if dt <= 0.0:
                return 1.0
            exp_d = np.exp(evals * dt)
            A = exp_d[:, None] * coeffs
            gram = A.conj().T @ (VhV @ A)
            sign, logdet = np.linalg.slogdet(gram)
            if sign <= 0:
                return 0.0
            return float(np.exp(0.5 * logdet - model.alpha * n_monitored * dt))

        bn_end = _fast_branch_norm(T_rem)
        if bn_end >= U:
            # No jump in remaining time — propagate to T using cached eig.
            exp_d = np.exp(evals * T_rem)
            orbs_tilde = V @ (exp_d[:, None] * coeffs)
            q_mat, _ = np.linalg.qr(orbs_tilde, mode="reduced")
            cov = covariance_from_orbitals(q_mat)
            break

        # Find jump time via Brent's method: ~5–8 evals vs ~40 for bisection.
        try:
            dt_star = brentq(
                lambda dt: _fast_branch_norm(dt) - U,
                0.0, T_rem,
                xtol=bisection_tol,
                maxiter=50,
                full_output=False,
            )
        except ValueError:
            # Bracketing failure (numerical edge case) — fall back to midpoint.
            dt_star = 0.5 * T_rem

        # Propagate to jump time using cached eigendecomposition.
        # Reuses coeffs already computed above — no extra V_inv multiply.
        exp_d_star = np.exp(evals * dt_star)
        orbs_tilde = V @ (exp_d_star[:, None] * coeffs)
        q_mat, r_mat = np.linalg.qr(orbs_tilde, mode="reduced")
        orbitals = q_mat
        cov = covariance_from_orbitals(orbitals)
        t += dt_star

        # Select jump channel proportional to q_j (vectorised over all bonds).
        probs = np.clip(0.5 * (1.0 - cov[_ja, _jb]), 0.0, 1.0)
        total = probs.sum()
        if total < 1e-15:
            break
        probs /= total
        channel = int(rng.choice(n_monitored, p=probs))

        # Apply jump and recover orthonormal orbitals.
        _, cov = apply_projective_jump(cov, model.jump_pairs[channel])
        orbitals = orbitals_from_covariance(cov)

        jump_times.append(t)
        jump_channels.append(channel)

    return GaussianTrajectoryResult(
        final_covariance=cov,
        n_jumps=len(jump_times),
        jump_times=jump_times,
        jump_channels=jump_channels,
    )


def topological_entanglement_entropy(covariance: np.ndarray) -> float:
    """Topological entanglement entropy from the ABDC partition.

    Accepts a 2L x 2L real antisymmetric Majorana covariance matrix and returns
    S_top = S_AB + S_BC - S_B - S_D, where the ABDC partition divides L sites
    into four equal quarters: A=[0,L/4), B=[L/4,L/2), D=[3L/4,L), C=[L/2,3L/4).

    In the large-alpha, small-w limit (topological phase) the steady-state
    average approaches 1; in the small-alpha, large-w limit (critical phase)
    it remains O(1) but the product B_L = S_top * S_L diverges with L.

    Parameters
    ----------
    covariance : (2L, 2L) ndarray
        Majorana covariance matrix.

    Returns
    -------
    float
        The topological entanglement entropy.

    Raises
    ------
    ValueError
        If L is not divisible by 4.
    """
    n = covariance.shape[0]
    if n % 8 != 0:
        raise ValueError(
            f"topological_entanglement_entropy requires L divisible by 4, "
            f"got covariance shape {n}x{n} implying L={n//2}"
        )
    L = n // 2
    q = L // 4
    Gamma = np.asarray(covariance, dtype=np.float64)

    # Convert Majorana covariance -> (G, F) correlation matrices via unitary
    # c_j = (gamma_{2j} + i*gamma_{2j+1})/2, so <c^dag_j c_k> and <c_j c_k>
    # are obtained from the Majorana two-point function <gamma_a gamma_b> = delta_{ab} + i*Gamma_{ab}.
    U_mat = np.zeros((L, n), dtype=np.complex128)
    for j in range(L):
        U_mat[j, 2 * j] = 0.5
        U_mat[j, 2 * j + 1] = 0.5j
    M = np.eye(n) + 1j * Gamma
    G = U_mat @ M @ U_mat.conj().T   # G_jk = <c_j^dag c_k>
    F = U_mat @ M @ U_mat.T          # F_jk = <c_j c_k>

    def _region_entropy(G, F, idx):
        """Von Neumann entropy of a region from the (G, F) correlation matrix (log2)."""
        idx = np.asarray(idx)
        G_X = G[np.ix_(idx, idx)]
        F_X = F[np.ix_(idx, idx)]
        n_x = len(idx)
        C_X = np.block([
            [G_X, F_X],
            [-F_X.conj(), np.eye(n_x) - G_X.T]
        ])
        eigvals = np.linalg.eigvalsh(C_X)
        eps = 1e-14
        eigvals = np.clip(eigvals.real, eps, 1.0 - eps)
        # BdG eigenvalues come in (lambda, 1-lambda) pairs; divide by 2 to avoid double-counting.
        return 0.5 * float(-np.sum(eigvals * np.log2(eigvals) + (1.0 - eigvals) * np.log2(1.0 - eigvals)))

    # ABDC partition: A=[0,q), B=[q,2q), C=[2q,3q), D=[3q,4q)
    A = np.arange(0, q)
    B = np.arange(q, 2 * q)
    C = np.arange(2 * q, 3 * q)
    D = np.arange(3 * q, 4 * q)

    AB = np.concatenate([A, B])
    BC = np.concatenate([B, C])

    S_AB = _region_entropy(G, F, AB)
    S_BC = _region_entropy(G, F, BC)
    S_B = _region_entropy(G, F, B)
    S_D = _region_entropy(G, F, D)

    return S_AB + S_BC - S_B - S_D
