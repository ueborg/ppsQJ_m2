from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


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


def effective_generator(L: int, w: float, gamma_m: float) -> np.ndarray:
    h_eff = np.asarray(majorana_hamiltonian_generator(L, w), dtype=np.complex128)
    for bond in range(L - 1):
        a, b = bond_jump_pair(bond)
        h_eff[a, b] -= 0.5j * gamma_m
        h_eff[b, a] += 0.5j * gamma_m
    return h_eff


@dataclass(frozen=True)
class GaussianChainModel:
    L: int
    w: float
    gamma_m: float
    h_hamiltonian: np.ndarray
    h_effective: np.ndarray
    jump_pairs: tuple[tuple[int, int], ...]
    gamma0: np.ndarray
    orbitals0: np.ndarray


def build_gaussian_chain_model(L: int, w: float, gamma_m: float) -> GaussianChainModel:
    gamma0 = neel_covariance(L)
    return GaussianChainModel(
        L=L,
        w=w,
        gamma_m=gamma_m,
        h_hamiltonian=majorana_hamiltonian_generator(L, w),
        h_effective=effective_generator(L, w, gamma_m),
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
    gamma_m: float = 0.0,
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
        branch_norm = float(np.exp(log_abs_det_r - 0.5 * gamma_m * n_monitored * dt))
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
