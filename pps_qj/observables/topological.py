"""Topological and dual-topological entanglement entropy from Majorana covariances.

Implements the four-region Kitaev-Preskill-style combination used in
Kells, Meidan & Romito (SciPost 2023) as an MIPT diagnostic on the
measurement-only free-fermion chain.

This is a standalone implementation; it deliberately does NOT call
``pps_qj.gaussian_backend.topological_entanglement_entropy`` (which uses a
different ABDC partition S_AB + S_BC - S_B - S_D).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "subsystem_entropy",
    "topological_entropy",
    "dual_topological_entropy",
    "compute_all_observables",
]


def _binary_entropy_from_nu(nu: np.ndarray) -> float:
    nu = np.clip(np.asarray(nu, dtype=np.float64), 0.0, 1.0)
    p = 0.5 * (1.0 + nu)
    s = np.zeros_like(p)
    mask = (p > 1e-12) & (p < 1.0 - 1e-12)
    s[mask] = -(p[mask] * np.log2(p[mask]) + (1.0 - p[mask]) * np.log2(1.0 - p[mask]))
    return float(np.sum(s))


def _entropy_from_majorana_subblock(gamma_sub: np.ndarray) -> float:
    # gamma_sub is real antisymmetric; 1j * gamma_sub is Hermitian with
    # eigenvalues ±ν_m (ν_m >= 0). eigvalsh returns them sorted ascending.
    eigs = np.linalg.eigvalsh(1j * np.asarray(gamma_sub, dtype=np.complex128))
    eigs = np.real(eigs)
    k = eigs.size // 2
    # Take the non-negative half (last k entries after ascending sort).
    nu = np.abs(eigs[k:])
    return _binary_entropy_from_nu(nu)


def subsystem_entropy(gamma: np.ndarray, site_indices: list[int]) -> float:
    """Von Neumann entropy (bits) of a subsystem from the Majorana covariance.

    Parameters
    ----------
    gamma
        ``(2L, 2L)`` real antisymmetric Majorana covariance matrix.
        Convention: ``gamma[m, n] = (i/2) <[eta_m, eta_n]>`` with zero-based
        Majorana indices (``eta_1 -> 0``, ``eta_2 -> 1``, ...).
    site_indices
        1-indexed site labels. Site ``j`` contributes Majorana indices
        ``2*(j-1)`` and ``2*(j-1)+1`` (zero-based).
    """
    if len(site_indices) == 0:
        return 0.0
    maj_idx: list[int] = []
    for s in site_indices:
        base = 2 * (int(s) - 1)
        maj_idx.append(base)
        maj_idx.append(base + 1)
    gamma = np.asarray(gamma, dtype=np.float64)
    sub = gamma[np.ix_(maj_idx, maj_idx)]
    return _entropy_from_majorana_subblock(sub)


def topological_entropy(gamma: np.ndarray, L: int) -> float:
    """Kitaev-Preskill combination ``S_AB + S_BD - S_B - S_ABD``.

    Partition of the L-site chain into four contiguous quarters (1-indexed):
    ``A=[1, L/4]``, ``B=[L/4+1, L/2]``, ``D=[L/2+1, 3L/4]``, ``C=[3L/4+1, L]``.
    Returns ``np.nan`` if ``L % 4 != 0``.
    """
    if L % 4 != 0:
        return float("nan")
    q = L // 4
    A = list(range(1, q + 1))
    B = list(range(q + 1, 2 * q + 1))
    D = list(range(2 * q + 1, 3 * q + 1))
    S_AB = subsystem_entropy(gamma, A + B)
    S_BD = subsystem_entropy(gamma, B + D)
    S_B = subsystem_entropy(gamma, B)
    S_ABD = subsystem_entropy(gamma, A + B + D)
    return S_AB + S_BD - S_B - S_ABD


def dual_topological_entropy(
    gamma: np.ndarray,
    jump_pairs,
) -> float:
    """Dual-basis topological entropy evaluated on the d-mode chain.

    The d-mode for bond ``j`` has Majorana operators ``eta_{a_j}, eta_{b_j}``
    where ``(a_j, b_j) = jump_pairs[j]`` are zero-based Majorana indices on the
    original chain. We extract the ``(2 L_d) x (2 L_d)`` subblock indexed by
    ``{a_0, b_0, a_1, b_1, ...}`` and apply the same four-region combination,
    treating d-mode ``j`` as having Majorana pair ``(2j, 2j+1)`` in the subblock.

    Returns ``np.nan`` if ``L_d = len(jump_pairs)`` is not divisible by 4.
    """
    pairs = list(jump_pairs)
    L_d = len(pairs)
    if L_d % 4 != 0:
        return float("nan")
    maj_idx: list[int] = []
    for (a, b) in pairs:
        maj_idx.append(int(a))
        maj_idx.append(int(b))
    gamma = np.asarray(gamma, dtype=np.float64)
    gamma_d = gamma[np.ix_(maj_idx, maj_idx)]

    q = L_d // 4

    def _d_entropy(d_sites: list[int]) -> float:
        if len(d_sites) == 0:
            return 0.0
        idx: list[int] = []
        for s in d_sites:
            idx.append(2 * s)
            idx.append(2 * s + 1)
        sub = gamma_d[np.ix_(idx, idx)]
        return _entropy_from_majorana_subblock(sub)

    AB = list(range(0, 2 * q))
    BD = list(range(q, 3 * q))
    B = list(range(q, 2 * q))
    ABD = list(range(0, 3 * q))
    return _d_entropy(AB) + _d_entropy(BD) - _d_entropy(B) - _d_entropy(ABD)


def compute_all_observables(
    gamma: np.ndarray,
    L: int,
    jump_pairs,
) -> dict:
    """Compute all entanglement observables from a single covariance matrix.

    Returns a dict with keys ``S_half``, ``S_top``, ``S_top_d``, ``B_L``,
    ``B_L_prime``. NaN propagates when a quantity is undefined at this L.
    """
    if L % 4 != 0:
        raise ValueError(
            f"compute_all_observables requires L divisible by 4 (got L={L}); "
            "the four-region Kitaev-Preskill partition is otherwise ill-defined."
        )
    half = list(range(1, L // 2 + 1))  # sites 1..L/2 (1-indexed)
    S_half = subsystem_entropy(gamma, half)
    S_top = topological_entropy(gamma, L)
    S_top_d = dual_topological_entropy(gamma, jump_pairs)
    B_L = float("nan") if np.isnan(S_top) else S_top * S_half
    B_L_prime = float("nan") if np.isnan(S_top_d) else S_top_d * S_half
    return {
        "S_half": float(S_half),
        "S_top": float(S_top),
        "S_top_d": float(S_top_d),
        "B_L": float(B_L),
        "B_L_prime": float(B_L_prime),
    }
