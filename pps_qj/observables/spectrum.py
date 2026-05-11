"""Renyi entropies and single-particle correlation functions from Majorana covariance.

These observables are derived from the *same* eigendecomposition of the
restricted Majorana covariance that the von Neumann entropy uses, so
computing S_n for additional n adds negligible cost once one has done the
eigvalsh call.

Conformal-invariance tests:
  In a free Dirac CFT with central charge c, the Renyi entropies of a
  half-interval scale as
      S_n(L) ~ c_n * ln L,   c_n = (c/6) * (1 + 1/n)
  so the *ratio* c_n / c_1 should equal (1 + 1/n) / 2 independent of c:
      c_2 / c_1 = 3/4
      c_3 / c_1 = 4/6 = 2/3
      c_4 / c_1 = 5/8
  Independent verification of any pair of Renyi indices testing this ratio
  pins down the conformal structure.

Single-particle correlation function:
  Mapping Majorana operators to Dirac:
      chi_{2j-1} = c_j + c_j_dag,    chi_{2j} = -i (c_j - c_j_dag)
  Given Gamma_{a,b} = (i/2) <[chi_a, chi_b]> (the convention used by this code),
  the single-particle correlation matrix C_{ij} = <c_j_dag c_i> is derived
  by inverting the Dirac-Majorana mapping and using <c_j c_i>=0
  (the pairing terms vanish for particle-number-conserving states).
  For our model (which has hopping but no pairing) this is exact.

  In a free Dirac CFT at half-filling, the off-diagonal correlator
      C(r) = <c_0_dag c_r>
  decays as r^{-1} at long distances. The decay exponent provides a
  CFT-independent test of conformality.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "renyi_entropies_from_covariance",
    "renyi_entropies_batched",
    "single_particle_correlation",
    "translation_averaged_correlation_decay",
]


def _renyi_from_nu(nu: np.ndarray, n: int) -> float:
    """Renyi entropy S_n in nats from the symplectic eigenvalues nu in [0, 1].

    S_n = (1/(1-n)) sum log[ ((1+nu)/2)^n + ((1-nu)/2)^n ]

    Limit n -> 1 recovers the von Neumann entropy by L'Hopital.
    """
    nu = np.clip(np.asarray(nu, dtype=np.float64), 0.0, 1.0)
    p = 0.5 * (1.0 + nu)
    q = 1.0 - p
    if n == 1:
        s_per = np.zeros_like(p)
        mask = (p > 1e-15) & (q > 1e-15)
        s_per[mask] = -(p[mask] * np.log(p[mask]) + q[mask] * np.log(q[mask]))
        return float(np.sum(s_per))
    p_n = np.where(p > 1e-15, p**n, 0.0)
    q_n = np.where(q > 1e-15, q**n, 0.0)
    arg = p_n + q_n
    arg = np.where(arg > 1e-300, arg, 1e-300)
    return float(np.sum(np.log(arg)) / (1.0 - n))


def renyi_entropies_from_covariance(
    Gamma: np.ndarray,
    ell: int,
    ns: tuple = (1, 2, 3),
    base: float = 2.0,
) -> dict:
    """Compute Renyi entropies S_n for n in ns from a Majorana covariance.

    Parameters
    ----------
    Gamma : (2L, 2L) real antisymmetric Majorana covariance matrix.
    ell   : subsystem size in sites (sites 1..ell).
    ns    : tuple of Renyi indices to compute. n=1 is von Neumann.
    base  : entropy base. 2 for bits, e for nats.

    Returns
    -------
    dict {n: S_n} with all entropies on the same logarithmic base.
    """
    if ell == 0:
        return {n: 0.0 for n in ns}
    Gamma = np.asarray(Gamma, dtype=np.float64)
    n_majorana = 2 * ell
    sub = Gamma[:n_majorana, :n_majorana]
    eigs = np.linalg.eigvalsh(1j * sub.astype(np.complex128)).real
    k = eigs.size // 2
    nu = np.abs(eigs[k:])
    out = {}
    log_to_base = (1.0 / np.log(base)) if base != np.e else 1.0
    for n in ns:
        S_nats = _renyi_from_nu(nu, int(n))
        out[int(n)] = S_nats * log_to_base
    return out


def renyi_entropies_batched(
    covs,
    ell: int,
    ns: tuple = (1, 2, 3),
    base: float = 2.0,
) -> np.ndarray:
    """Vectorised Renyi entropies for a population of clones.

    Parameters
    ----------
    covs : list or ndarray of length N_c with (2L, 2L) Majorana covariances.
    ell  : subsystem size in sites.

    Returns
    -------
    ndarray of shape (N_c, len(ns)), columns ordered as ns.
    """
    if ell == 0:
        N = len(covs)
        return np.zeros((N, len(ns)), dtype=np.float64)
    two_ell = 2 * ell
    log_to_base = (1.0 / np.log(base)) if base != np.e else 1.0
    try:
        subs = np.stack([c[:two_ell, :two_ell] for c in covs], axis=0)
        eigs = np.linalg.eigvalsh((1j * subs).astype(np.complex128))
        k = two_ell // 2
        nu = np.clip(np.abs(eigs[:, k:]), 0.0, 1.0)
        p = 0.5 * (1.0 + nu)
        q = 1.0 - p
        N = nu.shape[0]
        out = np.zeros((N, len(ns)), dtype=np.float64)
        for col, n in enumerate(ns):
            if int(n) == 1:
                mask = (p > 1e-15) & (q > 1e-15)
                s_per = np.zeros_like(p)
                s_per[mask] = -(p[mask] * np.log(p[mask]) + q[mask] * np.log(q[mask]))
                out[:, col] = s_per.sum(axis=1) * log_to_base
            else:
                p_n = np.where(p > 1e-15, p ** int(n), 0.0)
                q_n = np.where(q > 1e-15, q ** int(n), 0.0)
                arg = p_n + q_n
                arg = np.where(arg > 1e-300, arg, 1e-300)
                out[:, col] = (np.log(arg).sum(axis=1) / (1.0 - int(n))) * log_to_base
        return out
    except np.linalg.LinAlgError:
        N = len(covs)
        out = np.zeros((N, len(ns)), dtype=np.float64)
        for i, c in enumerate(covs):
            res = renyi_entropies_from_covariance(c, ell, ns=ns, base=base)
            for col, n in enumerate(ns):
                out[i, col] = res[int(n)]
        return out


def single_particle_correlation(Gamma: np.ndarray) -> np.ndarray:
    """Single-particle correlation matrix C_{ij} = <c_i_dag c_j> from Majorana covariance.

    Assumes the state is particle-number-preserving (no pairing), which holds
    for our model (Kitaev-like hopping with density measurements).

    Convention (verified against build_gaussian_chain_model):
        chi_{2j}   = c_j + c_j_dag        (Majorana "real" part of site j, zero-based)
        chi_{2j+1} = -i (c_j - c_j_dag)   ("imag" part)
        Gamma_{ab} = (i/2) <[chi_a, chi_b]>

    Derivation gives, for the particle-conserving (no pairing) case:
        Gamma_{2i,   2j+1} = 2 Re(C_{ij}) - delta_{ij}
        Gamma_{2i,   2j}   = -2 Im(C_{ij})         (= 0 on diagonal: <n_i> is real)
    so the full correlation matrix is
        C_{ij} = (Gamma_{2i,2j+1} + delta_{ij}) / 2  -  i * Gamma_{2i,2j} / 2

    Parameters
    ----------
    Gamma : (2L, 2L) real antisymmetric Majorana covariance.

    Returns
    -------
    C : (L, L) complex Hermitian matrix with C_{ii} = <n_i> on the diagonal.
    """
    Gamma = np.asarray(Gamma, dtype=np.float64)
    n2 = Gamma.shape[0]
    if n2 % 2:
        raise ValueError("Gamma must have even dimension.")
    L = n2 // 2

    # Block-reshape Gamma into 2x2 blocks: G[i, j, a, b] = Gamma_{2i+a, 2j+b}
    G = Gamma.reshape(L, 2, L, 2).transpose(0, 2, 1, 3)

    # Re(C_{ij}) = (Gamma_{2i, 2j+1} + delta_{ij}) / 2
    # Im(C_{ij}) = -Gamma_{2i, 2j} / 2
    re_part = 0.5 * G[:, :, 0, 1]
    np.fill_diagonal(re_part, 0.5 * (np.diag(G[:, :, 0, 1]) + 1.0))
    im_part = -0.5 * G[:, :, 0, 0]

    C = re_part + 1j * im_part
    # Enforce Hermiticity (small numerical asymmetry possible)
    C = 0.5 * (C + C.conj().T)
    return C


def translation_averaged_correlation_decay(C: np.ndarray, r_max=None):
    """Translation-average the correlation function for a clean C(r) curve.

    For a translationally-invariant infinite chain, C_{ij} depends only on
    r = |i - j|. For our finite OBC chain, we compute
        C(r) = (1/(L-r)) sum_{i:1..L-r} |C_{i, i+r}|
    excluding self-correlation r = 0.

    Returns (r_array, C_r_array) where C_r_array[k] is the mean |C| at separation
    r_array[k] = k+1 (i.e. r_array starts at 1).
    """
    C = np.asarray(C)
    L = C.shape[0]
    if r_max is None:
        r_max = L - 1
    r_max = min(int(r_max), L - 1)
    rs = np.arange(1, r_max + 1)
    out = np.zeros(r_max)
    for ridx, r in enumerate(rs):
        diag = np.array([C[i, i + r] for i in range(L - r)])
        out[ridx] = float(np.mean(np.abs(diag)))
    return rs, out
