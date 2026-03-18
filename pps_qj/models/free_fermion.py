from __future__ import annotations

import numpy as np

from pps_qj.types import ModelSpec


def canonical_vacuum_gamma(L: int) -> np.ndarray:
    """Covariance matrix for the fermionic vacuum |000...0>.

    Each 2x2 block is -J_2 = [[0, -1], [1, 0]], corresponding to
    an empty site (n_j = 0).
    """
    g = np.zeros((2 * L, 2 * L), dtype=np.float64)
    for j in range(L):
        a = 2 * j
        b = 2 * j + 1
        g[a, b] = -1.0
        g[b, a] = 1.0
    return g


def alternating_gamma(L: int) -> np.ndarray:
    """Covariance matrix for the alternating |up dn up dn ...> state.

    Under Jordan-Wigner, spin-up = occupied (n=1), spin-down = empty (n=0).
    The j-th 2x2 block is (2n_j - 1) J_2 (eq 212).
    """
    g = np.zeros((2 * L, 2 * L), dtype=np.float64)
    for j in range(L):
        a = 2 * j
        b = 2 * j + 1
        sign = (-1.0) ** j  # j=0 occupied(+1), j=1 empty(-1), ...
        g[a, b] = sign
        g[b, a] = -sign
    return g


def initial_V_from_gamma(Gamma: np.ndarray) -> np.ndarray:
    """Derive the 2L x L orbital matrix V from a pure-state Gamma.

    V consists of the -1 eigenvectors of iΓ, consistent with Γ = i(2VV† - I).
    """
    n = Gamma.shape[0]
    L = n // 2
    vals, vecs = np.linalg.eig(1j * Gamma)
    # Eigenvalues of iΓ are ±1 for pure states; select -1 eigenspace
    idx = np.argsort(vals.real)[:L]
    V = vecs[:, idx]
    # Orthonormalize
    V, _ = np.linalg.qr(V, mode="reduced")
    return V


def free_fermion_gaussian_model(
    L: int,
    w: float,
    gamma: float,
    initial_pattern: str = "alternating",
) -> ModelSpec:
    """Build a free-fermion Gaussian model with open boundary conditions.

    Implements Model II from Section 5.12.14 of the theory document.
    The single-particle effective matrix h_eff (derived from the many-body
    H_eff = H - (i/2)Σ L†L) has entries for each bond j (0-indexed, j=0,...,L-2):

        h_eff[2j,   2j+3] =  w - i*gamma/2
        h_eff[2j+3, 2j  ] = -(w - i*gamma/2)
        h_eff[2j+1, 2j+2] = -w
        h_eff[2j+2, 2j+1] =  w

    (These are the numerically confirmed values. The imaginary part is −iγ/2 because
    the orbital propagator exp(h_eff τ)V acts on the ket only; for non-Hermitian H_eff
    the bra accumulates conjugate phases, so the effective propagation uses h_eff* → sign flip.
    eq:heff_entries in sec5new.tex also has a factor-of-2 error — see Issue 4.)

    The Majorana pair for bond j is (a, b) = (2j, 2j+3) in 0-indexed
    convention, corresponding to d†_j d_j = 1/2(1 - i γ_a γ_b).
    """
    n = 2 * L
    A = np.zeros((n, n), dtype=np.complex128)

    jump_pairs: list[tuple[int, int]] = []

    # Open boundary conditions: L-1 bonds
    for j in range(L - 1):
        a = 2 * j              # first Majorana of site j
        b = 2 * (j + 1) + 1    # second Majorana of site j+1 = 2j + 3
        c = 2 * j + 1          # second Majorana of site j
        d = 2 * (j + 1)        # first Majorana of site j+1 = 2j + 2

        # Single-particle h_eff entries (sec5new.tex eq:heff_entries, corrected
        # for the factor-of-2 error in the LaTeX; see Issue 4 in the analysis).
        # The imaginary part carries a minus sign relative to the LaTeX because
        # the orbital propagator V(τ) = exp(h_eff τ)V acts on the ket, and for
        # non-Hermitian H_eff the bra and ket accumulate conjugate phases.
        # Numerically confirmed: h_{2j,2j+3} = w - iγ/2 is correct.
        # Cross-pairing 1 (jump pair): γ_{2j} ↔ γ_{2j+3}
        A[a, b] = w - 0.5j * gamma
        A[b, a] = -(w - 0.5j * gamma)
        # Cross-pairing 2: γ_{2j+1} ↔ γ_{2j+2}
        A[c, d] = -w
        A[d, c] = w

        jump_pairs.append((a, b))

    # Select initial covariance matrix
    if initial_pattern == "vacuum":
        g0 = canonical_vacuum_gamma(L)
    else:
        g0 = alternating_gamma(L)

    # Compute initial orbital matrix V (eq 214)
    V0 = initial_V_from_gamma(g0)

    return ModelSpec(
        L=L,
        H=None,
        jump_ops=[],
        gamma=gamma,
        w=w,
        boundary="open",
        model_id="free_fermion_model_ii",
        initial_gamma=g0,
        majorana_A_eff=A,
        majorana_jump_pairs=jump_pairs,
        initial_V=V0,
    )
