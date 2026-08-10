"""Parity-sector-reduced exact backward pass for the monitored free-fermion chain.

The full exact backward pass (``run_exact_backward_pass``) materialises the
adjoint Liouvillian on operator space of dimension :math:`d_{\\text{full}}^2`
where :math:`d_{\\text{full}} = 2^L`. For :math:`L \\geq 10` the dense
superoperator is no longer storable.

This chain does **not** conserve total particle number — the bond jump
operators :math:`d_j = \\tfrac12(c_l^\\dagger + c_l + c_r - c_r^\\dagger)` are
Majorana-like and change :math:`N` by :math:`\\pm 1`. However the *fermion
parity* :math:`(-1)^{N_{\\text{tot}}}` **is** conserved: the Hamiltonian is
bilinear in :math:`c, c^\\dagger` (preserves :math:`N`, hence parity), and the
jump projectors :math:`P_j = d_j^\\dagger d_j` change :math:`N` by even amounts
(preserve parity). So the entire tilted generator
:math:`\\mathcal{L}_\\zeta^\\dagger` preserves the parity sector containing
the Néel initial state. Restricting to that sector halves :math:`d` (sector
dim :math:`2^{L-1}`) and quarters :math:`d^2` — a factor-4 storage win, not
the factor-:math:`(2^L / \\binom{L}{L/2})^2` particle-number reduction one
might naively expect.

This module provides:

- :func:`neel_parity_sector_indices`: indices of basis states with the same
  fermion parity as the Néel state.
- :func:`project_to_sector`: restrict a sparse :math:`d_{\\text{full}} \\times
  d_{\\text{full}}` operator to the sector basis.
- :func:`build_tilted_adjoint_action`: ``LinearOperator`` implementing
  :math:`O \\mapsto i[H,O] + 2\\alpha\\sum_j (\\zeta P_j O P_j - \\tfrac12
  \\{P_j, O\\})` on the sector, never forming the
  :math:`d^2 \\times d^2` matrix.
- :class:`SectorReducedBackwardData`: drop-in replacement for
  ``ExactBackwardData`` exposing ``operator_at(t)`` (returns the embedded
  :math:`d_{\\text{full}} \\times d_{\\text{full}}` operator) and
  ``overlap(t, state)`` (computed in sector coordinates — never embeds).
- :func:`run_exact_backward_pass_sector`: the entry point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, expm_multiply

from pps_qj.exact_backend import ExactSpinChainModel


def neel_parity_sector_indices(L: int) -> np.ndarray:
    """Return computational-basis indices in the same fermion-parity sector as
    the Néel state.

    The local basis convention (see :func:`pps_qj.exact_backend.neel_state`)
    is :math:`|0\\rangle = (1, 0)^T = \\text{occupied}`, so :math:`N_{\\text{tot}}`
    on basis index :math:`i` equals :math:`L - \\text{popcount}(i)` and the
    fermion parity is :math:`(-1)^{L - \\text{popcount}(i)}`. The Néel state
    :math:`|10\\,10\\cdots\\rangle` lives at index ``0b0101...`` with
    :math:`\\text{popcount} = L/2` (for even :math:`L`), so its parity sector
    contains all indices :math:`i` with
    :math:`\\text{popcount}(i) \\equiv L/2 \\pmod 2`.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if L % 2 != 0:
        raise ValueError("Néel half-filling requires even L")
    target_parity = (L // 2) % 2
    indices = [i for i in range(1 << L) if bin(i).count("1") % 2 == target_parity]
    return np.asarray(indices, dtype=np.intp)


# Backwards-compatible alias for the old name (initial design assumed the
# wrong conserved quantity; keeping the new name as canonical).
half_filling_sector_indices = neel_parity_sector_indices


def project_to_sector(operator: sp.spmatrix, sector_indices: np.ndarray) -> sp.csr_matrix:
    """Restrict a sparse operator to the sector spanned by ``sector_indices``.

    Equivalent to ``operator[np.ix_(sector_indices, sector_indices)]`` but works
    for sparse inputs without forming the dense intermediate.
    """
    csc = operator.tocsc()[:, sector_indices]
    csr = csc.tocsr()[sector_indices, :]
    return csr.tocsr()


def build_tilted_adjoint_action(
    h_sec: sp.csr_matrix,
    projectors_sec: tuple[sp.csr_matrix, ...],
    alpha: float,
    zeta: float,
) -> LinearOperator:
    """Action-only adjoint Liouvillian on operator space of the sector.

    Implements :math:`O \\mapsto i[H,O] + 2\\alpha\\sum_j(\\zeta P_j O P_j -
    \\tfrac12 \\{P_j, O\\})` as a :class:`scipy.sparse.linalg.LinearOperator`
    on :math:`\\mathbb{C}^{d^2}`, where vectorisation is column-major (Fortran)
    so the convention matches :func:`pps_qj.exact_backend.lindbladian_superoperator`.
    """
    d = h_sec.shape[0]
    h_sec = h_sec.tocsr()
    projectors = tuple(P.tocsr() for P in projectors_sec)
    coeff = 2.0 * alpha

    def _apply(vec: np.ndarray, sign_commutator: float) -> np.ndarray:
        O = np.asarray(vec, dtype=np.complex128).reshape((d, d), order="F")
        out = (1j * sign_commutator) * (h_sec @ O - O @ h_sec)
        for P in projectors:
            PO = P @ O
            OP = O @ P
            if zeta != 0.0:
                out = out + coeff * zeta * (P @ OP)
            out = out - 0.5 * coeff * (PO + OP)
        return out.reshape(d * d, order="F")

    def _matvec(vec: np.ndarray) -> np.ndarray:
        return _apply(vec, +1.0)

    def _rmatvec(vec: np.ndarray) -> np.ndarray:
        # Hilbert-Schmidt adjoint: H, P_j Hermitian → only commutator flips sign.
        return _apply(vec, -1.0)

    return LinearOperator(
        shape=(d * d, d * d),
        matvec=_matvec,
        rmatvec=_rmatvec,
        dtype=np.complex128,
    )


@dataclass
class SectorReducedBackwardData:
    """Sector-reduced backward-pass result with the same interface as
    :class:`pps_qj.backward_pass.ExactBackwardData`.

    Stores :math:`G_t` only on the half-filling sector (a :math:`d \\times d`
    matrix where :math:`d = \\binom{L}{L/2}`) at a grid of times. Linear
    interpolation is used between samples; :meth:`operator_at` embeds back to
    the full :math:`2^L \\times 2^L` Hilbert space (transient memory only),
    :meth:`overlap` works entirely in the sector.
    """

    model: ExactSpinChainModel
    T: float
    zeta: float
    sector_indices: np.ndarray
    sample_times: np.ndarray  # length n_samples
    sample_operators: np.ndarray  # (n_samples, d, d) complex128
    _full_dim: int = 0

    def __post_init__(self) -> None:
        self._full_dim = self.model.dim

    @property
    def sector_dim(self) -> int:
        return int(self.sector_indices.shape[0])

    def _operator_sector_at(self, t: float) -> np.ndarray:
        """Linearly interpolate the cached sector operator at time ``t``."""
        if not (0.0 <= t <= self.T + 1e-12):
            raise ValueError(f"t={t} outside [0, {self.T}]")
        grid = self.sample_times
        t_clipped = float(np.clip(t, grid[0], grid[-1]))
        idx = int(np.clip(np.searchsorted(grid, t_clipped), 1, len(grid) - 1))
        t0, t1 = grid[idx - 1], grid[idx]
        w = (t_clipped - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (1.0 - w) * self.sample_operators[idx - 1] + w * self.sample_operators[idx]

    def operator_at(self, t: float) -> np.ndarray:
        """Embed the interpolated sector operator into the full
        :math:`2^L \\times 2^L` Hilbert space.

        Allocates a transient :math:`(2^L)^2` complex128 array per call.
        """
        O_sec = self._operator_sector_at(t)
        full = np.zeros((self._full_dim, self._full_dim), dtype=np.complex128)
        full[np.ix_(self.sector_indices, self.sector_indices)] = O_sec
        return full

    def overlap(self, t: float, state: np.ndarray) -> float:
        """Compute :math:`\\langle\\psi|G_t|\\psi\\rangle` directly in sector
        coordinates — does not allocate the embedded matrix."""
        state_sec = np.asarray(state, dtype=np.complex128)[self.sector_indices]
        O_sec = self._operator_sector_at(t)
        value = np.vdot(state_sec, O_sec @ state_sec)
        return float(np.real_if_close(value, tol=1_000.0).real)


def run_exact_backward_pass_sector(
    model: ExactSpinChainModel,
    T: float,
    zeta: float,
    *,
    n_samples: int = 64,
    expm_traceA: Optional[float] = None,
) -> SectorReducedBackwardData:
    """Run the exact backward pass restricted to the half-filling sector.

    Builds ``H_sec`` and ``{P_j_sec}`` as sparse :math:`d \\times d` matrices
    (:math:`d = \\binom{L}{L/2}`), constructs the action-only adjoint
    superoperator as a :class:`LinearOperator` on :math:`\\mathbb{C}^{d^2}`,
    and uses :func:`scipy.sparse.linalg.expm_multiply` to evolve
    :math:`\\text{vec}(I_d)` backwards in time at ``n_samples`` linearly
    spaced times in :math:`[0, T]`. The d²×d² superoperator is **never**
    materialised.
    """
    sector_indices = half_filling_sector_indices(model.L)
    d = sector_indices.shape[0]

    h_sec = project_to_sector(model.hamiltonian, sector_indices)
    projectors_sec = tuple(
        project_to_sector(P, sector_indices) for P in model.jump_projectors
    )
    superop = build_tilted_adjoint_action(h_sec, projectors_sec, model.alpha, zeta)

    # Initial vec(I_d) — 1s on the diagonal in column-major flat indexing.
    identity_vec = np.zeros(d * d, dtype=np.complex128)
    identity_vec[np.arange(d) * (d + 1)] = 1.0

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2 for linear interpolation")

    # Compute trace of the superoperator on the sector's operator space.
    # Required by expm_multiply's grid mode for accurate Padé scaling — without
    # it, grid mode silently degrades to ~1e-2 error at L=6, ζ=0.5, T=2.
    # Trace decomposition (HS basis):
    #   tr(i[H,·]) = i tr(H) d − i d tr(H) = 0
    #   tr(2α ζ P·P) = 2α ζ tr(P)²
    #   tr(2α (−½){P,·}) = 2α (−½)(d tr(P) + tr(P) d) = −2α d tr(P)
    if expm_traceA is None:
        traceA_value = 0.0
        for P in projectors_sec:
            trP = float(np.real(P.diagonal().sum()))
            traceA_value += 2.0 * model.alpha * (zeta * trP * trP - d * trP)
    else:
        traceA_value = float(expm_traceA)

    # expm_multiply with a sample grid: integrates from 0 to T with intermediate
    # snapshots. Returns shape (n_samples, d*d).
    samples_flat = expm_multiply(
        superop,
        identity_vec,
        start=0.0,
        stop=T,
        num=n_samples,
        endpoint=True,
        traceA=traceA_value,
    )
    sample_times_tau = np.linspace(0.0, T, n_samples)

    # Each row corresponds to evolution by tau = T - t for some t. Map to t-grid:
    # operator at time t equals the evolved-by-(T - t) start vector.
    sample_operators = samples_flat.reshape((n_samples, d, d), order="F")
    sample_times = T - sample_times_tau  # descending
    # Reverse so sample_times is ascending.
    sample_times = sample_times[::-1].copy()
    sample_operators = sample_operators[::-1].copy()

    return SectorReducedBackwardData(
        model=model,
        T=T,
        zeta=zeta,
        sector_indices=sector_indices,
        sample_times=sample_times,
        sample_operators=sample_operators,
    )
