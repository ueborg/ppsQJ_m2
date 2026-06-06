from __future__ import annotations

"""Exact Fock-space reference for the Case A two-measurement QJ model.

Small-L validation gold standard for gaussian_backend_caseA.py. H = 0; two
projective measurement channels:
  * site density   P_site_j = n_j = c_j^dag c_j           at rate gamma_rate
  * bond/Bogoliubov P_bond_b = d_b^dag d_b  -- built IDENTICALLY to
    build_exact_spin_chain_model so the bond channel is guaranteed to match
    the already-validated Case B exact<->Gaussian correspondence -- at rate
    alpha_rate.

Effective no-click generator  h_eff = -i (gamma_rate * sum_j P_site_j
                                          + alpha_rate * sum_b P_bond_b),
i.e. damping coefficient = rate per channel (same convention as Case B's
h_eff = H - i*alpha*jump_sum). Channel selection at a jump is rate-weighted:
prob_m proportional to rate_m * <psi|P_m|psi>.

Dense matrices; intended for L <= 10. This isolates the only genuinely new
piece (the site channel) so a Gaussian-vs-exact mismatch points straight at
the site-channel convention rather than the (validated) bond channel.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import expm_multiply

from .exact_backend import build_jordan_wigner_operators, neel_state
from pps_qj.observables.basic import entanglement_entropy_statevector


@dataclass(frozen=True)
class ExactCaseAModel:
    L: int
    gamma_rate: float
    alpha_rate: float
    site_projectors: tuple  # length L,   csr
    bond_projectors: tuple  # length L-1, csr
    all_projectors: tuple   # site then bond, length 2L-1
    rates: np.ndarray       # (2L-1,)
    h_effective: sp.csr_matrix
    initial_state: np.ndarray

    @property
    def dim(self) -> int:
        return 2 ** self.L


def build_exact_caseA_model(L: int, gamma_rate: float, alpha_rate: float) -> ExactCaseAModel:
    c_ops, cd_ops = build_jordan_wigner_operators(L)
    dim = 2 ** L

    site_projs = []
    for j in range(L):
        P = (cd_ops[j] @ c_ops[j]).tocsr()        # n_j (occupied projector)
        site_projs.append(P)

    bond_projs = []
    for bond in range(L - 1):
        left, right = bond, bond + 1
        d_op = 0.5 * (cd_ops[left] + c_ops[left] + c_ops[right] - cd_ops[right])
        P = (d_op.getH() @ d_op).tocsr()          # identical to Case B bond op
        bond_projs.append(P)

    jump_sum = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for P in site_projs:
        jump_sum = jump_sum + gamma_rate * P
    for P in bond_projs:
        jump_sum = jump_sum + alpha_rate * P
    h_eff = (-1j * jump_sum).tocsr()              # H = 0 in Case A

    rates = np.array([gamma_rate] * L + [alpha_rate] * (L - 1), dtype=np.float64)
    all_projs = tuple(site_projs) + tuple(bond_projs)

    return ExactCaseAModel(
        L=L,
        gamma_rate=gamma_rate,
        alpha_rate=alpha_rate,
        site_projectors=tuple(site_projs),
        bond_projectors=tuple(bond_projs),
        all_projectors=all_projs,
        rates=rates,
        h_effective=h_eff,
        initial_state=neel_state(L),
    )


def _propagate(model: ExactCaseAModel, psi: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0.0:
        return np.asarray(psi, dtype=np.complex128).copy()
    return np.asarray(expm_multiply((-1j * model.h_effective) * dt, psi), dtype=np.complex128)


def _survival(model: ExactCaseAModel, psi: np.ndarray, dt: float) -> float:
    psi_t = _propagate(model, psi, dt)
    return float(np.real(np.vdot(psi_t, psi_t)))


def caseA_qj_trajectory_exact(
    model: ExactCaseAModel,
    T: float,
    rng: np.random.Generator,
    tol: float = 1e-7,
) -> dict:
    """Born-rule QJ trajectory in the exact Fock space (zeta = 1).

    Returns a dict with the half-chain entanglement entropy at t = T plus
    click diagnostics. Channel selection is rate-weighted.
    """
    psi = np.asarray(model.initial_state, dtype=np.complex128).copy()
    t = 0.0
    n_jumps = 0
    n_site = 0
    n_bond = 0
    L = model.L
    projs = model.all_projectors
    rates = model.rates

    while t < T:
        U = float(rng.uniform(0.0, 1.0))
        T_rem = T - t
        if _survival(model, psi, T_rem) >= U:
            psi = _propagate(model, psi, T_rem)
            psi /= np.linalg.norm(psi)
            break
        try:
            dt_star = brentq(
                lambda dt: _survival(model, psi, dt) - U,
                0.0, T_rem, xtol=tol, maxiter=100,
            )
        except ValueError:
            dt_star = 0.5 * T_rem
        psi = _propagate(model, psi, dt_star)
        psi /= np.linalg.norm(psi)
        t += dt_star

        weights = np.array(
            [rates[m] * float(np.real(np.vdot(psi, projs[m] @ psi)))
             for m in range(len(projs))],
            dtype=np.float64,
        )
        weights = np.clip(weights, 0.0, None)
        tot = weights.sum()
        if tot < 1e-15:
            break
        weights /= tot
        m = int(rng.choice(len(projs), p=weights))
        psi = projs[m] @ psi
        psi = psi / np.linalg.norm(psi)
        n_jumps += 1
        if m < L:
            n_site += 1
        else:
            n_bond += 1

    S_half = float(entanglement_entropy_statevector(psi, L, L // 2))
    return {
        "S_half": S_half,
        "n_jumps": n_jumps,
        "n_site": n_site,
        "n_bond": n_bond,
    }
