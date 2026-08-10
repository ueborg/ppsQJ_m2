from __future__ import annotations

"""Case A QJ-PPS Gaussian backend: two competing measurements, H = 0.

Companion to ``gaussian_backend.py`` (Case B). The Case A model is:
  * no Hamiltonian (H = 0)
  * site-density channel  L_j  = c_j^dag c_j = (1 + i g_{2j} g_{2j+1}) / 2,
    measured pair (2j, 2j+1), at rate ``gamma_rate``   (j = 0 .. L-1)
  * bond/Bogoliubov channel L~_b = d_b^dag d_b, measured pair (2b, 2b+3),
    at rate ``alpha_rate``   (b = 0 .. L-2)  -- identical operator to Case B
  * lambda_A = alpha_rate / (alpha_rate + gamma_rate), with the two rates
    constrained to sum to 1 by the caller.

All Gaussian conventions (covariance sign, projective jump, branch-norm
definition, fast Gram-matrix survival) are inherited verbatim from
``gaussian_backend.py``. The ONLY structural differences from Case B:

  (1) GENERATOR: the non-Hermitian damping rate differs by channel --
      gamma_rate on the site pairs, alpha_rate on the bond pairs.

  (2) UNIFORM BRANCH-NORM TERM: the c-number part of the no-click survival
      probability is the RATE-WEIGHTED click count
          uniform_decay = gamma_rate * L + alpha_rate * (L - 1)
      NOT (single rate) * (number of pairs). In Case B all rates equal alpha
      so this collapses to ``alpha * n_monitored``; in Case A it does not.
      Using the Case B form here leaves the waiting-time distribution correct
      ONLY at the self-dual point gamma_rate == alpha_rate (lambda_A = 1/2)
      and silently corrupts every other lambda_A -- it would still pass a
      lambda_A=1/2 smoke test. Validated against the exact Fock-space backend
      in tests/validate_caseA.py.

  (3) CHANNEL SELECTION: when a jump fires, channel m is chosen with weight
          rate_m * 0.5 * (1 - Gamma[a_m, b_m])
      i.e. rate-weighted instantaneous click density. Case B factors the
      single rate out of the normalisation; Case A cannot.

  (4) JUMP: ``apply_projective_jump`` is channel-agnostic and reused unchanged.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .gaussian_backend import (
    bond_jump_pair,
    neel_covariance,
    covariance_from_orbitals,
    orbitals_from_covariance,
    apply_projective_jump,
    GaussianTrajectoryResult,
)


def site_jump_pair(site: int) -> tuple[int, int]:
    """Majorana pair for the on-site density n_j = c_j^dag c_j.

    n_j = (1 + i g_{2j} g_{2j+1}) / 2, so measuring n_j is measuring the
    parity i g_{2j} g_{2j+1} on the pair (2j, 2j+1) -- the two Majoranas on
    the same physical site (cf. bond_jump_pair, which straddles neighbours).

    Defined locally rather than added to gaussian_backend.py so the validated
    Case B file is left untouched (CASE_A spec section 9 backout plan).
    """
    return 2 * site, 2 * site + 1


def effective_generator_caseA(
    L: int, gamma_rate: float, alpha_rate: float
) -> np.ndarray:
    """Non-Hermitian Majorana generator for Case A (no Hamiltonian).

    Per measured pair (a, b) at rate r the damping is h[a,b] -= i r,
    h[b,a] += i r, matching the bond term in effective_generator(). With
    gamma_rate == 0 this reduces exactly to effective_generator(L, w=0,
    alpha=alpha_rate); with alpha_rate == 0 it is block-diagonal site-by-site.
    """
    h_eff = np.zeros((2 * L, 2 * L), dtype=np.complex128)
    for j in range(L):                        # site channel, rate gamma_rate
        a, b = site_jump_pair(j)
        h_eff[a, b] -= 1j * gamma_rate
        h_eff[b, a] += 1j * gamma_rate
    for bond in range(L - 1):                 # bond channel, rate alpha_rate
        a, b = bond_jump_pair(bond)
        h_eff[a, b] -= 1j * alpha_rate
        h_eff[b, a] += 1j * alpha_rate
    return h_eff


@dataclass(frozen=True)
class GaussianCaseAModel:
    """Cached Case A model. Mirrors GaussianChainModel where it must to be a
    drop-in for the trajectory/cloning machinery, but carries per-channel
    rates and the rate-weighted uniform decay."""
    L: int
    gamma_rate: float
    alpha_rate: float
    lambda_A: float
    h_effective: np.ndarray
    # Flat channel list: first ``n_site`` (= L) entries are site pairs at
    # gamma_rate, the remaining L-1 are bond pairs at alpha_rate.
    n_site: int
    jump_pairs: tuple[tuple[int, int], ...]
    ja: np.ndarray        # (2L-1,) int   first Majorana index of each pair
    jb: np.ndarray        # (2L-1,) int   second Majorana index of each pair
    rates: np.ndarray     # (2L-1,) float per-channel rate (gamma_rate|alpha_rate)
    uniform_decay: float  # gamma_rate*L + alpha_rate*(L-1)  [see module docstring (2)]
    gamma0: np.ndarray
    orbitals0: np.ndarray
    # Cached eigendecomposition of h_effective (see Case B for the rationale).
    h_eff_evals: np.ndarray
    h_eff_V: np.ndarray
    h_eff_V_inv: np.ndarray
    h_eff_VhV: np.ndarray


def build_caseA_model(
    L: int, gamma_rate: float, alpha_rate: float
) -> GaussianCaseAModel:
    h_eff = effective_generator_caseA(L, gamma_rate, alpha_rate)
    h_eff_c = np.asarray(h_eff, dtype=np.complex128)
    evals, V = np.linalg.eig(h_eff_c)
    V_inv = np.linalg.inv(V)

    pairs: list[tuple[int, int]] = []
    rates_list: list[float] = []
    for j in range(L):                        # site channels first
        pairs.append(site_jump_pair(j))
        rates_list.append(gamma_rate)
    for bond in range(L - 1):                 # then bond channels
        pairs.append(bond_jump_pair(bond))
        rates_list.append(alpha_rate)

    ja = np.array([p[0] for p in pairs], dtype=np.intp)
    jb = np.array([p[1] for p in pairs], dtype=np.intp)
    rates = np.array(rates_list, dtype=np.float64)

    gamma0 = neel_covariance(L)
    denom = gamma_rate + alpha_rate
    return GaussianCaseAModel(
        L=L,
        gamma_rate=gamma_rate,
        alpha_rate=alpha_rate,
        lambda_A=(alpha_rate / denom) if denom > 0.0 else float("nan"),
        h_effective=h_eff,
        n_site=L,
        jump_pairs=tuple(pairs),
        ja=ja, jb=jb, rates=rates,
        uniform_decay=float(gamma_rate * L + alpha_rate * (L - 1)),
        gamma0=gamma0,
        orbitals0=orbitals_from_covariance(gamma0),
        h_eff_evals=evals,
        h_eff_V=V,
        h_eff_V_inv=V_inv,
        h_eff_VhV=V.conj().T @ V,
    )


def gaussian_born_rule_trajectory_caseA(
    model: GaussianCaseAModel,
    T: float,
    rng: np.random.Generator,
    bisection_tol: float = 1e-6,
    *,
    gamma0_override: Optional[np.ndarray] = None,
    orbitals0_override: Optional[np.ndarray] = None,
) -> GaussianTrajectoryResult:
    """Exact Born-rule quantum-jump trajectory for Case A.

    Structurally identical to gaussian_born_rule_trajectory (Case B); the only
    differences are the rate-weighted uniform decay (model.uniform_decay) in
    the survival probability and the rate-weighted channel selection.

    ``jump_channels`` stores the raw flat channel index in 0 .. 2L-2:
    indices < model.n_site (= L) are site clicks, indices >= n_site are bond
    clicks. The caller derives the site/bond split from that boundary.
    """
    evals = model.h_eff_evals
    V = model.h_eff_V
    V_inv = model.h_eff_V_inv
    VhV = model.h_eff_VhV
    _ja = model.ja
    _jb = model.jb
    _rates = model.rates
    uniform_decay = model.uniform_decay
    n_channels = _ja.shape[0]

    orbitals = (
        np.asarray(model.orbitals0, dtype=np.complex128).copy()
        if orbitals0_override is None
        else np.asarray(orbitals0_override, dtype=np.complex128).copy()
    )
    cov = (
        np.asarray(model.gamma0, dtype=np.float64).copy()
        if gamma0_override is None
        else np.asarray(gamma0_override, dtype=np.float64).copy()
    )
    t = 0.0
    jump_times: list[float] = []
    jump_channels: list[int] = []

    while t < T:
        U = float(rng.uniform(0.0, 1.0))
        T_rem = T - t
        coeffs = V_inv @ orbitals  # (2L, L), reused in branch-norm + propagation

        def _branch_norm(dt: float) -> float:
            """No-click survival probability = |det R|^2 ... via Gram logdet,
            times the rate-weighted uniform decay exp(-uniform_decay * dt)."""
            if dt <= 0.0:
                return 1.0
            exp_d = np.exp(evals * dt)
            A = exp_d[:, None] * coeffs
            gram = A.conj().T @ (VhV @ A)
            try:
                L_chol = np.linalg.cholesky(gram)
                log_half = float(np.sum(np.log(np.abs(np.diag(L_chol)))))
                return float(np.exp(log_half - uniform_decay * dt))
            except np.linalg.LinAlgError:
                sign, logdet = np.linalg.slogdet(gram)
                if sign <= 0 or not np.isfinite(logdet):
                    return 0.0
                return float(np.exp(0.5 * logdet - uniform_decay * dt))

        bn_end = _branch_norm(T_rem)
        if bn_end >= U:
            # No jump in the remaining time -- propagate to T.
            exp_d = np.exp(evals * T_rem)
            orbs_tilde = V @ (exp_d[:, None] * coeffs)
            q_mat, _ = np.linalg.qr(orbs_tilde, mode="reduced")
            orbitals = q_mat
            cov = covariance_from_orbitals(q_mat)
            break

        try:
            dt_star = brentq(
                lambda dt: _branch_norm(dt) - U,
                0.0, T_rem, xtol=bisection_tol, maxiter=50,
            )
        except ValueError:
            dt_star = 0.5 * T_rem

        exp_d_star = np.exp(evals * dt_star)
        orbs_tilde = V @ (exp_d_star[:, None] * coeffs)
        q_mat, _ = np.linalg.qr(orbs_tilde, mode="reduced")
        orbitals = q_mat
        cov = covariance_from_orbitals(orbitals)
        t += dt_star

        # Rate-weighted channel selection (vectorised over all 2L-1 channels).
        probs = _rates * np.clip(0.5 * (1.0 - cov[_ja, _jb]), 0.0, 1.0)
        total = probs.sum()
        if total < 1e-15:
            break
        probs = probs / total
        channel = int(rng.choice(n_channels, p=probs))

        _, cov = apply_projective_jump(cov, model.jump_pairs[channel])
        orbitals = orbitals_from_covariance(cov)
        jump_times.append(t)
        jump_channels.append(channel)

    return GaussianTrajectoryResult(
        final_covariance=cov,
        final_orbitals=orbitals,
        n_jumps=len(jump_times),
        jump_times=jump_times,
        jump_channels=jump_channels,
    )
