from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pps_qj.backends.base import StateBackend
from pps_qj.core.numerics import matrix_exponential
from pps_qj.types import ModelSpec


@dataclass
class GaussianStateBackend(StateBackend):
    """Gaussian-state backend using the orbital matrix V as primary state.

    Implements Algorithm 3 from Section 5.12.17 of the theory document.
    The state is represented by a 2L x L complex orbital matrix V satisfying
    V†V = I_L, from which the covariance matrix is derived as
    Γ = i(2VV† - I_{2L})  (eq 176).
    """

    model: ModelSpec
    Gamma: np.ndarray

    def __post_init__(self) -> None:
        if self.model.majorana_A_eff is None:
            raise ValueError("model.majorana_A_eff is required for Gaussian backend")
        if not self.model.majorana_jump_pairs:
            raise ValueError("model.majorana_jump_pairs is required for Gaussian backend")
        self.A_eff = np.asarray(self.model.majorana_A_eff, dtype=np.complex128)
        self.jump_pairs = list(self.model.majorana_jump_pairs)
        self.Gamma = np.asarray(self.Gamma, dtype=np.float64)

        # Derive V from Gamma (or use provided V)
        if self.model.initial_V is not None:
            self.V = np.asarray(self.model.initial_V, dtype=np.complex128).copy()
        else:
            self.V = self._derive_V_from_gamma(self.Gamma)

        # Ensure Gamma is consistent with V
        self.Gamma = self._gamma_from_V(self.V)

    def _derive_V_from_gamma(self, Gamma: np.ndarray) -> np.ndarray:
        """Derive orbital matrix V from covariance matrix Gamma.

        V consists of the -1 eigenvectors of iΓ, consistent with Γ = i(2VV† - I).
        """
        n = Gamma.shape[0]
        L = n // 2
        vals, vecs = np.linalg.eig(1j * Gamma)
        # Select eigenvectors with eigenvalue closest to -1
        idx = np.argsort(vals.real)[:L]
        V = vecs[:, idx]
        V, _ = np.linalg.qr(V, mode="reduced")
        return V

    def _gamma_from_V(self, V: np.ndarray) -> np.ndarray:
        """Compute Γ = i(2VV† - I) from V.  (eq 176/180)

        When V's columns are the -1 eigenvectors of iΓ,
        iΓ V = -V, so i·Γ = I - 2VV†, giving Γ = i(2VV† - I).
        """
        n = V.shape[0]
        Gamma = 1j * (2.0 * (V @ V.conj().T) - np.eye(n))
        return np.real(Gamma)

    def copy(self) -> "GaussianStateBackend":
        be = GaussianStateBackend(self.model, self.Gamma.copy())
        be.V = self.V.copy()
        return be

    def normalize(self) -> None:
        """QR-normalize the orbital matrix V, then recompute Γ."""
        self.V, _ = np.linalg.qr(self.V, mode="reduced")
        self.Gamma = self._gamma_from_V(self.V)

    def purity(self) -> float:
        """Compute Tr(-Γ²)/(2L). For pure states this equals 1."""
        m = self.Gamma @ self.Gamma
        n = self.Gamma.shape[0]
        return float(np.real(np.trace(-m)) / n)

    def propagate_no_click(self, tau: float) -> None:
        """Propagate under no-click evolution and normalize.

        V_new = QR(M(τ) V)  where M = exp(h_eff τ)  (Algorithm 3, lines 1-5)
        """
        if tau < 0:
            raise ValueError("tau must be non-negative")
        if tau == 0.0:
            return
        M = matrix_exponential(self.A_eff, tau)
        V_tilde = M @ self.V
        self.V, _ = np.linalg.qr(V_tilde, mode="reduced")
        self.Gamma = self._gamma_from_V(self.V)

    def survival(self, tau: float) -> float:
        """Exact no-click survival probability for the current Gaussian state.

        S(tau) = exp(-gamma*(L-1)*tau/2) * |det R|

        where V_tilde = QR = exp(h_eff tau) V.

        Uses QR of V_tilde (not the Gram matrix) to avoid catastrophic
        cancellation at large tau.  When the standard matrix exponential
        overflows (eigenvalue * tau exceeds float64 range), falls back to
        a balanced computation that shifts eigenvalues by -lambda_max so
        all exponents are non-positive, then compensates analytically.
        """
        if tau < 0:
            return 1.0
        if tau == 0.0:
            return 1.0

        M = matrix_exponential(self.A_eff, tau)
        V_tilde = M @ self.V

        if not np.all(np.isfinite(V_tilde)):
            return self._survival_balanced(tau)

        return self._survival_from_vtilde(V_tilde, tau)

    def _survival_balanced(self, tau: float) -> float:
        """Overflow-safe survival using eigenvalue-shifted matrix exponential."""
        vals, vecs = np.linalg.eig(self.A_eff)
        inv_vecs = np.linalg.inv(vecs)
        lambda_max = float(np.max(vals.real))

        shifted_exp = np.diag(np.exp((vals - lambda_max) * tau))
        M_balanced = vecs @ shifted_exp @ inv_vecs
        V_balanced = M_balanced @ self.V

        if not np.all(np.isfinite(V_balanced)):
            return 0.0

        _, R = np.linalg.qr(V_balanced, mode="reduced")
        diag_abs = np.abs(np.diag(R))
        if np.any(diag_abs <= 1e-300):
            return 0.0

        L_dim = R.shape[0]
        log_abs_det_R = float(np.sum(np.log(diag_abs))) + lambda_max * tau * L_dim

        decay_rate = 0.5 * self.model.gamma * len(self.jump_pairs)
        log_s = -decay_rate * tau + log_abs_det_R
        if log_s > 0.0:
            log_s = 0.0
        if log_s < -745.0:
            return 0.0
        return float(np.exp(log_s))

    def _survival_from_vtilde(self, V_tilde: np.ndarray, tau: float) -> float:
        """Compute survival from a finite V_tilde via QR."""
        _, R = np.linalg.qr(V_tilde, mode="reduced")
        diag_abs = np.abs(np.diag(R))
        if np.any(diag_abs <= 1e-300):
            return 0.0
        log_abs_det_R = float(np.sum(np.log(diag_abs)))

        decay_rate = 0.5 * self.model.gamma * len(self.jump_pairs)
        log_s = -decay_rate * tau + log_abs_det_R
        if log_s > 0.0:
            log_s = 0.0
        if log_s < -745.0:
            return 0.0
        return float(np.exp(log_s))

    def channel_rates(self) -> np.ndarray:
        """Compute rate_j = gamma * (1 - Γ_{a_j, b_j})/2 for all channels.  (eq 209/210)"""
        rates = np.empty(len(self.jump_pairs), dtype=np.float64)
        for idx, (a, b) in enumerate(self.jump_pairs):
            q = 0.5 * (1.0 - self.Gamma[a, b])
            q = min(max(float(q), 0.0), 1.0)
            rates[idx] = self.model.gamma * q
        return rates

    def apply_jump(self, j: int) -> None:
        """Apply jump on bond j using the covariance matrix update (eq 208).

        Implements Algorithm 3, ApplyJump (lines 15-28).
        """
        a, b = self.jump_pairs[j]
        G = self.Gamma
        sigma = float(G[a, b])
        denom = 1.0 - sigma
        if abs(denom) < 1e-12:
            raise ValueError("Selected jump has near-zero probability")

        n = G.shape[0]

        # Extract columns a and b
        u = G[:, a].copy()
        v = G[:, b].copy()

        # Rank-2 antisymmetric update on complement block (eq 207)
        mask = np.ones(n, dtype=bool)
        mask[a] = False
        mask[b] = False
        keep = np.where(mask)[0]

        u_k = u[keep]
        v_k = v[keep]
        correction = (np.outer(u_k, v_k) - np.outer(v_k, u_k)) / denom
        G_new = G.copy()
        G_new[np.ix_(keep, keep)] += correction

        # Zero cross-block entries (eq 202)
        G_new[a, :] = 0.0
        G_new[:, a] = 0.0
        G_new[b, :] = 0.0
        G_new[:, b] = 0.0

        # Fix pair block: Γ'_{ab} = +1, Γ'_{ba} = -1 (empty mode after annihilation)
        G_new[a, b] = 1.0
        G_new[b, a] = -1.0

        self.Gamma = G_new

        # Re-derive V from updated Gamma (Algorithm 3, line 27)
        self.V = self._derive_V_from_gamma(self.Gamma)

    def analytic_waiting_time(self, r: float) -> float | None:
        """Return None to force bisection with exact survival probability."""
        return None
