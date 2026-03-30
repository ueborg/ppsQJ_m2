from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pps_qj.backends.base import StateBackend
from pps_qj.core.numerics import heff_from, safe_normalize
from pps_qj.types import ModelSpec


@dataclass
class ExactStateBackend(StateBackend):
    model: ModelSpec
    psi: np.ndarray

    def __post_init__(self) -> None:
        self.psi = np.asarray(self.psi, dtype=np.complex128)
        self.normalize()
        self.H = np.asarray(self.model.H, dtype=np.complex128)
        self.jump_ops = [np.asarray(op, dtype=np.complex128) for op in self.model.jump_ops]
        self.heff = heff_from(self.H, self.jump_ops)
        self.A = -1j * self.heff
        self._vals, self._vecs = np.linalg.eig(self.A)
        self._inv_vecs = np.linalg.inv(self._vecs)

        self._single_projector = False
        self._projector = None
        if len(self.jump_ops) == 1 and np.allclose(self.H, 0.0):
            gamma = self.model.gamma
            if gamma > 0:
                P = (self.jump_ops[0].conj().T @ self.jump_ops[0]) / gamma
                if np.allclose(P @ P, P, atol=1e-9, rtol=1e-9):
                    self._single_projector = True
                    self._projector = P

    def copy(self) -> "ExactStateBackend":
        return ExactStateBackend(self.model, self.psi.copy())

    def normalize(self) -> None:
        self.psi = safe_normalize(self.psi)

    def purity(self) -> float:
        rho = np.outer(self.psi, self.psi.conj())
        return float(np.real(np.trace(rho @ rho)))

    def _propagated(self, tau: float) -> np.ndarray:
        if tau == 0.0:
            return self.psi.copy()
        phase = np.exp(self._vals * tau)
        U = self._vecs @ np.diag(phase) @ self._inv_vecs
        return U @ self.psi

    def propagate_no_click(self, tau: float) -> None:
        if tau < 0:
            raise ValueError("tau must be non-negative")
        self.psi = safe_normalize(self._propagated(tau))

    def survival(self, tau: float) -> float:
        if tau < 0:
            return 1.0
        x = self._propagated(tau)
        return float(np.real(np.vdot(x, x)))

    def channel_rates(self) -> np.ndarray:
        rates = []
        for op in self.jump_ops:
            y = op @ self.psi
            rates.append(np.real(np.vdot(y, y)))
        return np.asarray(rates, dtype=np.float64)

    def apply_jump(self, j: int) -> None:
        self.psi = safe_normalize(self.jump_ops[j] @ self.psi)

    def analytic_waiting_time(self, r: float) -> float | None:
        if not self._single_projector:
            return None

        gamma = self.model.gamma
        q = float(np.real(np.vdot(self.psi, self._projector @ self.psi)))
        q = min(max(q, 0.0), 1.0)
        s_inf = 1.0 - q
        if r < s_inf + 1e-15:
            return float("inf")
        if q <= 1e-15:
            return float("inf")

        arg = (r - (1.0 - q)) / q
        arg = min(max(arg, 1e-15), 1.0)
        return -(1.0 / gamma) * np.log(arg)
