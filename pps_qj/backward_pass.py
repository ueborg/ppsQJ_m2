from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply

from pps_qj.exact_backend import ExactSpinChainModel, lindbladian_superoperator
from pps_qj.gaussian_backend import GaussianChainModel, project_to_physical_covariance


def _four_point(G: np.ndarray, a: int, b: int, c: int, d: int) -> complex:
    return G[a, b] * G[c, d] - G[a, c] * G[b, d] + G[a, d] * G[b, c]


def _monitoring_moment_matrices(
    covariance: np.ndarray,
    jump_pair: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return matrices for <J B_mn J> and <{J, B_mn}>."""
    C = np.asarray(covariance, dtype=np.float64)
    G = np.eye(C.shape[0], dtype=np.complex128) - 1j * C
    a, b = jump_pair
    pair = {a, b}
    sandwich = np.zeros_like(C, dtype=np.float64)
    anticommutator = np.zeros_like(C, dtype=np.float64)

    for m in range(C.shape[0]):
        for n in range(m + 1, C.shape[1]):
            f1 = _four_point(G, a, b, m, n)
            f2 = _four_point(G, m, n, a, b)
            anti = C[m, n] + 0.5 * (f1 + f2)
            anticommutator[m, n] = float(np.real_if_close(anti, tol=1_000.0).real)
            anticommutator[n, m] = -anticommutator[m, n]

            same_membership = ((m in pair) == (n in pair))
            if same_membership:
                sand = 0.5 * (1j * G[m, n] + f2)
                sandwich[m, n] = float(np.real_if_close(sand, tol=1_000.0).real)
                sandwich[n, m] = -sandwich[m, n]

    return sandwich, anticommutator


def gaussian_backward_rhs(
    tau: float,
    y: np.ndarray,
    model: GaussianChainModel,
    zeta: float,
    clip_epsilon: float = 1e-9,
) -> np.ndarray:
    del tau
    n = 2 * model.L
    C = y[:-1].reshape((n, n))
    C = project_to_physical_covariance(C, eps=clip_epsilon)

    q_sum = 0.0
    for pair in model.jump_pairs:
        a, b = pair
        q_sum += 0.5 * (1.0 - C[a, b])
    scalar_rate = model.gamma_m * (zeta - 1.0) * q_sum

    rhs = model.h_hamiltonian @ C - C @ model.h_hamiltonian
    for pair in model.jump_pairs:
        sandwich, anticommutator = _monitoring_moment_matrices(C, pair)
        rhs += model.gamma_m * (zeta * sandwich - 0.5 * anticommutator)

    rhs -= scalar_rate * C
    rhs = 0.5 * (rhs - rhs.T)
    dlogz_dtau = scalar_rate

    out = np.empty_like(y)
    out[:-1] = rhs.reshape(n * n)
    out[-1] = dlogz_dtau
    return out


def k_matrix_from_covariance(covariance: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    C = project_to_physical_covariance(covariance, eps=eps)
    spectrum, vectors = np.linalg.eigh(1j * C)
    clipped = np.clip(spectrum.real, -1.0 + eps, 1.0 - eps)
    matrix = vectors @ np.diag(np.arctanh(clipped)) @ vectors.conj().T
    K = np.real_if_close(-2j * matrix, tol=1_000.0).real
    return 0.5 * (K - K.T)


def gaussian_mu_from_covariance_and_z(covariance: np.ndarray, z_scalar: float, eps: float = 1e-9) -> float:
    spectrum = np.linalg.eigvalsh(1j * project_to_physical_covariance(covariance, eps=eps))
    positive = spectrum[spectrum > 0]
    clipped = np.clip(positive.real, 0.0, 1.0 - eps)
    return float(np.log(z_scalar) + 0.5 * np.sum(np.log(1.0 - clipped**2)))


@dataclass(frozen=True)
class GaussianBackwardData:
    model: GaussianChainModel
    T: float
    zeta: float
    solution: object
    sample_tau: np.ndarray
    sample_covariances: np.ndarray
    sample_z: np.ndarray
    clip_epsilon: float = 1e-9

    def state_at(self, t: float) -> tuple[np.ndarray, float]:
        if not (0.0 <= t <= self.T):
            raise ValueError("t must lie in [0, T]")
        tau = self.T - t
        flat = self.solution.sol(tau)
        n = 2 * self.model.L
        covariance = flat[:-1].reshape((n, n))
        covariance = project_to_physical_covariance(covariance, eps=self.clip_epsilon)
        z_scalar = float(np.exp(flat[-1]))
        return covariance, z_scalar

    def covariance_at(self, t: float) -> np.ndarray:
        return self.state_at(t)[0]

    def z_at(self, t: float) -> float:
        return self.state_at(t)[1]

    def generator_at(self, t: float) -> tuple[np.ndarray, float]:
        covariance, z_scalar = self.state_at(t)
        return k_matrix_from_covariance(covariance, eps=self.clip_epsilon), gaussian_mu_from_covariance_and_z(
            covariance,
            z_scalar,
            eps=self.clip_epsilon,
        )


def run_gaussian_backward_pass(
    model: GaussianChainModel,
    T: float,
    zeta: float,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    sample_points: int = 257,
    clip_epsilon: float = 1e-9,
    max_step: float | None = None,
) -> GaussianBackwardData:
    n = 2 * model.L
    y0 = np.zeros(n * n + 1, dtype=np.float64)
    y0[-1] = 0.0
    solution = solve_ivp(
        fun=lambda tau, y: gaussian_backward_rhs(
            tau,
            y,
            model=model,
            zeta=zeta,
            clip_epsilon=clip_epsilon,
        ),
        t_span=(0.0, T),
        y0=y0,
        method="DOP853",
        dense_output=True,
        rtol=rtol,
        atol=atol,
        max_step=max_step if max_step is not None else np.inf,
    )
    if not solution.success:
        raise RuntimeError(f"Gaussian backward pass failed: {solution.message}")

    sample_tau = np.linspace(0.0, T, sample_points)
    samples = solution.sol(sample_tau)
    covariances = np.empty((sample_tau.size, n, n), dtype=np.float64)
    z_values = np.empty(sample_tau.size, dtype=np.float64)
    for idx, tau in enumerate(sample_tau):
        flat = samples[:, idx]
        covariances[idx] = project_to_physical_covariance(
            flat[:-1].reshape((n, n)),
            eps=clip_epsilon,
        )
        z_values[idx] = float(np.exp(flat[-1]))

    return GaussianBackwardData(
        model=model,
        T=T,
        zeta=zeta,
        solution=solution,
        sample_tau=sample_tau,
        sample_covariances=covariances,
        sample_z=z_values,
        clip_epsilon=clip_epsilon,
    )


@dataclass
class ExactBackwardData:
    model: ExactSpinChainModel
    T: float
    zeta: float
    superoperator_adjoint: object
    identity_vector: np.ndarray
    _cache: dict[float, np.ndarray] = field(default_factory=dict)

    def operator_at(self, t: float) -> np.ndarray:
        if not (0.0 <= t <= self.T):
            raise ValueError("t must lie in [0, T]")
        key = round(float(t), 12)
        if key not in self._cache:
            tau = self.T - float(t)
            vector = expm_multiply(self.superoperator_adjoint * tau, self.identity_vector)
            self._cache[key] = np.asarray(vector, dtype=np.complex128).reshape(
                (self.model.dim, self.model.dim),
                order="F",
            )
        return self._cache[key]

    def overlap(self, t: float, state: np.ndarray) -> float:
        operator = self.operator_at(t)
        value = np.vdot(state, operator @ state)
        return float(np.real_if_close(value, tol=1_000.0).real)


def run_exact_backward_pass(
    model: ExactSpinChainModel,
    T: float,
    zeta: float,
) -> ExactBackwardData:
    superoperator_adjoint = lindbladian_superoperator(model, zeta=zeta, adjoint=True)
    identity_vector = np.eye(model.dim, dtype=np.complex128).reshape(model.dim * model.dim, order="F")
    return ExactBackwardData(
        model=model,
        T=T,
        zeta=zeta,
        superoperator_adjoint=superoperator_adjoint,
        identity_vector=identity_vector,
    )
