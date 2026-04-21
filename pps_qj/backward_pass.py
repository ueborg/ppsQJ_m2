from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply

from pps_qj.exact_backend import ExactSpinChainModel, lindbladian_superoperator
from pps_qj.gaussian_backend import GaussianChainModel, orbitals_from_covariance, project_to_physical_covariance


def _monitoring_moment_matrices_fast(
    G: np.ndarray,
    C: np.ndarray,
    jump_pair: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised sandwich and anticommutator matrices for jump pair (a,b).

    Replaces the original O(n²) Python double-loop with O(1) numpy outer-product
    operations on the precomputed G = I - iC matrix.

    Mathematical identity (vectorised over all m,n simultaneously):
        f1[m,n] = G[a,b]*G[m,n] - G[a,m]*G[b,n] + G[a,n]*G[b,m]
              = g_ab*G - outer(G[a,:], G[b,:]) + outer(G[b,:], G[a,:])
        f2[m,n] = G[a,b]*G[m,n] - G[m,a]*G[n,b] + G[m,b]*G[n,a]
              = g_ab*G - outer(G[:,a], G[:,b]) + outer(G[:,b], G[:,a])
    """
    a, b = jump_pair
    g_ab = G[a, b]

    # Vectorised four-point functions
    f1 = g_ab * G - np.outer(G[a, :], G[b, :]) + np.outer(G[b, :], G[a, :])
    f2 = g_ab * G - np.outer(G[:, a], G[:, b]) + np.outer(G[:, b], G[:, a])

    # Anticommutator: antisymmetric part of C + 0.5*Re(f1+f2)
    anti = C + 0.5 * np.real(f1 + f2)
    anticommutator = 0.5 * (anti - anti.T)

    # Sandwich: 0.5*Re(1j*G + f2) where both indices share membership in {a,b}
    n = G.shape[0]
    in_pair = np.zeros(n, dtype=bool)
    in_pair[a] = True
    in_pair[b] = True
    same = np.outer(in_pair, in_pair) | np.outer(~in_pair, ~in_pair)
    sand = np.where(same, 0.5 * np.real(1j * G + f2), 0.0)
    sandwich = 0.5 * (sand - sand.T)

    return sandwich, anticommutator


def _monitoring_moment_matrices(
    covariance: np.ndarray,
    jump_pair: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Public wrapper — computes G then delegates to the fast vectorised kernel."""
    C = np.asarray(covariance, dtype=np.float64)
    n = C.shape[0]
    G = np.eye(n, dtype=np.complex128) - 1j * C
    return _monitoring_moment_matrices_fast(G, C, jump_pair)


def _clip_covariance_inplace(C: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Cheap O(n²) antisymmetric clamp — replaces the O(n³) eigh projection.

    Enforces |C[i,j]| ≤ 1-eps and exact antisymmetry without eigendecomposition.
    The full eigh-based projection is retained for the sample-point post-processing
    where correctness matters more than speed.
    """
    C = 0.5 * (C - C.T)                              # enforce antisymmetry
    np.clip(C, -(1.0 - eps), 1.0 - eps, out=C)       # enforce |entries| ≤ 1
    return C


def gaussian_backward_rhs(
    tau: float,
    y: np.ndarray,
    zeta: float,
    alpha: float,
    h_hamiltonian: np.ndarray,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    K: int,
    n: int,
    clip_epsilon: float = 1e-9,
) -> np.ndarray:
    """ODE right-hand side for the Gaussian backward pass.

    All model-derived quantities (a_idx, b_idx, K, h_hamiltonian) are passed
    as pre-computed arguments so the closure built in run_gaussian_backward_pass
    captures them without rebuilding on every call.
    """
    del tau
    C = y[:-1].reshape((n, n))
    C = _clip_covariance_inplace(C, eps=clip_epsilon)   # cheap O(n²), no eigh
    G = np.eye(n, dtype=np.complex128) - 1j * C

    q_vals = 0.5 * (1.0 - C[a_idx, b_idx])
    q_sum = float(q_vals.sum())
    scalar_rate = 2.0 * alpha * (zeta - 1.0) * q_sum

    g_ab_sum = G[a_idx, b_idx].sum()
    Ga     = G[a_idx, :]   # (K, n)
    Gb     = G[b_idx, :]   # (K, n)
    Ga_col = G[:, a_idx]   # (n, K)
    Gb_col = G[:, b_idx]   # (n, K)

    f1_sum = G * g_ab_sum - Ga.T @ Gb + Gb.T @ Ga
    f2_sum = G * g_ab_sum - Ga_col @ Gb_col.T + Gb_col @ Ga_col.T

    anti_raw = K * C + 0.5 * np.real(f1_sum + f2_sum)
    anticommutator_sum = 0.5 * (anti_raw - anti_raw.T)

    sand_raw = 0.5 * (K * C + np.real(f2_sum))
    sandwich_sum = 0.5 * (sand_raw - sand_raw.T)

    rhs = h_hamiltonian @ C - C @ h_hamiltonian
    rhs += 2.0 * alpha * (zeta * sandwich_sum - 0.5 * anticommutator_sum)
    rhs -= scalar_rate * C
    rhs = 0.5 * (rhs - rhs.T)

    out = np.empty_like(y)
    out[:-1] = rhs.reshape(n * n)
    out[-1] = scalar_rate
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
    sample_orbitals: np.ndarray   # (N, 2L, L) complex — for orbital-based overlap
    sample_z: np.ndarray
    sample_log_z: np.ndarray      # log(z) precomputed — avoids log(~0) in trajectories
    clip_epsilon: float = 1e-9

    def state_at(self, t: float) -> tuple[np.ndarray, float]:
        if not (0.0 <= t <= self.T):
            raise ValueError("t must lie in [0, T]")
        t_grid = self.T - self.sample_tau
        t = float(np.clip(t, t_grid[0], t_grid[-1]))
        idx = int(np.clip(np.searchsorted(t_grid, t), 1, len(t_grid) - 1))
        t0, t1 = t_grid[idx - 1], t_grid[idx]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        cov = (1.0 - w) * self.sample_covariances[idx - 1] + w * self.sample_covariances[idx]
        cov = _clip_covariance_inplace(cov, eps=self.clip_epsilon)
        z = float((1.0 - w) * self.sample_z[idx - 1] + w * self.sample_z[idx])
        return cov, z

    def orbitals_at(self, t: float) -> tuple[np.ndarray, float]:
        """Return (backward_orbitals W, log_z) at time t — for orbital-based overlap."""
        if not (0.0 <= t <= self.T):
            raise ValueError("t must lie in [0, T]")
        t_grid = self.T - self.sample_tau
        t = float(np.clip(t, t_grid[0], t_grid[-1]))
        idx = int(np.clip(np.searchsorted(t_grid, t), 1, len(t_grid) - 1))
        t0, t1 = t_grid[idx - 1], t_grid[idx]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        # Interpolate in orbital space (approximately correct for small intervals).
        orbs = (1.0 - w) * self.sample_orbitals[idx - 1] + w * self.sample_orbitals[idx]
        # Re-orthogonalise to keep columns orthonormal after interpolation.
        orbs, _ = np.linalg.qr(orbs, mode="reduced")
        log_z = float((1.0 - w) * self.sample_log_z[idx - 1] + w * self.sample_log_z[idx])
        return orbs, log_z

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
    rtol: float = 1e-5,
    atol: float = 1e-7,
    sample_points: int = 500,
    clip_epsilon: float = 1e-9,
    max_step: float | None = None,
    show_progress: bool = False,
) -> GaussianBackwardData:
    n = 2 * model.L
    y0 = np.zeros(n * n + 1, dtype=np.float64)
    y0[-1] = 0.0

    # Precompute model-derived quantities once — captured by closure so the
    # RHS never rebuilds them on each of the ~10,000+ solver evaluations.
    K = len(model.jump_pairs)
    a_idx = np.array([p[0] for p in model.jump_pairs], dtype=np.intp)
    b_idx = np.array([p[1] for p in model.jump_pairs], dtype=np.intp)
    _alpha = model.alpha
    _h_ham = model.h_hamiltonian

    def _rhs(tau: float, y: np.ndarray) -> np.ndarray:
        return gaussian_backward_rhs(
            tau, y,
            zeta=zeta, alpha=_alpha, h_hamiltonian=_h_ham,
            a_idx=a_idx, b_idx=b_idx, K=K, n=n,
            clip_epsilon=clip_epsilon,
        )

    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(
            total=T, unit="τ", unit_scale=True,
            desc=f"bwd L={model.L} ζ={zeta:.2f}",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f}τ [{elapsed}<{remaining}]",
            leave=False,
        )
        _last_tau = [0.0]

        def _rhs_tracked(tau, y):
            if tau > _last_tau[0]:
                pbar.update(tau - _last_tau[0])
                _last_tau[0] = tau
            return _rhs(tau, y)

        rhs_fn = _rhs_tracked
    else:
        pbar = None
        rhs_fn = _rhs

    try:
        solution = solve_ivp(
            fun=rhs_fn,
            t_span=(0.0, T),
            y0=y0,
            method="RK45",
            dense_output=True,
            rtol=rtol,
            atol=atol,
            max_step=max_step if max_step is not None else np.inf,
        )
    finally:
        if pbar is not None:
            pbar.close()

    if not solution.success:
        raise RuntimeError(f"Gaussian backward pass failed: {solution.message}")

    sample_tau = np.linspace(0.0, T, sample_points)
    samples = solution.sol(sample_tau)
    L = model.L
    covariances = np.empty((sample_tau.size, n, n), dtype=np.float64)
    orbitals_arr = np.empty((sample_tau.size, n, L), dtype=np.complex128)
    z_values = np.empty(sample_tau.size, dtype=np.float64)
    log_z_values = np.empty(sample_tau.size, dtype=np.float64)
    for idx, tau in enumerate(sample_tau):
        flat = samples[:, idx]
        cov = project_to_physical_covariance(flat[:-1].reshape((n, n)), eps=clip_epsilon)
        covariances[idx] = cov
        orbitals_arr[idx] = orbitals_from_covariance(cov)
        log_z_values[idx] = float(flat[-1])          # log(z) directly from ODE state
        z_values[idx] = float(np.exp(flat[-1]))

    return GaussianBackwardData(
        model=model,
        T=T,
        zeta=zeta,
        solution=solution,
        sample_tau=sample_tau,
        sample_covariances=covariances,
        sample_orbitals=orbitals_arr,
        sample_z=z_values,
        sample_log_z=log_z_values,
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
