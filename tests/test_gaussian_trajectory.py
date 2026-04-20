"""Sanity-check tests for gaussian_born_rule_trajectory.

Test 1: trajectory-averaged covariance vs Lindblad master equation.
Test 2: trajectory-averaged S_{L/2} vs exact backend Born-rule trajectories.
"""
from __future__ import annotations

import numpy as np
import pytest

from pps_qj.gaussian_backend import (
    build_gaussian_chain_model,
    entanglement_entropy,
    gaussian_born_rule_trajectory,
)
from pps_qj.exact_backend import (
    build_exact_spin_chain_model,
    build_jordan_wigner_operators,
    integrate_lindblad,
    ordinary_quantum_jump_trajectory,
    half_chain_entanglement_entropy,
)


def _covariance_from_density_matrix(
    rho: np.ndarray, L: int, c_ops, cd_ops
) -> np.ndarray:
    """Extract the Majorana covariance matrix from a density matrix."""
    n = 2 * L
    gamma_ops = []
    for j in range(L):
        gamma_ops.append((c_ops[j] + cd_ops[j]).toarray())
        gamma_ops.append(1j * (cd_ops[j] - c_ops[j]).toarray())
    C = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(a + 1, n):
            val = np.trace(gamma_ops[a] @ gamma_ops[b] @ rho)
            val -= np.trace(gamma_ops[b] @ gamma_ops[a] @ rho)
            C[a, b] = 0.5 * float(np.real(1j * val))
            C[b, a] = -C[a, b]
    return C


@pytest.mark.slow
def test_trajectory_averaged_covariance_vs_lindblad():
    """Trajectory-averaged Γ agrees with Lindblad to within statistical tolerance."""
    L, w, alpha, T = 4, 0.5, 0.5, 1.0
    N_traj = 2000

    # Gaussian trajectories
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    rng = np.random.default_rng(42)
    C_avg = np.zeros((2 * L, 2 * L), dtype=np.float64)
    for _ in range(N_traj):
        result = gaussian_born_rule_trajectory(model, T=T, rng=rng)
        C_avg += result.final_covariance
    C_avg /= N_traj

    # Lindblad reference
    exact = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
    _, rhos = integrate_lindblad(exact, T=T, t_eval=np.array([T]))
    rho_lindblad = rhos[-1]
    c_ops, cd_ops = build_jordan_wigner_operators(L)
    C_lindblad = _covariance_from_density_matrix(rho_lindblad, L, c_ops, cd_ops)

    max_err = float(np.max(np.abs(C_avg - C_lindblad)))
    threshold = 3.0 / np.sqrt(N_traj)  # ≈ 0.067
    assert max_err < threshold, (
        f"max |C_gauss - C_lindblad| = {max_err:.4f} exceeds {threshold:.4f}"
    )


@pytest.mark.slow
def test_entropy_vs_exact_backend():
    """Born-rule <S_{L/2}> agrees between Gaussian and exact backends."""
    L, w, alpha, T = 4, 0.5, 0.5, 1.0
    N_traj = 2000

    # Gaussian Born-rule
    gm = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    rng_g = np.random.default_rng(100)
    S_gauss = []
    for _ in range(N_traj):
        result = gaussian_born_rule_trajectory(gm, T=T, rng=rng_g)
        S_gauss.append(entanglement_entropy(result.final_covariance, L // 2))
    S_gauss = np.array(S_gauss)

    # Exact Born-rule
    ex = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
    rng_e = np.random.default_rng(200)
    S_exact = []
    for _ in range(N_traj):
        traj = ordinary_quantum_jump_trajectory(ex, T=T, rng=rng_e)
        S_exact.append(half_chain_entanglement_entropy(traj.final_state, L))
    S_exact = np.array(S_exact)

    mean_g = float(np.mean(S_gauss))
    mean_e = float(np.mean(S_exact))
    se_g = float(np.std(S_gauss) / np.sqrt(N_traj))
    se_e = float(np.std(S_exact) / np.sqrt(N_traj))
    combined_se = np.sqrt(se_g**2 + se_e**2)

    z_score = abs(mean_g - mean_e) / combined_se
    assert z_score < 2.0, (
        f"|<S>_gauss - <S>_exact| / sigma = {z_score:.2f} "
        f"(means: {mean_g:.4f} vs {mean_e:.4f})"
    )


if __name__ == "__main__":
    print("Running test_trajectory_averaged_covariance_vs_lindblad ...")
    test_trajectory_averaged_covariance_vs_lindblad()
    print("PASS")
    print("Running test_entropy_vs_exact_backend ...")
    test_entropy_vs_exact_backend()
    print("PASS")
    print("All tests passed.")
