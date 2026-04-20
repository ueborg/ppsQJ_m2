from __future__ import annotations

import numpy as np

from pps_qj.exact_backend import (
    _propagate_unnormalized,
    build_exact_spin_chain_model,
    half_chain_entanglement_entropy,
    integrate_lindblad,
    ordinary_quantum_jump_trajectory,
    postselected_no_click_trajectory,
)
from pps_qj.observables.basic import entanglement_entropy_statevector


SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def _kron_many_dense(operators: list[np.ndarray]) -> np.ndarray:
    out = operators[0]
    for op in operators[1:]:
        out = np.kron(out, op)
    return out


def _single_site_dense(L: int, site: int, op: np.ndarray) -> np.ndarray:
    operators = [I2.copy() for _ in range(L)]
    operators[site] = op
    return _kron_many_dense(operators)


def _spin_xy_hamiltonian(L: int, w: float) -> np.ndarray:
    hamiltonian = np.zeros((2**L, 2**L), dtype=np.complex128)
    for site in range(L - 1):
        x_left = _single_site_dense(L, site, SX)
        x_right = _single_site_dense(L, site + 1, SX)
        y_left = _single_site_dense(L, site, SY)
        y_right = _single_site_dense(L, site + 1, SY)
        hamiltonian += 0.5 * w * (x_left @ x_right + y_left @ y_right)
    return hamiltonian


def _reduced_density_entropy(psi: np.ndarray, L: int, cut: int) -> float:
    dim_left = 2**cut
    dim_right = 2 ** (L - cut)
    psi_matrix = np.asarray(psi, dtype=np.complex128).reshape((dim_left, dim_right))
    rho_left = psi_matrix @ psi_matrix.conj().T
    eigenvalues = np.linalg.eigvalsh(rho_left)
    eigenvalues = np.clip(eigenvalues.real, 1e-15, 1.0)
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def test_exact_hamiltonian_matches_target_xy_chain() -> None:
    model = build_exact_spin_chain_model(L=4, w=0.7, alpha=0.2)
    expected = _spin_xy_hamiltonian(model.L, model.w)
    np.testing.assert_allclose(model.hamiltonian.toarray(), expected, atol=1e-12)


def test_monitored_projectors_are_bond_fermion_projectors() -> None:
    model = build_exact_spin_chain_model(L=4, w=0.5, alpha=0.4)
    identity = np.eye(model.dim, dtype=np.complex128)
    for bond, projector in enumerate(model.jump_projectors):
        gamma_left = (model.c_ops[bond] + model.cd_ops[bond]).toarray()
        gamma_right = (1j * (model.cd_ops[bond + 1] - model.c_ops[bond + 1])).toarray()
        expected = 0.5 * (identity + 1j * gamma_left @ gamma_right)
        dense_projector = projector.toarray()
        np.testing.assert_allclose(dense_projector, dense_projector.conj().T, atol=1e-12)
        np.testing.assert_allclose(dense_projector @ dense_projector, dense_projector, atol=1e-12)
        np.testing.assert_allclose(dense_projector, expected, atol=1e-12)


def test_no_click_norm_loss_matches_total_jump_rate() -> None:
    model = build_exact_spin_chain_model(L=4, w=0.5, alpha=0.4)
    dt = 1e-6
    psi0 = model.initial_state
    psi_tilde = _propagate_unnormalized(model, psi0, dt)
    survival = float(np.real(np.vdot(psi_tilde, psi_tilde)))
    expected_rate = 2.0 * model.alpha * sum(
        float(np.real(np.vdot(psi0, projector @ psi0))) for projector in model.jump_projectors
    )
    finite_difference_rate = (1.0 - survival) / dt
    np.testing.assert_allclose(finite_difference_rate, expected_rate, atol=2e-6)


def test_ordinary_quantum_jumps_match_exact_lindblad_average() -> None:
    model = build_exact_spin_chain_model(L=2, w=0.7, alpha=0.4)
    rng = np.random.default_rng(123)
    n_trajectories = 4000
    rho_mc = np.zeros((model.dim, model.dim), dtype=np.complex128)
    for _ in range(n_trajectories):
        trajectory = ordinary_quantum_jump_trajectory(model, T=0.6, rng=rng)
        psi = trajectory.final_state
        rho_mc += np.outer(psi, psi.conj())
    rho_mc /= n_trajectories

    _, rhos = integrate_lindblad(model, T=0.6, t_eval=np.array([0.6]))
    np.testing.assert_allclose(rho_mc, rhos[-1], atol=8e-3)


def test_postselected_no_click_trajectory_matches_deterministic_heff_evolution() -> None:
    model = build_exact_spin_chain_model(L=4, w=0.5, alpha=0.4)
    trajectory = postselected_no_click_trajectory(model, T=0.7)
    expected = _propagate_unnormalized(model, model.initial_state, 0.7)
    expected = expected / np.linalg.norm(expected)
    phase = np.vdot(expected, trajectory.final_state)
    corrected = trajectory.final_state if abs(phase) == 0.0 else trajectory.final_state * np.exp(-1j * np.angle(phase))
    np.testing.assert_allclose(corrected, expected, atol=1e-12)
    assert trajectory.n_jumps == 0
    assert trajectory.diagnostics["survival_probability"] > 0.0


def test_half_chain_entropy_matches_direct_reduced_density_matrix() -> None:
    product_model = build_exact_spin_chain_model(L=12, w=0.0, alpha=0.5)
    assert half_chain_entanglement_entropy(product_model.initial_state, product_model.L) < 1e-11

    rng = np.random.default_rng(20260413)
    psi = rng.normal(size=2**8) + 1j * rng.normal(size=2**8)
    psi = psi / np.linalg.norm(psi)
    expected = _reduced_density_entropy(psi, L=8, cut=4)
    observed = half_chain_entanglement_entropy(psi, L=8)
    helper = entanglement_entropy_statevector(psi, 8, 4)
    np.testing.assert_allclose(observed, expected, atol=1e-12)
    np.testing.assert_allclose(helper, expected, atol=1e-12)
