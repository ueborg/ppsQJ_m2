from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from pps_qj.backward_pass import run_exact_backward_pass, run_gaussian_backward_pass
from pps_qj.doob_wtmc import conditioned_survival_gaussian, doob_exact_trajectory, doob_gaussian_trajectory
from pps_qj.exact_backend import (
    build_exact_spin_chain_model,
    exact_model_consistency,
    ordinary_quantum_jump_trajectory,
    procedure_a_trajectory,
    procedure_b_trajectory,
    procedure_c_local_trajectory,
)
from pps_qj.gaussian_backend import (
    apply_projective_jump,
    build_gaussian_chain_model,
    bond_jump_pair,
    covariance_from_orbitals,
    entanglement_entropy,
)
from pps_qj.overlaps import gaussian_overlap


def _single_mode_analytics(zeta: float, alpha: float = 0.5, T: float = 2.0, q0: float = 0.5) -> dict[str, float]:
    survival = (1.0 - q0) + q0 * np.exp(-2.0 * alpha * T)
    partition = (1.0 - q0) + q0 * np.exp(-2.0 * (1.0 - zeta) * alpha * T)
    q_no_click = survival / partition
    r_no_click = survival**zeta
    return {
        "S": float(survival),
        "Z": float(partition),
        "Q_no_click": float(q_no_click),
        "R_no_click": float(r_no_click),
    }


def _histogram(counts: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    counts = np.asarray(counts, dtype=int)
    minlength = int(np.max(counts, initial=0)) + 1
    if weights is None:
        hist = np.bincount(counts, minlength=minlength).astype(np.float64)
    else:
        hist = np.bincount(counts, weights=np.asarray(weights, dtype=np.float64), minlength=minlength).astype(np.float64)
    return hist / hist.sum()


@lru_cache(maxsize=None)
def _single_mode_models(zeta: float, T: float = 2.0):
    exact = build_exact_spin_chain_model(L=2, w=0.0, alpha=0.5)
    gauss = build_gaussian_chain_model(L=2, w=0.0, alpha=0.5)
    return exact, gauss, run_exact_backward_pass(exact, T, zeta), run_gaussian_backward_pass(gauss, T, zeta, sample_points=65)


@lru_cache(maxsize=None)
def _single_mode_algorithm_counts(zeta: float, n_traj: int, seed: int = 123, T: float = 2.0) -> dict[str, np.ndarray]:
    exact, gauss, _, backward_gauss = _single_mode_models(zeta, T)
    outputs: dict[str, np.ndarray] = {}

    rng = np.random.default_rng(seed)
    outputs["doob"] = np.array(
        [doob_gaussian_trajectory(gauss, backward_gauss, T, zeta, rng).n_jumps for _ in range(n_traj)],
        dtype=int,
    )

    rng = np.random.default_rng(seed)
    outputs["procedure_a"] = np.array(
        [procedure_a_trajectory(exact, T, zeta, rng).n_jumps for _ in range(n_traj)],
        dtype=int,
    )

    rng = np.random.default_rng(seed)
    outputs["procedure_b"] = np.array(
        [procedure_b_trajectory(exact, T, zeta, rng).n_jumps for _ in range(n_traj)],
        dtype=int,
    )

    rng = np.random.default_rng(seed)
    outputs["procedure_c"] = np.array(
        [procedure_c_local_trajectory(exact, T, zeta, rng).n_jumps for _ in range(n_traj)],
        dtype=int,
    )
    return outputs


@lru_cache(maxsize=None)
def _single_mode_born_counts(n_traj: int, seed: int = 321, T: float = 2.0) -> np.ndarray:
    exact = build_exact_spin_chain_model(L=2, w=0.0, alpha=0.5)
    rng = np.random.default_rng(seed)
    return np.array([ordinary_quantum_jump_trajectory(exact, T, rng).n_jumps for _ in range(n_traj)], dtype=int)


def test_1_zeta_one_recovery() -> None:
    model = build_exact_spin_chain_model(L=4, w=0.0, alpha=0.5)
    consistency = exact_model_consistency(model)
    assert consistency["hamiltonian_hermiticity_error"] < 1e-12
    assert consistency["max_projector_error"] < 1e-12
    assert consistency["min_jump_sum_eig"] >= -1e-10
    assert consistency["max_jump_sum_eig"] <= model.L - 1 + 1e-10

    exact_small = build_exact_spin_chain_model(L=2, w=0.0, alpha=0.5)
    backward_exact = run_exact_backward_pass(exact_small, T=2.0, zeta=1.0)
    rng_doob = np.random.default_rng(7)
    rng_born = np.random.default_rng(7)
    doob = doob_exact_trajectory(exact_small, backward_exact, 2.0, 1.0, rng_doob)
    born = ordinary_quantum_jump_trajectory(exact_small, 2.0, rng_born)

    assert doob.channels == born.channels
    np.testing.assert_allclose(doob.jump_times, born.jump_times, atol=1e-10)

    gauss = build_gaussian_chain_model(L=4, w=0.5, alpha=0.5)
    backward_gauss = run_gaussian_backward_pass(gauss, T=2.0, zeta=1.0, sample_points=17)
    for t in (0.0, 1.0, 2.0):
        C_t, z_t = backward_gauss.state_at(t)
        np.testing.assert_allclose(C_t, 0.0, atol=1e-9)
        assert z_t == pytest.approx(1.0, abs=1e-9)


def test_2_zeta_to_zero_concentrates_no_click_sector() -> None:
    zeta = 0.01
    analytics = _single_mode_analytics(zeta)
    counts = _single_mode_algorithm_counts(zeta, n_traj=800)["doob"]
    mean_clicks = float(np.mean(counts))
    p0 = float(np.mean(counts == 0))

    assert mean_clicks < 0.05
    assert p0 > 0.95
    assert p0 == pytest.approx(analytics["Q_no_click"], abs=0.03)


def test_3_single_mode_exact_formulas() -> None:
    zeta = 0.5
    exact, gauss, backward_exact, backward_gauss = _single_mode_models(zeta)
    analytics = _single_mode_analytics(zeta)

    z_exact = backward_exact.overlap(0.0, exact.initial_state)
    C_0, z_scalar = backward_gauss.state_at(0.0)
    z_gauss = gaussian_overlap(C_0, gauss.gamma0, z_scalar=z_scalar)
    assert z_exact == pytest.approx(analytics["Z"], abs=1e-10)
    assert z_gauss == pytest.approx(analytics["Z"], abs=1e-8)

    counts = _single_mode_algorithm_counts(zeta, n_traj=1_000)
    p0_doob = float(np.mean(counts["doob"] == 0))
    p0_local = float(np.mean(counts["procedure_c"] == 0))

    assert p0_doob == pytest.approx(analytics["Q_no_click"], abs=0.03)
    assert p0_local == pytest.approx(analytics["R_no_click"], abs=0.04)
    assert abs(analytics["Q_no_click"] - analytics["R_no_click"]) > 0.05
    assert abs(p0_doob - p0_local) > 0.04


def test_4_commuting_case_backward_pass_and_rates() -> None:
    L = 4
    T = 2.0
    zeta = 0.5
    model = build_gaussian_chain_model(L=L, w=0.0, alpha=0.5)
    exact = build_exact_spin_chain_model(L=L, w=0.0, alpha=0.5)
    backward_exact = run_exact_backward_pass(exact, T=T, zeta=zeta)
    backward = run_gaussian_backward_pass(model, T=T, zeta=zeta, sample_points=17)
    C_0, z_0 = backward.state_at(0.0)
    sigma_expected = np.tanh(0.5 * (1.0 - zeta) * T)

    allowed = set()
    for pair in model.jump_pairs:
        a, b = pair
        allowed.add((a, b))
        allowed.add((b, a))
        assert C_0[a, b] == pytest.approx(sigma_expected, abs=5e-6)
        assert C_0[b, a] == pytest.approx(-sigma_expected, abs=5e-6)

    for m in range(C_0.shape[0]):
        for n in range(C_0.shape[1]):
            if m == n or (m, n) in allowed:
                continue
            assert abs(C_0[m, n]) < 1e-7

    gamma_state = model.gamma0
    overlap_pre = gaussian_overlap(C_0, gamma_state, z_scalar=z_0)
    G_exact = backward_exact.operator_at(0.0)
    psi0 = exact.initial_state
    exact_overlap_pre = float(np.real(np.vdot(psi0, G_exact @ psi0)))
    for pair, projector in zip(model.jump_pairs, exact.jump_projectors):
        q_j, gamma_post = apply_projective_jump(gamma_state, pair)
        overlap_post = gaussian_overlap(C_0, gamma_post, z_scalar=z_0)
        rate_gauss = zeta * 2.0 * model.alpha * q_j * overlap_post / overlap_pre
        exact_numerator = np.vdot(psi0, projector.toarray() @ G_exact @ projector.toarray() @ psi0)
        rate_exact = zeta * 2.0 * model.alpha * float(np.real(exact_numerator)) / exact_overlap_pre
        assert rate_gauss == pytest.approx(rate_exact, rel=1e-6, abs=1e-8)


def test_5_partition_function_and_moment_consistency() -> None:
    born_counts = _single_mode_born_counts(2_000)
    zetas = [0.9, 0.7, 0.5, 0.3]

    for zeta in zetas:
        _, gauss, _, backward = _single_mode_models(zeta)
        C_0, z_scalar = backward.state_at(0.0)
        z_doob = gaussian_overlap(C_0, gauss.gamma0, z_scalar=z_scalar)

        weights = zeta**born_counts
        z_weighted = float(np.mean(weights))
        q_mean_weighted = float(np.average(born_counts, weights=weights))
        doob_counts = _single_mode_algorithm_counts(zeta, n_traj=1_000)["doob"]
        doob_mean = float(np.mean(doob_counts))

        assert z_doob == pytest.approx(z_weighted, abs=0.03)
        assert doob_mean == pytest.approx(q_mean_weighted, abs=0.10)


def test_6_click_count_distribution_matches_weighted_born() -> None:
    zeta = 0.5
    born_counts = _single_mode_born_counts(2_000)
    weights = zeta**born_counts
    pmf_weighted = _histogram(born_counts, weights=weights)
    doob_counts = _single_mode_algorithm_counts(zeta, n_traj=1_000)["doob"]
    pmf_doob = _histogram(doob_counts)

    size = max(len(pmf_weighted), len(pmf_doob))
    padded_weighted = np.pad(pmf_weighted, (0, size - len(pmf_weighted)))
    padded_doob = np.pad(pmf_doob, (0, size - len(pmf_doob)))
    total_variation = 0.5 * np.abs(padded_weighted - padded_doob).sum()

    assert total_variation < 0.12


def test_7_entanglement_entropy_decreases_with_zeta() -> None:
    L = 4
    T = 2.0
    zetas = [1.0, 0.5, 0.1]
    means = []

    for zeta in zetas:
        model = build_gaussian_chain_model(L=L, w=0.0, alpha=0.5)
        backward = run_gaussian_backward_pass(model, T=T, zeta=zeta, sample_points=65)
        rng = np.random.default_rng(1234)
        entropies = []
        for _ in range(200):
            trajectory = doob_gaussian_trajectory(model, backward, T, zeta, rng)
            entropies.append(entanglement_entropy(covariance_from_orbitals(trajectory.final_state), L // 2))
        means.append(float(np.mean(entropies)))

    assert means[0] > means[1] > means[2]


def test_8_conditioned_survival_is_monotone() -> None:
    model = build_gaussian_chain_model(L=4, w=0.5, alpha=0.5)
    backward = run_gaussian_backward_pass(model, T=2.0, zeta=0.5, sample_points=65)
    values = [
        conditioned_survival_gaussian(model, backward, model.orbitals0, 0.0, float(dt))
        for dt in np.linspace(0.0, 2.0, 21)
    ]

    assert values[0] == pytest.approx(1.0, abs=1e-12)
    assert values[-1] < 1.0
    assert values[-1] > 0.0
    assert all(values[i] >= values[i + 1] - 1e-10 for i in range(len(values) - 1))


def test_9_procedure_c_differs_from_qs_but_doob_matches_ab() -> None:
    zeta = 0.5
    analytics = _single_mode_analytics(zeta)
    counts = _single_mode_algorithm_counts(zeta, n_traj=1_000)

    p0_doob = float(np.mean(counts["doob"] == 0))
    p0_a = float(np.mean(counts["procedure_a"] == 0))
    p0_b = float(np.mean(counts["procedure_b"] == 0))
    p0_c = float(np.mean(counts["procedure_c"] == 0))

    assert p0_doob == pytest.approx(analytics["Q_no_click"], abs=0.03)
    assert p0_a == pytest.approx(analytics["Q_no_click"], abs=0.03)
    assert p0_b == pytest.approx(analytics["Q_no_click"], abs=0.04)
    assert p0_c == pytest.approx(analytics["R_no_click"], abs=0.04)
    assert abs(p0_doob - p0_c) > 0.04

    mean_doob = float(np.mean(counts["doob"]))
    mean_a = float(np.mean(counts["procedure_a"]))
    mean_b = float(np.mean(counts["procedure_b"]))
    mean_c = float(np.mean(counts["procedure_c"]))

    assert abs(mean_doob - mean_a) < 0.08
    assert abs(mean_doob - mean_b) < 0.08
    assert abs(mean_doob - mean_c) > 0.04
