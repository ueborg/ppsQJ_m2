from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import kstest

from pps_qj.backward_pass import run_exact_backward_pass, run_gaussian_backward_pass
from pps_qj.core.numerics import safe_normalize
from pps_qj.doob_wtmc import doob_gaussian_trajectory
from pps_qj.exact_backend import (
    _propagate_unnormalized,
    _sample_channel,
    _sample_waiting_time,
    build_exact_spin_chain_model,
    ordinary_quantum_jump_trajectory,
)
from pps_qj.gaussian_backend import (
    apply_projective_jump,
    build_gaussian_chain_model,
    covariance_from_orbitals,
    entanglement_entropy,
    jump_probability,
    orbitals_from_covariance,
    propagate_no_click_orbitals,
)
from pps_qj.observables.basic import entanglement_entropy_statevector
from pps_qj.overlaps import exact_operator_overlap, gaussian_overlap
from pps_qj.part6_validation import (
    _counts_hist,
    _effective_sample_size,
    _run_loop,
    _save_figure,
    _total_variation,
    _weighted_mean_sem,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


@dataclass(frozen=True)
class ExtendedValidationConfig:
    output_dir: str = "outputs/extended_validation"
    seed: int = 20260330
    overlap_micro_n_states: int = 6
    test10_born_n: int = 5000
    test10_doob_n: int = 1000
    test11_born_n: int = 1000
    test11_grid_points: int = 41
    test12_survival_grid_points: int = 100
    test12_doob_n: int = 5000
    test13_born_n: int = 3000
    test14_search_n: int = 250
    test15_born_n: int = 2000
    test15_doob_n: int = 500
    test16_born_n: int = 500
    test16_doob_n: int = 200
    test17_born_n: int = 500
    test17_doob_n: int = 200
    test18_born_n: int = 2000
    test18_doob_n: int = 1000
    plotA_n_traj: int = 200
    plotA_zeta_points: int = 50
    plotB_n_traj: int = 120
    plotB_zeta_points: int = 20
    plotC_grid_points: int = 201
    plot_dpi: int = 160
    generate_report_plots: bool = True


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _weighted_array_mean_sem(samples: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    arr = np.asarray(samples, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(w))
    if total <= 0.0:
        return np.zeros(arr.shape[1:], dtype=np.float64), np.zeros(arr.shape[1:], dtype=np.float64), 0.0
    normalized = w / total
    mean = np.tensordot(normalized, arr, axes=(0, 0))
    centered = arr - mean
    var = np.tensordot(normalized, centered * centered, axes=(0, 0))
    ess = _effective_sample_size(w)
    sem = np.sqrt(np.maximum(var, 0.0) / max(ess, 1.0))
    return np.asarray(mean, dtype=np.float64), np.asarray(sem, dtype=np.float64), float(ess)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.size == 0:
        return np.zeros(len(quantiles), dtype=np.float64)
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted)
    cdf /= cdf[-1]
    return np.interp(np.asarray(quantiles, dtype=np.float64), cdf, x_sorted)


def _weighted_ecdf(values: np.ndarray, weights: np.ndarray, xs: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if vals.size == 0:
        return np.zeros_like(xs, dtype=np.float64)
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cdf = np.cumsum(w)
    total = float(cdf[-1])
    if total <= 0.0:
        return np.zeros_like(xs, dtype=np.float64)
    cdf = cdf / total
    idx = np.searchsorted(vals, xs, side="right") - 1
    out = np.zeros_like(xs, dtype=np.float64)
    mask = idx >= 0
    out[mask] = cdf[idx[mask]]
    return out


def _weighted_ks_statistic(
    x: np.ndarray,
    x_weights: np.ndarray,
    y: np.ndarray,
    y_weights: np.ndarray | None = None,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    wx = np.asarray(x_weights, dtype=np.float64)
    wy = np.ones_like(y, dtype=np.float64) if y_weights is None else np.asarray(y_weights, dtype=np.float64)
    support = np.unique(np.concatenate([x, y]))
    if support.size == 0:
        return 0.0
    cdf_x = _weighted_ecdf(x, wx, support)
    cdf_y = _weighted_ecdf(y, wy, support)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _majorana_ops(model) -> tuple[sparse.csr_matrix, ...]:
    ops: list[sparse.csr_matrix] = []
    for c_op, cd_op in zip(model.c_ops, model.cd_ops):
        ops.append((c_op + cd_op).tocsr())
        ops.append((1j * (cd_op - c_op)).tocsr())
    return tuple(ops)


def _covariance_from_exact_state(state: np.ndarray, gamma_ops: tuple[sparse.csr_matrix, ...]) -> np.ndarray:
    n = len(gamma_ops)
    transformed = [op @ state for op in gamma_ops]
    covariance = np.zeros((n, n), dtype=np.float64)
    for m in range(n):
        for n_idx in range(m + 1, n):
            value = np.vdot(transformed[m], transformed[n_idx])
            covariance[m, n_idx] = float(np.real_if_close(1j * value, tol=1_000.0).real)
            covariance[n_idx, m] = -covariance[m, n_idx]
    return covariance


def _number_ops(model) -> tuple[sparse.csr_matrix, ...]:
    return tuple((cd_op @ c_op).tocsr() for c_op, cd_op in zip(model.c_ops, model.cd_ops))


def _nearest_neighbor_number_ops(number_ops: tuple[sparse.csr_matrix, ...]) -> tuple[sparse.csr_matrix, ...]:
    return tuple((number_ops[idx] @ number_ops[idx + 1]).tocsr() for idx in range(len(number_ops) - 1))


def _occupations_from_exact_state(state: np.ndarray, number_ops: tuple[sparse.csr_matrix, ...]) -> np.ndarray:
    return np.asarray(
        [float(np.real(np.vdot(state, op @ state))) for op in number_ops],
        dtype=np.float64,
    )


def _nn_products_from_exact_state(state: np.ndarray, nn_ops: tuple[sparse.csr_matrix, ...]) -> np.ndarray:
    return np.asarray(
        [float(np.real(np.vdot(state, op @ state))) for op in nn_ops],
        dtype=np.float64,
    )


def _occupations_from_covariance(covariance: np.ndarray) -> np.ndarray:
    Gamma = np.asarray(covariance, dtype=np.float64)
    L = Gamma.shape[0] // 2
    occ = np.zeros(L, dtype=np.float64)
    for site in range(L):
        a = 2 * site
        b = a + 1
        occ[site] = 0.5 * (1.0 + Gamma[a, b])
    return occ


def _nn_products_from_covariance(covariance: np.ndarray) -> np.ndarray:
    Gamma = np.asarray(covariance, dtype=np.float64)
    G = np.eye(Gamma.shape[0], dtype=np.complex128) - 1j * Gamma
    L = Gamma.shape[0] // 2
    out = np.zeros(L - 1, dtype=np.float64)
    for site in range(L - 1):
        a = 2 * site
        b = a + 1
        c = 2 * (site + 1)
        d = c + 1
        e4 = G[a, b] * G[c, d] - G[a, c] * G[b, d] + G[a, d] * G[b, c]
        value = 0.25 * (1.0 + Gamma[a, b] + Gamma[c, d] - e4)
        out[site] = float(np.real_if_close(value, tol=1_000.0).real)
    return out


def _entropy_from_covariance(covariance: np.ndarray, cut: int) -> float:
    return float(entanglement_entropy(np.asarray(covariance, dtype=np.float64), cut))


def _observe_exact_born_trajectory(
    model,
    T: float,
    rng: np.random.Generator,
    *,
    observe_times: np.ndarray | None = None,
    backward_data=None,
    zeta: float | None = None,
    entropy_cut: int | None = None,
    number_ops: tuple[sparse.csr_matrix, ...] | None = None,
    nn_ops: tuple[sparse.csr_matrix, ...] | None = None,
    gamma_ops: tuple[sparse.csr_matrix, ...] | None = None,
    record_covariances: bool = False,
    record_compensator: bool = False,
) -> dict[str, Any]:
    observe = np.asarray(observe_times if observe_times is not None else [], dtype=np.float64)
    n_obs = observe.size
    entropies = np.zeros(n_obs, dtype=np.float64) if entropy_cut is not None else None
    occupations = np.zeros((n_obs, model.L), dtype=np.float64) if number_ops is not None else None
    nn_products = np.zeros((n_obs, model.L - 1), dtype=np.float64) if nn_ops is not None else None
    covariances = np.zeros((n_obs, 2 * model.L, 2 * model.L), dtype=np.float64) if record_covariances else None
    counts_t = np.zeros(n_obs, dtype=int) if (backward_data is not None or record_compensator) else None
    overlaps_t = np.zeros(n_obs, dtype=np.float64) if backward_data is not None else None
    martingale_t = np.zeros(n_obs, dtype=np.float64) if backward_data is not None and zeta is not None else None
    compensator_t = np.zeros(n_obs, dtype=np.float64) if record_compensator else None

    state = np.asarray(model.initial_state, dtype=np.complex128).copy()
    t = 0.0
    obs_idx = 0
    jump_times: list[float] = []
    channels: list[int] = []
    lambda_total = 0.0

    def record_state(psi_obs: np.ndarray, idx: int, count_so_far: int, lambda_before: float, delta: float) -> None:
        if entropies is not None:
            entropies[idx] = entanglement_entropy_statevector(psi_obs, model.L, entropy_cut)
        if occupations is not None and number_ops is not None:
            occupations[idx] = _occupations_from_exact_state(psi_obs, number_ops)
        if nn_products is not None and nn_ops is not None:
            nn_products[idx] = _nn_products_from_exact_state(psi_obs, nn_ops)
        if covariances is not None and gamma_ops is not None:
            covariances[idx] = _covariance_from_exact_state(psi_obs, gamma_ops)
        if counts_t is not None:
            counts_t[idx] = count_so_far
        if record_compensator and compensator_t is not None:
            psi_tilde = _propagate_unnormalized(model, state, delta)
            survival = max(float(np.real(np.vdot(psi_tilde, psi_tilde))), 1e-300)
            compensator_t[idx] = lambda_before - np.log(survival)
        if backward_data is not None and overlaps_t is not None:
            overlap = backward_data.overlap(float(observe[idx]), psi_obs)
            overlaps_t[idx] = overlap
            if martingale_t is not None and zeta is not None:
                martingale_t[idx] = overlap * (zeta**count_so_far)

    while t < T:
        dt = _sample_waiting_time(model, state, rng, T_horizon=T - t)
        segment_end = T if not np.isfinite(dt) or t + dt >= T else t + dt
        while obs_idx < n_obs and observe[obs_idx] <= segment_end + 1e-12:
            delta = float(observe[obs_idx] - t)
            psi_obs = safe_normalize(_propagate_unnormalized(model, state, delta))
            record_state(psi_obs, obs_idx, len(channels), lambda_total, delta)
            obs_idx += 1

        delta_segment = T - t if segment_end >= T else float(dt)
        psi_tilde = _propagate_unnormalized(model, state, delta_segment)
        segment_survival = max(float(np.real(np.vdot(psi_tilde, psi_tilde))), 1e-300)
        if record_compensator:
            lambda_total += -np.log(segment_survival)

        if segment_end >= T:
            state = safe_normalize(psi_tilde)
            t = T
            break

        state = safe_normalize(psi_tilde)
        t += float(dt)
        channel = _sample_channel(model, state, rng)
        state = safe_normalize(model.jump_projectors[channel] @ state)
        jump_times.append(t)
        channels.append(channel)

    while obs_idx < n_obs:
        record_state(state, obs_idx, len(channels), lambda_total, 0.0)
        obs_idx += 1

    return {
        "n_jumps": len(channels),
        "jump_times": jump_times,
        "channels": channels,
        "final_state": state,
        "entropies": entropies,
        "occupations": occupations,
        "nn_products": nn_products,
        "covariances": covariances,
        "counts_t": counts_t,
        "overlaps_t": overlaps_t,
        "martingale_t": martingale_t,
        "compensator_t": compensator_t,
        "lambda_total": float(lambda_total),
    }


def _observe_gaussian_doob_trajectory(
    model,
    backward,
    T: float,
    zeta: float,
    rng: np.random.Generator,
    *,
    observe_times: np.ndarray | None = None,
    entropy_cut: int | None = None,
    record_occupations: bool = False,
    record_nn_products: bool = False,
    record_covariances: bool = False,
    survival_grid_points: int = 0,
) -> dict[str, Any]:
    observe = np.asarray(observe_times if observe_times is not None else [], dtype=np.float64)
    n_obs = observe.size
    entropies = np.zeros(n_obs, dtype=np.float64) if entropy_cut is not None else None
    occupations = np.zeros((n_obs, model.L), dtype=np.float64) if record_occupations else None
    nn_products = np.zeros((n_obs, model.L - 1), dtype=np.float64) if record_nn_products else None
    covariances = np.zeros((n_obs, 2 * model.L, 2 * model.L), dtype=np.float64) if record_covariances else None

    orbitals = np.asarray(model.orbitals0, dtype=np.complex128).copy()
    t = 0.0
    obs_idx = 0
    jump_times: list[float] = []
    channels: list[int] = []
    segments: list[dict[str, Any]] = []

    def record_covariance(covariance: np.ndarray, idx: int) -> None:
        if entropies is not None:
            entropies[idx] = _entropy_from_covariance(covariance, entropy_cut)
        if occupations is not None:
            occupations[idx] = _occupations_from_covariance(covariance)
        if nn_products is not None:
            nn_products[idx] = _nn_products_from_covariance(covariance)
        if covariances is not None:
            covariances[idx] = np.asarray(covariance, dtype=np.float64)

    while t < T:
        gamma_now = covariance_from_orbitals(orbitals)
        C_now, z_now = backward.state_at(t)
        denominator = gaussian_overlap(C_now, gamma_now, z_scalar=z_now)
        r = float(rng.uniform(0.0, 1.0))
        max_dt = T - t

        def survival_fn(dt: float) -> float:
            evolution = propagate_no_click_orbitals(
                orbitals,
                model.h_effective,
                dt,
                gamma_m=model.gamma_m,
                n_monitored=len(model.jump_pairs),
            )
            C_t, z_t = backward.state_at(t + dt)
            numerator = evolution.branch_norm * gaussian_overlap(C_t, evolution.covariance, z_scalar=z_t)
            return float(numerator / denominator)

        segment = {
            "t_start": float(t),
            "denominator": float(denominator),
            "uniform_threshold": float(r),
            "max_dt": float(max_dt),
        }
        if survival_grid_points > 1:
            grid = np.linspace(0.0, max_dt, survival_grid_points)
            segment["times"] = list(t + grid)
            segment["values"] = [survival_fn(float(dt)) for dt in grid]

        if survival_fn(max_dt) > r:
            segment["realized_dt"] = float(max_dt)
            segment["realized_survival"] = float(survival_fn(max_dt))
            segment["jumped"] = False
            segments.append(segment)
            segment_end = T
            jump_dt = None
        else:
            left = 0.0
            right = max_dt
            for _ in range(30):
                mid = 0.5 * (left + right)
                if survival_fn(mid) > r:
                    left = mid
                else:
                    right = mid
            jump_dt = 0.5 * (left + right)
            segment["realized_dt"] = float(jump_dt)
            segment["realized_survival"] = float(survival_fn(jump_dt))
            segment["jumped"] = True
            segments.append(segment)
            segment_end = t + jump_dt

        while obs_idx < n_obs and observe[obs_idx] <= segment_end + 1e-12:
            evolution_obs = propagate_no_click_orbitals(
                orbitals,
                model.h_effective,
                float(observe[obs_idx] - t),
                gamma_m=model.gamma_m,
                n_monitored=len(model.jump_pairs),
            )
            record_covariance(evolution_obs.covariance, obs_idx)
            obs_idx += 1

        if jump_dt is None:
            evolution = propagate_no_click_orbitals(
                orbitals,
                model.h_effective,
                max_dt,
                gamma_m=model.gamma_m,
                n_monitored=len(model.jump_pairs),
            )
            orbitals = evolution.orbitals_normalized
            t = T
            break

        evolution = propagate_no_click_orbitals(
            orbitals,
            model.h_effective,
            jump_dt,
            gamma_m=model.gamma_m,
            n_monitored=len(model.jump_pairs),
        )
        pre_covariance = evolution.covariance
        pre_orbitals = evolution.orbitals_normalized
        t += jump_dt

        C_jump, z_jump = backward.state_at(t)
        overlap_pre = gaussian_overlap(C_jump, pre_covariance, z_scalar=z_jump)
        rates = np.zeros(len(model.jump_pairs), dtype=np.float64)
        post_covariances: list[np.ndarray] = []
        for idx, jump_pair in enumerate(model.jump_pairs):
            q, post_covariance = apply_projective_jump(pre_covariance, jump_pair)
            overlap_post = gaussian_overlap(C_jump, post_covariance, z_scalar=z_jump)
            rates[idx] = zeta * model.gamma_m * q * overlap_post / overlap_pre
            post_covariances.append(post_covariance)

        if np.sum(rates) <= 0.0:
            orbitals = pre_orbitals
            continue

        channel = int(rng.choice(len(rates), p=rates / np.sum(rates)))
        orbitals = orbitals_from_covariance(post_covariances[channel])
        jump_times.append(t)
        channels.append(channel)

    if obs_idx < n_obs:
        final_covariance = covariance_from_orbitals(orbitals)
        while obs_idx < n_obs:
            record_covariance(final_covariance, obs_idx)
            obs_idx += 1

    return {
        "n_jumps": len(channels),
        "jump_times": jump_times,
        "channels": channels,
        "final_state": orbitals,
        "entropies": entropies,
        "occupations": occupations,
        "nn_products": nn_products,
        "covariances": covariances,
        "segments": segments,
    }


def _run_overlap_microtest(config: ExtendedValidationConfig) -> dict[str, Any]:
    errors: list[float] = []
    examples: list[dict[str, float]] = []
    for L in (2, 4):
        operator_exact = build_exact_spin_chain_model(L=L, w=0.0, gamma_m=1.0)
        operator_gauss = build_gaussian_chain_model(L=L, w=0.0, gamma_m=1.0)
        backward_exact = run_exact_backward_pass(operator_exact, T=2.0, zeta=0.5)
        backward_gauss = run_gaussian_backward_pass(operator_gauss, T=2.0, zeta=0.5, sample_points=65)
        state_model = build_exact_spin_chain_model(L=L, w=0.5, gamma_m=1.0)
        gamma_ops = _majorana_ops(state_model)
        rng = _rng(config.seed + 10 * L)
        for idx in range(config.overlap_micro_n_states):
            trajectory = ordinary_quantum_jump_trajectory(state_model, 1.5, rng)
            psi = np.asarray(trajectory.final_state, dtype=np.complex128)
            gamma = _covariance_from_exact_state(psi, gamma_ops)
            t = float(rng.choice(np.array([0.0, 0.4, 0.8, 1.2, 1.6], dtype=np.float64)))
            exact_value = backward_exact.overlap(t, psi)
            C_t, z_t = backward_gauss.state_at(t)
            gaussian_value = gaussian_overlap(C_t, gamma, z_scalar=z_t)
            rel_error = abs(gaussian_value - exact_value) / max(abs(exact_value), 1e-12)
            errors.append(rel_error)
            if len(examples) < 6:
                examples.append(
                    {
                        "L": float(L),
                        "t": t,
                        "exact": float(exact_value),
                        "gaussian": float(gaussian_value),
                        "relative_error": float(rel_error),
                    }
                )
    return {
        "passed": bool(max(errors, default=0.0) < 1e-8),
        "parameters": {"L_values": [2, 4], "n_states_per_L": config.overlap_micro_n_states},
        "metrics": {
            "max_relative_error": float(max(errors, default=0.0)),
            "mean_relative_error": float(np.mean(errors)) if errors else 0.0,
            "examples": examples,
        },
    }


def _run_test_10(config: ExtendedValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = 4
    w = 0.5
    gamma_m = 1.0
    T = 2.0
    zeta = 0.5

    exact = build_exact_spin_chain_model(L=L, w=w, gamma_m=gamma_m)
    gauss = build_gaussian_chain_model(L=L, w=w, gamma_m=gamma_m)
    backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)

    rng_born = _rng(config.seed + 1001)
    born_records = _run_loop(
        "Test 10 Born",
        config.test10_born_n,
        lambda: _observe_exact_born_trajectory(exact, T, rng_born, record_compensator=True),
    )
    counts = np.asarray([rec["n_jumps"] for rec in born_records], dtype=int)
    lambdas = np.asarray([rec["lambda_total"] for rec in born_records], dtype=np.float64)

    w_q_raw = zeta**counts
    z_hat = float(np.mean(w_q_raw))
    w_q = w_q_raw / max(np.sum(w_q_raw), 1e-300)

    w_r_raw = w_q_raw * np.exp((1.0 - zeta) * lambdas)
    z_r_hat = float(np.mean(w_r_raw))
    w_r = w_r_raw / max(np.sum(w_r_raw), 1e-300)

    log_ratio = -(1.0 - zeta) * lambdas - np.log(max(z_hat, 1e-300)) + np.log(max(z_r_hat, 1e-300))

    eq_n = float(np.sum(w_q * counts))
    er_n = float(np.sum(w_r * counts))
    eq_lambda = float(np.sum(w_q * lambdas))
    er_lambda = float(np.sum(w_r * lambdas))
    eq_n_sem = _weighted_mean_sem(counts.astype(np.float64), w_q_raw)[1]
    er_n_sem = _weighted_mean_sem(counts.astype(np.float64), w_r_raw)[1]

    rng_doob = _rng(config.seed + 1002)
    doob_counts = np.asarray(
        _run_loop("Test 10 Doob", config.test10_doob_n, lambda: doob_gaussian_trajectory(gauss, backward, T, zeta, rng_doob).n_jumps),
        dtype=int,
    )
    doob_mean = float(np.mean(doob_counts))
    doob_sem = float(np.std(doob_counts, ddof=1) / np.sqrt(len(doob_counts))) if len(doob_counts) > 1 else 0.0

    fig1, ax1 = plt.subplots(figsize=(6.8, 4.8))
    scatter = ax1.scatter(lambdas, w_q_raw, c=counts, cmap="viridis", s=12, alpha=0.65)
    ax1.set_xlabel(r"$\Lambda_T$")
    ax1.set_ylabel(r"$\zeta^{N_T}$")
    ax1.set_title("Test 10: Born trajectories in $(\\Lambda_T, \\zeta^{N_T})$")
    cbar = fig1.colorbar(scatter, ax=ax1)
    cbar.set_label(r"$N_T$")
    scatter_path = output_dir / "test10_lambda_vs_zeta_power.png"
    _save_figure(fig1, scatter_path, config.plot_dpi)

    fig2, ax2 = plt.subplots(figsize=(6.8, 4.6))
    ax2.hist(log_ratio, bins=40, density=True, alpha=0.85, color="#3d5a80")
    ax2.set_xlabel(r"$\log(dQ_s/dR_\zeta)$")
    ax2.set_ylabel("Density")
    ax2.set_title("Test 10: Non-degenerate $Q_s / R_\\zeta$ ratio")
    hist_path = output_dir / "test10_log_q_over_r_hist.png"
    _save_figure(fig2, hist_path, config.plot_dpi)

    passed = bool(
        np.std(log_ratio) > 1e-3
        and abs(eq_n - doob_mean) < 3.0 * np.sqrt(eq_n_sem**2 + doob_sem**2) + 0.02
        and (abs(eq_n - er_n) > 0.02 or abs(eq_lambda - er_lambda) > 0.02)
    )
    return {
        "passed": passed,
        "parameters": {"L": L, "w": w, "gamma_m": gamma_m, "T": T, "zeta": zeta},
        "metrics": {
            "Z_qs_hat": z_hat,
            "Z_rzeta_hat": z_r_hat,
            "E_Qs_NT": eq_n,
            "E_Rzeta_NT": er_n,
            "E_Qs_LambdaT": eq_lambda,
            "E_Rzeta_LambdaT": er_lambda,
            "doob_mean_clicks": doob_mean,
            "qs_weight_ess": _effective_sample_size(w_q_raw),
            "rzeta_weight_ess": _effective_sample_size(w_r_raw),
            "log_ratio_std": float(np.std(log_ratio)),
        },
        "artifacts": {
            "scatter_plot": str(scatter_path),
            "ratio_histogram": str(hist_path),
        },
    }


def _run_test_11(config: ExtendedValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = 4
    w = 0.5
    gamma_m = 1.0
    T = 2.0
    zeta = 0.5
    observe_times = np.linspace(0.0, T, config.test11_grid_points)

    exact = build_exact_spin_chain_model(L=L, w=w, gamma_m=gamma_m)
    backward = run_exact_backward_pass(exact, T=T, zeta=zeta)

    rng = _rng(config.seed + 1101)
    records = _run_loop(
        "Test 11 Born",
        config.test11_born_n,
        lambda: _observe_exact_born_trajectory(
            exact,
            T,
            rng,
            observe_times=observe_times,
            backward_data=backward,
            zeta=zeta,
        ),
    )
    martingale = np.asarray([rec["martingale_t"] for rec in records], dtype=np.float64)
    mean_m = np.mean(martingale, axis=0)
    p10 = np.percentile(martingale, 10.0, axis=0)
    p90 = np.percentile(martingale, 90.0, axis=0)
    final_values = martingale[:, -1]
    z_exact = backward.overlap(0.0, exact.initial_state)
    final_sem = float(np.std(final_values, ddof=1) / np.sqrt(len(final_values))) if len(final_values) > 1 else 0.0

    idx1 = observe_times.size // 2
    idx2 = -1
    m1 = martingale[:, idx1]
    m2 = martingale[:, idx2]
    edges = np.quantile(m1, np.linspace(0.0, 1.0, 11))
    bin_errors: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (m1 >= left) & (m1 <= right if right == edges[-1] else m1 < right)
        if np.count_nonzero(mask) < 10:
            continue
        mean1 = float(np.mean(m1[mask]))
        mean2 = float(np.mean(m2[mask]))
        sem1 = float(np.std(m1[mask], ddof=1) / np.sqrt(np.count_nonzero(mask)))
        sem2 = float(np.std(m2[mask], ddof=1) / np.sqrt(np.count_nonzero(mask)))
        bin_errors.append(abs(mean2 - mean1) / (3.0 * max(np.sqrt(sem1**2 + sem2**2), 1e-12)))

    fig1, ax1 = plt.subplots(figsize=(6.8, 4.6))
    ax1.scatter(m1, m2, s=10, alpha=0.55, color="#006d77")
    lo = float(min(np.min(m1), np.min(m2)))
    hi = float(max(np.max(m1), np.max(m2)))
    ax1.plot([lo, hi], [lo, hi], color="black", linestyle="--", lw=1.0)
    ax1.set_xlabel(r"$M_{t_1}$")
    ax1.set_ylabel(r"$M_{t_2}$")
    ax1.set_title("Test 11: Martingale scatter at $t_1=T/2$, $t_2=T$")
    scatter_path = output_dir / "test11_martingale_scatter.png"
    _save_figure(fig1, scatter_path, config.plot_dpi)

    fig2, ax2 = plt.subplots(figsize=(7.0, 4.6))
    ax2.plot(observe_times, mean_m, lw=1.8, label=r"$E_P[M_t]$")
    ax2.fill_between(observe_times, p10, p90, alpha=0.25, label="10-90 percentile band")
    ax2.axhline(z_exact, color="black", linestyle="--", lw=1.0, label=r"$Z_\zeta$")
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$M_t$")
    ax2.set_title("Test 11: Martingale time series")
    ax2.legend(fontsize=9)
    timeseries_path = output_dir / "test11_martingale_timeseries.png"
    _save_figure(fig2, timeseries_path, config.plot_dpi)

    passed = bool(
        abs(float(np.mean(final_values)) - z_exact) < 3.0 * final_sem + 0.01
        and np.max(np.abs(mean_m - z_exact)) < 3.0 * final_sem + 0.02
        and max(bin_errors, default=0.0) < 1.0
    )
    return {
        "passed": passed,
        "parameters": {"L": L, "w": w, "gamma_m": gamma_m, "T": T, "zeta": zeta, "n_traj": config.test11_born_n},
        "metrics": {
            "Z_exact": z_exact,
            "mean_M_T": float(np.mean(final_values)),
            "sem_M_T": final_sem,
            "max_abs_mean_time_series_error": float(np.max(np.abs(mean_m - z_exact))),
            "max_binned_mean_error_in_3sigma_units": float(max(bin_errors, default=0.0)),
        },
        "artifacts": {
            "scatter_plot": str(scatter_path),
            "timeseries_plot": str(timeseries_path),
        },
    }


def _run_test_12(config: ExtendedValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = 4
    w = 0.5
    gamma_m = 1.0
    T = 2.0
    zeta = 0.5

    gauss = build_gaussian_chain_model(L=L, w=w, gamma_m=gamma_m)
    backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)
    single = doob_gaussian_trajectory(
        gauss,
        backward,
        T,
        zeta,
        _rng(config.seed + 1201),
        survival_grid_points=config.test12_survival_grid_points,
    )
    segments = list(single.diagnostics.get("conditioned_survival_segments", []))

    s0_errors: list[float] = []
    max_positive_increment = 0.0
    endpoint_values: list[float] = []
    for segment in segments:
        values = np.asarray(segment.get("values", []), dtype=np.float64)
        if values.size == 0:
            continue
        s0_errors.append(abs(values[0] - 1.0))
        diffs = np.diff(values)
        if diffs.size:
            max_positive_increment = max(max_positive_increment, float(np.max(diffs)))
        endpoint_values.append(float(values[-1]))

    rng = _rng(config.seed + 1202)
    transformed: list[float] = []
    overlay_curves: list[dict[str, Any]] = []
    for idx in range(config.test12_doob_n):
        trajectory = doob_gaussian_trajectory(gauss, backward, T, zeta, rng)
        for segment in trajectory.diagnostics.get("conditioned_survival_segments", []):
            if bool(segment.get("jumped", False)):
                s_end = float(segment.get("terminal_survival", 0.0))
                denom = max(1.0 - s_end, 1e-12)
                transformed.append((1.0 - float(segment["realized_survival"])) / denom)
        if idx < 8:
            rich = doob_gaussian_trajectory(
                gauss,
                backward,
                T,
                zeta,
                _rng(config.seed + 1250 + idx),
                survival_grid_points=config.test12_survival_grid_points,
            )
            overlay_curves.extend(rich.diagnostics.get("conditioned_survival_segments", []))

    transformed_array = np.asarray(transformed, dtype=np.float64)
    ks_stat = float(kstest(transformed_array, "uniform").statistic) if transformed_array.size else 1.0

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for segment in overlay_curves[:10]:
        times = np.asarray(segment.get("times", []), dtype=np.float64)
        values = np.asarray(segment.get("values", []), dtype=np.float64)
        if values.size == 0:
            continue
        color = plt.cm.viridis(np.clip(segment["denominator"], 0.0, 1.0))
        ax.plot(times - float(segment["t_start"]), values, color=color, alpha=0.85, lw=1.2)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$S^D(\tau)$")
    ax.set_title("Test 12: Doob survivor functions across segments")
    overlay_path = output_dir / "test12_survival_overlay.png"
    _save_figure(fig, overlay_path, config.plot_dpi)

    interior_endpoint_values = [value for idx, value in enumerate(endpoint_values) if idx < max(len(endpoint_values) - 1, 0)]
    passed = bool(
        max(s0_errors, default=0.0) < 1e-12
        and max_positive_increment <= 1e-10
        and all(0.0 < value < 1.0 for value in interior_endpoint_values)
        and ks_stat < 0.05
    )
    return {
        "passed": passed,
        "parameters": {"L": L, "w": w, "gamma_m": gamma_m, "T": T, "zeta": zeta},
        "metrics": {
            "n_segments_single_traj": len(segments),
            "max_s0_error": float(max(s0_errors, default=0.0)),
            "max_positive_increment": float(max_positive_increment),
            "min_endpoint_value": float(min(endpoint_values, default=0.0)),
            "max_endpoint_value": float(max(endpoint_values, default=0.0)),
            "ks_statistic_uniform_transform": ks_stat,
            "transform_interpreted_as": "(1 - S_realized) / (1 - S_terminal)",
            "n_realized_waits": int(transformed_array.size),
        },
        "artifacts": {"survival_overlay_plot": str(overlay_path)},
    }


def _run_test_14(config: ExtendedValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = 4
    w = 0.5
    gamma_m = 1.0
    T = 2.0
    zeta = 0.5
    t_mid = T / 2.0

    exact = build_exact_spin_chain_model(L=L, w=w, gamma_m=gamma_m)
    gauss = build_gaussian_chain_model(L=L, w=w, gamma_m=gamma_m)
    backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)
    gamma_ops = _majorana_ops(exact)

    rng = _rng(config.seed + 1401)
    states: list[dict[str, Any]] = []
    for _ in range(config.test14_search_n):
        record = _observe_exact_born_trajectory(
            exact,
            T,
            rng,
            observe_times=np.array([t_mid]),
            gamma_ops=gamma_ops,
            record_covariances=True,
        )
        covariance = record["covariances"][0]
        q_values = np.asarray([jump_probability(covariance, pair) for pair in gauss.jump_pairs], dtype=np.float64)
        states.append({"covariance": covariance, "q": q_values})

    best_pair: tuple[int, int] | None = None
    best_distance = -1.0
    best_q_mismatch = np.inf
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            q_mismatch = float(np.max(np.abs(states[i]["q"] - states[j]["q"])))
            if q_mismatch >= 0.05:
                continue
            distance = float(np.linalg.norm(states[i]["covariance"] - states[j]["covariance"], ord="fro"))
            if distance > best_distance:
                best_pair = (i, j)
                best_distance = distance
                best_q_mismatch = q_mismatch
    if best_pair is None:
        best_pair = (0, 1)
        best_q_mismatch = float(np.max(np.abs(states[0]["q"] - states[1]["q"])))

    state1 = states[best_pair[0]]
    state2 = states[best_pair[1]]
    C_mid, z_mid = backward.state_at(t_mid)

    def doob_rates(covariance: np.ndarray) -> np.ndarray:
        overlap_pre = gaussian_overlap(C_mid, covariance, z_scalar=z_mid)
        rates = np.zeros(len(gauss.jump_pairs), dtype=np.float64)
        for idx, pair in enumerate(gauss.jump_pairs):
            q_val, post_covariance = apply_projective_jump(covariance, pair)
            overlap_post = gaussian_overlap(C_mid, post_covariance, z_scalar=z_mid)
            rates[idx] = zeta * gamma_m * q_val * overlap_post / overlap_pre
        return rates

    r_rates_1 = zeta * gamma_m * state1["q"]
    r_rates_2 = zeta * gamma_m * state2["q"]
    d_rates_1 = doob_rates(state1["covariance"])
    d_rates_2 = doob_rates(state2["covariance"])

    xs = np.arange(len(gauss.jump_pairs))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(xs - 1.5 * width, r_rates_1, width=width, label=r"$R_\zeta$, state 1")
    ax.bar(xs - 0.5 * width, r_rates_2, width=width, label=r"$R_\zeta$, state 2")
    ax.bar(xs + 0.5 * width, d_rates_1, width=width, label="Doob, state 1")
    ax.bar(xs + 1.5 * width, d_rates_2, width=width, label="Doob, state 2")
    ax.set_xlabel("Bond index")
    ax.set_ylabel("Jump rate")
    ax.set_title("Test 14: Look-ahead distinguishes states with matched $q_j$")
    ax.legend(fontsize=8)
    plot_path = output_dir / "test14_lookahead_rates.png"
    _save_figure(fig, plot_path, config.plot_dpi)

    max_r_diff = float(np.max(np.abs(r_rates_1 - r_rates_2)))
    max_d_diff = float(np.max(np.abs(d_rates_1 - d_rates_2)))
    passed = bool(best_q_mismatch < 0.05 and max_r_diff < 0.03 and max_d_diff > 0.01)
    return {
        "passed": passed,
        "parameters": {"L": L, "w": w, "gamma_m": gamma_m, "T": T, "zeta": zeta, "t_mid": t_mid},
        "metrics": {
            "selected_pair_q_mismatch": best_q_mismatch,
            "selected_pair_covariance_distance": best_distance,
            "max_rzeta_rate_difference": max_r_diff,
            "max_doob_rate_difference": max_d_diff,
            "rzeta_rates_state1": r_rates_1.tolist(),
            "rzeta_rates_state2": r_rates_2.tolist(),
            "doob_rates_state1": d_rates_1.tolist(),
            "doob_rates_state2": d_rates_2.tolist(),
        },
        "artifacts": {"lookahead_rate_plot": str(plot_path)},
    }


def _run_test_13(config: ExtendedValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = 6
    gamma_m = 1.0
    T = 2.0
    zeta = 0.5
    w_ref = 0.5
    dts = [0.1, 0.05, 0.02, 0.01]

    exact = build_exact_spin_chain_model(L=L, w=w_ref, gamma_m=gamma_m)
    gauss = build_gaussian_chain_model(L=L, w=w_ref, gamma_m=gamma_m)
    backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=257)
    C_T, z_T = backward.state_at(T)
    C_0, z_0 = backward.state_at(0.0)
    z_overlap = gaussian_overlap(C_0, gauss.gamma0, z_scalar=z_0)

    rng_born = _rng(config.seed + 1301)
    born_counts = np.asarray(
        _run_loop("Test 13 Born", config.test13_born_n, lambda: ordinary_quantum_jump_trajectory(exact, T, rng_born).n_jumps),
        dtype=int,
    )
    born_weights = zeta**born_counts
    z_born = float(np.mean(born_weights))
    z_born_sem = float(np.std(born_weights, ddof=1) / np.sqrt(len(born_weights))) if len(born_weights) > 1 else 0.0

    convergence_errors: dict[str, float] = {}
    for dt in dts:
        backward_dt = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=257, max_step=dt)
        C_dt, z_dt = backward_dt.state_at(0.0)
        convergence_errors[str(dt)] = abs(gaussian_overlap(C_dt, gauss.gamma0, z_scalar=z_dt) - z_born)

    eig_min = 1.0
    eig_max = -1.0
    antisym_error = 0.0
    for covariance in backward.sample_covariances:
        spectrum = np.linalg.eigvalsh(1j * covariance)
        eig_min = min(eig_min, float(np.min(spectrum.real)))
        eig_max = max(eig_max, float(np.max(spectrum.real)))
        antisym_error = max(antisym_error, float(np.max(np.abs(covariance + covariance.T))))

    gauss_comm = build_gaussian_chain_model(L=L, w=0.0, gamma_m=gamma_m)
    backward_comm = run_gaussian_backward_pass(gauss_comm, T=T, zeta=zeta, sample_points=257)
    t_comm = T - backward_comm.sample_tau
    order_comm = np.argsort(t_comm)
    t_comm = t_comm[order_comm]
    cov_comm = backward_comm.sample_covariances[order_comm]
    alpha_errors: list[float] = []
    offpair_max = np.zeros(t_comm.size, dtype=np.float64)
    analytic_curve = np.tanh(0.5 * gamma_m * (1.0 - zeta) * (T - t_comm))
    for idx, covariance in enumerate(cov_comm):
        allowed = set()
        for pair in gauss_comm.jump_pairs:
            a, b = pair
            allowed.add((a, b))
            allowed.add((b, a))
            alpha_errors.append(abs(covariance[a, b] - analytic_curve[idx]))
            alpha_errors.append(abs(covariance[b, a] + analytic_curve[idx]))
        mask = np.ones_like(covariance, dtype=bool)
        for m in range(covariance.shape[0]):
            mask[m, m] = False
        for a, b in allowed:
            mask[a, b] = False
        offpair_max[idx] = float(np.max(np.abs(covariance[mask]))) if np.any(mask) else 0.0

    fig1, ax1 = plt.subplots(figsize=(7.0, 4.6))
    for bond, (a, b) in enumerate(gauss_comm.jump_pairs):
        ax1.plot(t_comm, cov_comm[:, a, b], lw=1.4, label=f"bond {bond + 1}")
    ax1.plot(t_comm, analytic_curve, color="black", linestyle="--", lw=1.4, label="analytic")
    ax1.plot(t_comm, offpair_max, color="#c1121f", linestyle=":", lw=1.2, label="max off-pair |C|")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Backward covariance entry")
    ax1.set_title("Test 13: Commuting backward covariance")
    ax1.legend(fontsize=8, ncol=2)
    comm_path = output_dir / "test13_commuting_backward_covariance.png"
    _save_figure(fig1, comm_path, config.plot_dpi)

    gauss_noncomm = build_gaussian_chain_model(L=L, w=w_ref, gamma_m=gamma_m)
    backward_noncomm = run_gaussian_backward_pass(gauss_noncomm, T=T, zeta=zeta, sample_points=257)
    t_noncomm = T - backward_noncomm.sample_tau
    order_noncomm = np.argsort(t_noncomm)
    t_noncomm = t_noncomm[order_noncomm]
    cov_noncomm = backward_noncomm.sample_covariances[order_noncomm]
    sampled_offpairs = [(0, 5), (1, 6), (2, 7)]
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.6))
    for bond, (a, b) in enumerate(gauss_noncomm.jump_pairs[:5]):
        ax2.plot(t_noncomm, cov_noncomm[:, a, b], lw=1.2, label=f"C[{a},{b}]")
    for m, n in sampled_offpairs:
        ax2.plot(t_noncomm, cov_noncomm[:, m, n], linestyle="--", lw=1.0, label=f"C[{m},{n}]")
    ax2.set_xlabel("t")
    ax2.set_ylabel("Covariance entry")
    ax2.set_title("Test 13: Non-commuting backward covariance")
    ax2.legend(fontsize=7, ncol=2)
    noncomm_path = output_dir / "test13_noncommuting_backward_covariance.png"
    _save_figure(fig2, noncomm_path, config.plot_dpi)

    errors_list = [convergence_errors[str(dt)] for dt in dts]
    monotone_convergence = all(errors_list[idx] >= errors_list[idx + 1] - 1e-12 for idx in range(len(errors_list) - 1))
    passed = bool(
        float(np.max(np.abs(C_T))) < 1e-12
        and abs(z_T - 1.0) < 1e-12
        and abs(z_overlap - z_born) < 3.0 * z_born_sem + 0.01
        and monotone_convergence
        and eig_min >= -1.0 - 1e-8
        and eig_max <= 1.0 + 1e-8
        and antisym_error < 1e-10
        and max(alpha_errors, default=0.0) < 1e-8
        and float(np.max(offpair_max)) < 1e-8
    )
    return {
        "passed": passed,
        "parameters": {"L": L, "gamma_m": gamma_m, "T": T, "zeta": zeta, "w_values": [0.0, w_ref]},
        "metrics": {
            "max_abs_C_T": float(np.max(np.abs(C_T))),
            "abs_z_T_minus_1": abs(z_T - 1.0),
            "Z_born_weighted": z_born,
            "Z_born_sem": z_born_sem,
            "Z_overlap_t0": z_overlap,
            "convergence_errors": convergence_errors,
            "monotone_convergence": monotone_convergence,
            "eig_min_iC": eig_min,
            "eig_max_iC": eig_max,
            "max_antisymmetry_error": antisym_error,
            "max_commuting_analytic_error": float(max(alpha_errors, default=0.0)),
            "max_commuting_offpair_error": float(np.max(offpair_max)),
            "weighted_born_ess": _effective_sample_size(born_weights),
        },
        "artifacts": {
            "commuting_plot": str(comm_path),
            "noncommuting_plot": str(noncomm_path),
        },
    }


def _run_test_15(config: ExtendedValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = 4
    w = 0.5
    gamma_m = 1.0
    T = 2.0
    zeta = 0.5
    observe_times = np.array([0.0, T / 4.0, T / 2.0, 3.0 * T / 4.0, T], dtype=np.float64)

    exact = build_exact_spin_chain_model(L=L, w=w, gamma_m=gamma_m)
    gauss = build_gaussian_chain_model(L=L, w=w, gamma_m=gamma_m)
    backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)
    number_ops = _number_ops(exact)
    nn_ops = _nearest_neighbor_number_ops(number_ops)

    rng_born = _rng(config.seed + 1501)
    born_records = _run_loop(
        "Test 15 Born",
        config.test15_born_n,
        lambda: _observe_exact_born_trajectory(
            exact,
            T,
            rng_born,
            observe_times=observe_times,
            number_ops=number_ops,
            nn_ops=nn_ops,
        ),
    )
    born_counts = np.asarray([rec["n_jumps"] for rec in born_records], dtype=int)
    born_weights = zeta**born_counts
    born_occ = np.asarray([rec["occupations"] for rec in born_records], dtype=np.float64)
    born_nn = np.asarray([rec["nn_products"] for rec in born_records], dtype=np.float64)
    born_occ_mean, born_occ_sem, born_ess = _weighted_array_mean_sem(born_occ, born_weights)
    born_nn_mean, born_nn_sem, _ = _weighted_array_mean_sem(born_nn, born_weights)
    born_conn = born_nn_mean - born_occ_mean[:, :-1] * born_occ_mean[:, 1:]

    rng_doob = _rng(config.seed + 1502)
    doob_records = _run_loop(
        "Test 15 Doob",
        config.test15_doob_n,
        lambda: _observe_gaussian_doob_trajectory(
            gauss,
            backward,
            T,
            zeta,
            rng_doob,
            observe_times=observe_times,
            record_occupations=True,
            record_nn_products=True,
        ),
    )
    doob_occ = np.asarray([rec["occupations"] for rec in doob_records], dtype=np.float64)
    doob_nn = np.asarray([rec["nn_products"] for rec in doob_records], dtype=np.float64)
    doob_occ_mean = np.mean(doob_occ, axis=0)
    doob_occ_sem = np.std(doob_occ, axis=0, ddof=1) / np.sqrt(max(doob_occ.shape[0], 1))
    doob_nn_mean = np.mean(doob_nn, axis=0)
    doob_nn_sem = np.std(doob_nn, axis=0, ddof=1) / np.sqrt(max(doob_nn.shape[0], 1))
    doob_conn = doob_nn_mean - doob_occ_mean[:, :-1] * doob_occ_mean[:, 1:]

    fig1, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), sharex=True, sharey=True)
    for site, ax in enumerate(axes.flat):
        ax.errorbar(observe_times, born_occ_mean[:, site], yerr=born_occ_sem[:, site], marker="o", lw=1.4, label="Born weighted")
        ax.errorbar(observe_times, doob_occ_mean[:, site], yerr=doob_occ_sem[:, site], marker="s", lw=1.4, linestyle="--", label="Doob")
        ax.set_title(f"Site {site + 1}")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\langle n_j(t) \rangle_{Q_s}$")
    axes.flat[0].legend(fontsize=8)
    occ_path = output_dir / "test15_occupation_profiles.png"
    _save_figure(fig1, occ_path, config.plot_dpi)

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.8))
    for bond in range(L - 1):
        ax2.errorbar(
            observe_times,
            born_conn[:, bond],
            yerr=born_nn_sem[:, bond],
            marker="o",
            lw=1.2,
            label=f"Born {bond + 1}-{bond + 2}",
        )
        ax2.errorbar(
            observe_times,
            doob_conn[:, bond],
            yerr=doob_nn_sem[:, bond],
            marker="s",
            lw=1.2,
            linestyle="--",
            label=f"Doob {bond + 1}-{bond + 2}",
        )
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$C_{j,j+1}(t)$")
    ax2.set_title("Test 15: Nearest-neighbor density correlators")
    ax2.legend(fontsize=7, ncol=2)
    corr_path = output_dir / "test15_nn_correlators.png"
    _save_figure(fig2, corr_path, config.plot_dpi)

    max_occ_diff = float(np.max(np.abs(doob_occ_mean - born_occ_mean)))
    max_conn_diff = float(np.max(np.abs(doob_conn - born_conn)))
    passed = bool(max_occ_diff < 0.05 and max_conn_diff < 0.08)
    return {
        "passed": passed,
        "parameters": {"L": L, "w": w, "gamma_m": gamma_m, "T": T, "zeta": zeta},
        "metrics": {
            "max_abs_occupation_difference": max_occ_diff,
            "max_abs_connected_nn_difference": max_conn_diff,
            "weighted_born_ess": born_ess,
        },
        "artifacts": {
            "occupation_plot": str(occ_path),
            "correlator_plot": str(corr_path),
        },
    }


def _run_test_16(config: ExtendedValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = 8
    w = 0.5
    gamma_m = 1.0
    T = 5.0
    zetas = [1.0, 0.7, 0.5, 0.3, 0.1]
    observe_times = np.array([T / 4.0, T / 2.0, 3.0 * T / 4.0, T], dtype=np.float64)

    exact = build_exact_spin_chain_model(L=L, w=w, gamma_m=gamma_m)
    gauss = build_gaussian_chain_model(L=L, w=w, gamma_m=gamma_m)
    benchmark: dict[str, Any] = {}

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    colors = plt.cm.cividis(np.linspace(0.1, 0.9, len(zetas)))

    max_in_sigma = 0.0
    for idx, zeta in enumerate(zetas):
        rng_born = _rng(config.seed + 1600 + idx)
        born_records = _run_loop(
            f"Test 16 Born zeta={zeta}",
            config.test16_born_n,
            lambda: _observe_exact_born_trajectory(
                exact,
                T,
                rng_born,
                observe_times=observe_times,
                entropy_cut=L // 2,
            ),
        )
        born_counts = np.asarray([rec["n_jumps"] for rec in born_records], dtype=int)
        born_weights = zeta**born_counts
        born_entropies = np.asarray([rec["entropies"] for rec in born_records], dtype=np.float64)
        born_mean, born_sem, born_ess = _weighted_array_mean_sem(born_entropies, born_weights)

        backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)
        rng_doob = _rng(config.seed + 1650 + idx)
        doob_records = _run_loop(
            f"Test 16 Doob zeta={zeta}",
            config.test16_doob_n,
            lambda: _observe_gaussian_doob_trajectory(
                gauss,
                backward,
                T,
                zeta,
                rng_doob,
                observe_times=observe_times,
                entropy_cut=L // 2,
            ),
        )
        doob_entropies = np.asarray([rec["entropies"] for rec in doob_records], dtype=np.float64)
        doob_mean = np.mean(doob_entropies, axis=0)
        doob_sem = np.std(doob_entropies, axis=0, ddof=1) / np.sqrt(max(doob_entropies.shape[0], 1))

        abs_diff = np.abs(doob_mean - born_mean)
        in_sigma = abs_diff / np.maximum(born_sem, 1e-12)
        max_in_sigma = max(max_in_sigma, float(np.max(in_sigma)))
        benchmark[str(zeta)] = {
            "born_mean": born_mean.tolist(),
            "born_sem": born_sem.tolist(),
            "doob_mean": doob_mean.tolist(),
            "doob_sem": doob_sem.tolist(),
            "abs_diff": abs_diff.tolist(),
            "in_sigma": in_sigma.tolist(),
            "weighted_born_ess": born_ess,
        }

        color = colors[idx]
        ax.plot(observe_times, doob_mean, color=color, lw=1.6, label=f"Doob zeta={zeta}")
        ax.plot(observe_times, born_mean, color=color, lw=1.4, linestyle="--", label=f"Born zeta={zeta}")
        ax.fill_between(observe_times, doob_mean - doob_sem, doob_mean + doob_sem, color=color, alpha=0.10)
        ax.fill_between(observe_times, born_mean - born_sem, born_mean + born_sem, color=color, alpha=0.20)

    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle S_{L/2}(t)\rangle_{Q_s}$")
    ax.set_title("Test 16: Entanglement benchmark with confidence bands")
    ax.legend(fontsize=7, ncol=2)
    plot_path = output_dir / "test16_entropy_benchmark.png"
    _save_figure(fig, plot_path, config.plot_dpi)

    passed = bool(max_in_sigma < 2.0)
    return {
        "passed": passed,
        "parameters": {"L": L, "w": w, "gamma_m": gamma_m, "T": T, "zetas": zetas},
        "metrics": {"benchmark": benchmark, "max_in_sigma": max_in_sigma},
        "artifacts": {"entropy_benchmark_plot": str(plot_path)},
    }


def _run_test_17(config: ExtendedValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zeta = 0.5
    w = 0.5
    gamma_m = 1.0
    T = 2.0
    systems = [4, 6, 8]

    relative_z_errors: list[float] = []
    click_errors: list[float] = []
    entropy_errors: list[float] = []
    metrics: dict[str, Any] = {}

    for idx, L in enumerate(systems):
        exact = build_exact_spin_chain_model(L=L, w=w, gamma_m=gamma_m)
        gauss = build_gaussian_chain_model(L=L, w=w, gamma_m=gamma_m)
        backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)

        rng_born = _rng(config.seed + 1700 + idx)
        born_records = _run_loop(
            f"Test 17 Born L={L}",
            config.test17_born_n,
            lambda: _observe_exact_born_trajectory(
                exact,
                T,
                rng_born,
                observe_times=np.array([T / 2.0]),
                entropy_cut=L // 2,
            ),
        )
        born_counts = np.asarray([rec["n_jumps"] for rec in born_records], dtype=int)
        born_weights = zeta**born_counts
        born_mean_clicks, _ = _weighted_mean_sem(born_counts.astype(np.float64), born_weights)
        born_entropy = np.asarray([rec["entropies"][0] for rec in born_records], dtype=np.float64)
        born_mean_entropy, _, born_ess = _weighted_mean_sem(born_entropy, born_weights)
        z_born = float(np.mean(born_weights))

        rng_doob = _rng(config.seed + 1750 + idx)
        doob_records = _run_loop(
            f"Test 17 Doob L={L}",
            config.test17_doob_n,
            lambda: _observe_gaussian_doob_trajectory(
                gauss,
                backward,
                T,
                zeta,
                rng_doob,
                observe_times=np.array([T / 2.0]),
                entropy_cut=L // 2,
            ),
        )
        doob_counts = np.asarray([rec["n_jumps"] for rec in doob_records], dtype=int)
        doob_mean_clicks = float(np.mean(doob_counts))
        doob_entropy = np.asarray([rec["entropies"][0] for rec in doob_records], dtype=np.float64)
        doob_mean_entropy = float(np.mean(doob_entropy))
        C0, z0 = backward.state_at(0.0)
        z_doob = gaussian_overlap(C0, gauss.gamma0, z_scalar=z0)

        rel_z_error = abs(z_doob - z_born) / max(abs(z_born), 1e-12)
        click_error = abs(doob_mean_clicks - born_mean_clicks)
        entropy_error = abs(doob_mean_entropy - born_mean_entropy)
        relative_z_errors.append(rel_z_error)
        click_errors.append(click_error)
        entropy_errors.append(entropy_error)
        metrics[str(L)] = {
            "Z_born": z_born,
            "Z_doob": z_doob,
            "relative_Z_error": rel_z_error,
            "mean_clicks_born": born_mean_clicks,
            "mean_clicks_doob": doob_mean_clicks,
            "mid_entropy_born": born_mean_entropy,
            "mid_entropy_doob": doob_mean_entropy,
            "weighted_born_ess": born_ess,
        }

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True)
    axes[0].plot(systems, relative_z_errors, marker="o")
    axes[0].set_ylabel(r"$|Z_D - Z_B| / Z_B$")
    axes[0].set_title("Partition function")
    axes[1].plot(systems, click_errors, marker="o")
    axes[1].set_ylabel(r"$|\langle N_T\rangle_D - \langle N_T\rangle_B|$")
    axes[1].set_title("Click count")
    axes[2].plot(systems, entropy_errors, marker="o")
    axes[2].set_ylabel(r"$|\langle S(T/2)\rangle_D - \langle S(T/2)\rangle_B|$")
    axes[2].set_title("Entropy")
    for ax in axes:
        ax.set_xlabel("L")
        ax.set_xscale("log", base=2)
    plot_path = output_dir / "test17_system_size_scaling.png"
    _save_figure(fig, plot_path, config.plot_dpi)

    passed = bool(
        max(relative_z_errors, default=0.0) < 0.20
        and max(click_errors, default=0.0) < 0.25
        and max(entropy_errors, default=0.0) < 0.25
    )
    return {
        "passed": passed,
        "parameters": {"L_values": systems, "w": w, "gamma_m": gamma_m, "T": T, "zeta": zeta},
        "metrics": metrics,
        "artifacts": {"scaling_plot": str(plot_path)},
    }


def _run_test_18(config: ExtendedValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = 4
    w = 0.5
    gamma_m = 1.0
    T = 2.0
    zeta = 0.5
    bins = np.linspace(0.0, T, 26)

    exact = build_exact_spin_chain_model(L=L, w=w, gamma_m=gamma_m)
    gauss = build_gaussian_chain_model(L=L, w=w, gamma_m=gamma_m)
    backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)

    rng_born = _rng(config.seed + 1801)
    born_records = _run_loop(
        "Test 18 Born",
        config.test18_born_n,
        lambda: _observe_exact_born_trajectory(exact, T, rng_born),
    )
    born_counts = np.asarray([rec["n_jumps"] for rec in born_records], dtype=int)
    born_weights = zeta**born_counts
    first_click_born: list[float] = []
    first_click_weights: list[float] = []
    interarrival_born: list[float] = []
    interarrival_weights: list[float] = []
    for rec, weight in zip(born_records, born_weights):
        jumps = rec["jump_times"]
        if jumps:
            first_click_born.append(float(jumps[0]))
            first_click_weights.append(float(weight))
        for left, right in zip(jumps[:-1], jumps[1:]):
            interarrival_born.append(float(right - left))
            interarrival_weights.append(float(weight))

    rng_doob = _rng(config.seed + 1802)
    doob_records = _run_loop(
        "Test 18 Doob",
        config.test18_doob_n,
        lambda: doob_gaussian_trajectory(gauss, backward, T, zeta, rng_doob),
    )
    first_click_doob: list[float] = []
    interarrival_doob: list[float] = []
    total_jumps = 0
    for rec in doob_records:
        jumps = rec.jump_times
        total_jumps += len(jumps)
        if jumps:
            first_click_doob.append(float(jumps[0]))
        for left, right in zip(jumps[:-1], jumps[1:]):
            interarrival_doob.append(float(right - left))

    first_click_born_arr = np.asarray(first_click_born, dtype=np.float64)
    first_click_doob_arr = np.asarray(first_click_doob, dtype=np.float64)
    interarrival_born_arr = np.asarray(interarrival_born, dtype=np.float64)
    interarrival_doob_arr = np.asarray(interarrival_doob, dtype=np.float64)
    first_click_weights_arr = np.asarray(first_click_weights, dtype=np.float64)
    interarrival_weights_arr = np.asarray(interarrival_weights, dtype=np.float64)
    ks_first_click = _weighted_ks_statistic(first_click_born_arr, first_click_weights_arr, first_click_doob_arr)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].hist(first_click_born_arr, bins=bins, weights=first_click_weights_arr, density=True, alpha=0.6, label="Born weighted")
    axes[0].hist(first_click_doob_arr, bins=bins, density=True, alpha=0.6, label="Doob")
    axes[0].set_xlabel(r"$t_1$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Test 18: First click time")
    axes[0].legend(fontsize=8)

    axes[1].hist(interarrival_born_arr, bins=bins, weights=interarrival_weights_arr, density=True, alpha=0.6, label="Born weighted")
    axes[1].hist(interarrival_doob_arr, bins=bins, density=True, alpha=0.6, label="Doob")
    lambda_bar = total_jumps / max(config.test18_doob_n * T, 1e-12)
    xs = np.linspace(0.0, T, 200)
    axes[1].plot(xs, lambda_bar * np.exp(-lambda_bar * xs), color="black", linestyle="--", label="exp ref")
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Test 18: Inter-arrival times")
    axes[1].legend(fontsize=8)
    plot_path = output_dir / "test18_click_time_statistics.png"
    _save_figure(fig, plot_path, config.plot_dpi)

    passed = bool(ks_first_click < 0.05)
    return {
        "passed": passed,
        "parameters": {"L": L, "w": w, "gamma_m": gamma_m, "T": T, "zeta": zeta},
        "metrics": {
            "ks_first_click_weighted_vs_doob": ks_first_click,
            "weighted_born_ess": _effective_sample_size(born_weights),
            "n_first_click_born": int(first_click_born_arr.size),
            "n_first_click_doob": int(first_click_doob_arr.size),
            "n_interarrival_born": int(interarrival_born_arr.size),
            "n_interarrival_doob": int(interarrival_doob_arr.size),
            "mean_doob_rate": float(lambda_bar),
        },
        "artifacts": {"click_time_plot": str(plot_path)},
    }


def _generate_plot_a(config: ExtendedValidationConfig, output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zetas = np.linspace(0.0, 1.0, config.plotA_zeta_points)
    systems = [4, 6, 8]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for idx, L in enumerate(systems):
        gauss = build_gaussian_chain_model(L=L, w=0.5, gamma_m=1.0)
        exact = build_exact_spin_chain_model(L=L, w=0.5, gamma_m=1.0)
        born_ref_rng = _rng(config.seed + 2000 + idx)
        born_ref = np.asarray(
            [ordinary_quantum_jump_trajectory(exact, 2.0, born_ref_rng).n_jumps for _ in range(config.plotA_n_traj)],
            dtype=np.float64,
        )
        born_ref_mean = float(np.mean(born_ref))
        means = []
        sems = []
        for zeta_idx, zeta in enumerate(zetas):
            backward = run_gaussian_backward_pass(gauss, T=2.0, zeta=float(zeta), sample_points=65)
            rng = _rng(config.seed + 2100 + 100 * idx + zeta_idx)
            counts = np.asarray(
                [doob_gaussian_trajectory(gauss, backward, 2.0, float(zeta), rng).n_jumps for _ in range(config.plotA_n_traj)],
                dtype=np.float64,
            )
            means.append(float(np.mean(counts)))
            sems.append(float(np.std(counts, ddof=1) / np.sqrt(len(counts))) if len(counts) > 1 else 0.0)
        ax.errorbar(zetas, means, yerr=sems, lw=1.3, label=f"L={L}")
        ax.axhline(born_ref_mean, linestyle="--", lw=1.0, color=ax.lines[-1].get_color())
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$\langle N_T \rangle_{Q_s}$")
    ax.set_title("Plot A: Mean click count vs zeta")
    ax.legend(fontsize=8)
    path = output_dir / "plot_A_mean_clicks_vs_zeta.png"
    _save_figure(fig, path, config.plot_dpi)
    return str(path)


def _generate_plot_b(config: ExtendedValidationConfig, output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zetas = np.linspace(0.05, 1.0, config.plotB_zeta_points)
    systems = [4, 6, 8, 12]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharex=True)
    for L in systems:
        gauss = build_gaussian_chain_model(L=L, w=0.5, gamma_m=1.0)
        cuts = [max(1, L // 4), L // 2, min(L - 1, (3 * L) // 4)]
        curves = {cut: [] for cut in cuts}
        for zeta_idx, zeta in enumerate(zetas):
            backward = run_gaussian_backward_pass(gauss, T=5.0, zeta=float(zeta), sample_points=65)
            rng = _rng(config.seed + 2200 + 1000 * L + zeta_idx)
            observe = np.linspace(0.0, 5.0, 9)
            cut_entropies = {cut: [] for cut in cuts}
            for _ in range(config.plotB_n_traj):
                record = _observe_gaussian_doob_trajectory(
                    gauss,
                    backward,
                    5.0,
                    float(zeta),
                    rng,
                    observe_times=observe,
                    record_covariances=True,
                )
                covariances = np.asarray(record["covariances"], dtype=np.float64)
                for cut in cuts:
                    cut_entropies[cut].append(float(np.mean([_entropy_from_covariance(cov, cut) for cov in covariances])))
            for cut in cuts:
                curves[cut].append(float(np.mean(cut_entropies[cut])) if cut_entropies[cut] else 0.0)
        for ax, cut in zip(axes, cuts):
            ax.plot(zetas, curves[cut], marker="o", lw=1.3, label=f"L={L}")
    for ax, label in zip(axes, ["L/4", "L/2", "3L/4"]):
        ax.set_xlabel(r"$\zeta$")
        ax.set_ylabel("Time-averaged entropy")
        ax.set_title(f"Cut {label}")
        ax.legend(fontsize=8)
    path = output_dir / "plot_B_entropy_vs_zeta.png"
    _save_figure(fig, path, config.plot_dpi)
    return str(path)


def _generate_plot_c(config: ExtendedValidationConfig, output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exact = build_exact_spin_chain_model(L=4, w=0.5, gamma_m=1.0)
    backward = run_exact_backward_pass(exact, T=2.0, zeta=0.5)
    observe_times = np.linspace(0.0, 2.0, config.plotC_grid_points)
    rng = _rng(config.seed + 2301)
    record = _observe_exact_born_trajectory(
        exact,
        2.0,
        rng,
        observe_times=observe_times,
        record_compensator=True,
    )
    counts_t = np.asarray(record["counts_t"], dtype=np.float64)
    lambda_t = np.asarray(record["compensator_t"], dtype=np.float64)
    zeta = 0.5
    z_hat = backward.overlap(0.0, exact.initial_state)
    rn_r = (zeta**counts_t) * np.exp((1.0 - zeta) * lambda_t)
    rn_q_terminal = np.zeros_like(observe_times)
    rn_q_terminal[-1] = (zeta ** record["n_jumps"]) / max(z_hat, 1e-12)

    fig, axes = plt.subplots(4, 1, figsize=(7.8, 8.4), sharex=True)
    axes[0].step(observe_times, counts_t, where="post")
    axes[0].set_ylabel(r"$N_t$")
    axes[1].plot(observe_times, lambda_t)
    axes[1].set_ylabel(r"$\Lambda_t$")
    axes[2].step(observe_times, zeta**counts_t, where="post", label=r"$\zeta^{N_t}$")
    axes[2].plot(observe_times, rn_r, label=r"$dR_\zeta/dP$", linestyle="--")
    axes[2].legend(fontsize=8)
    axes[2].set_ylabel("RN factor")
    axes[3].plot(observe_times, rn_r, label=r"$dR_\zeta/dP$", linestyle="--")
    axes[3].plot(observe_times, rn_q_terminal, label=r"$dQ_s/dP$ at $T$", marker="o")
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("Weight")
    axes[3].legend(fontsize=8)
    axes[0].set_title("Plot C: Radon-Nikodym factors along one Born trajectory")
    path = output_dir / "plot_C_radon_nikodym_trajectory.png"
    _save_figure(fig, path, config.plot_dpi)
    return str(path)


def _generate_plot_d(config: ExtendedValidationConfig, output_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = 6
    gamma_m = 1.0
    T = 2.0
    zeta = 0.5
    gauss = build_gaussian_chain_model(L=L, w=0.5, gamma_m=gamma_m)
    backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=257)
    t_vals = T - backward.sample_tau
    order = np.argsort(t_vals)
    t_vals = t_vals[order]
    covariances = backward.sample_covariances[order]
    analytic = np.tanh(0.5 * gamma_m * (1.0 - zeta) * (T - t_vals))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for idx, (a, b) in enumerate(gauss.jump_pairs):
        ax.plot(t_vals, covariances[:, a, b], lw=1.3, label=f"bond {idx + 1}")
    ax.plot(t_vals, analytic, color="black", linestyle="--", lw=1.4, label="w=0 analytic")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$C_{a_j,b_j}(t)$")
    ax.set_title("Plot D: Backward covariance evolution")
    ax.legend(fontsize=8, ncol=2)
    path = output_dir / "plot_D_backward_covariance_evolution.png"
    _save_figure(fig, path, config.plot_dpi)
    return str(path)


def run_extended_validation(config: ExtendedValidationConfig | None = None) -> dict[str, Any]:
    cfg = config or ExtendedValidationConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"config": asdict(cfg), "tests": {}, "artifacts": {}, "notes": []}
    total_start = time.perf_counter()

    micro = _run_overlap_microtest(cfg)
    report["overlap_microtest"] = micro

    test_runners = [
        ("test_10_compensator_factor", lambda: _run_test_10(cfg, output_dir)),
        ("test_11_martingale_property", lambda: _run_test_11(cfg, output_dir)),
        ("test_12_doob_survival_cdf", lambda: _run_test_12(cfg, output_dir)),
        ("test_13_backward_pass_stability", lambda: _run_test_13(cfg, output_dir)),
        ("test_14_lookahead_rates", lambda: _run_test_14(cfg, output_dir)),
        ("test_15_occupation_profile", lambda: _run_test_15(cfg, output_dir)),
        ("test_16_entropy_benchmark", lambda: _run_test_16(cfg, output_dir)),
        ("test_17_system_size_scaling", lambda: _run_test_17(cfg, output_dir)),
        ("test_18_click_time_statistics", lambda: _run_test_18(cfg, output_dir)),
    ]

    for idx, (name, runner) in enumerate(test_runners, 1):
        print(f"\n[{idx}/9] {name}", flush=True)
        t0 = time.perf_counter()
        result = runner()
        elapsed = time.perf_counter() - t0
        print(f"  -> {'PASS' if result.get('passed') else 'FAIL'}  ({elapsed:.1f}s)", flush=True)
        report["tests"][name] = result

    if cfg.generate_report_plots:
        report["artifacts"]["plot_A_mean_clicks_vs_zeta"] = _generate_plot_a(cfg, output_dir)
        report["artifacts"]["plot_B_entropy_vs_zeta"] = _generate_plot_b(cfg, output_dir)
        report["artifacts"]["plot_C_radon_nikodym_trajectory"] = _generate_plot_c(cfg, output_dir)
        report["artifacts"]["plot_D_backward_covariance_evolution"] = _generate_plot_d(cfg, output_dir)

    report["all_passed"] = bool(
        micro.get("passed", False) and all(result.get("passed", False) for result in report["tests"].values())
    )
    report["elapsed_seconds"] = float(time.perf_counter() - total_start)
    report["notes"].append("All weighted Born estimators report effective sample size where relevant.")
    return report
