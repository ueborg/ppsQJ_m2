from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pps_qj.backward_pass import run_exact_backward_pass, run_gaussian_backward_pass
from pps_qj.core.numerics import safe_normalize
from pps_qj.doob_wtmc import doob_exact_trajectory, doob_gaussian_trajectory
from pps_qj.exact_backend import (
    _propagate_unnormalized,
    _sample_channel,
    _sample_waiting_time,
    build_exact_spin_chain_model,
    integrate_lindblad,
    ordinary_quantum_jump_trajectory,
    procedure_b_trajectory,
    procedure_c_local_trajectory,
)
from pps_qj.gaussian_backend import (
    apply_projective_jump,
    build_gaussian_chain_model,
    covariance_from_orbitals,
    entanglement_entropy,
    orbitals_from_covariance,
    propagate_no_click_orbitals,
)
from pps_qj.observables.basic import entanglement_entropy_statevector
from pps_qj.overlaps import gaussian_overlap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


@dataclass(frozen=True)
class Part6ValidationConfig:
    output_dir: str = "outputs/part6_validation_full"
    seed: int = 20260330
    test1_n_traj: int = 2000
    test1_obs_points: int = 17
    test2_born_n: int = 400
    test2_doob_n: int = 400
    test2_proc_c_n: int = 400
    test2_zeta: float = 0.1
    test3_n_traj: int = 500
    test4_born_n: int = 300
    test4_doob_n: int = 300
    test5_born_n: int = 600
    test5_doob_n: int = 300
    test6_born_n: int = 600
    test6_doob_n: int = 600
    test6_proc_c_n: int = 600
    test7_large_n: int = 20
    test7_exact_n: int = 10
    test7_obs_points: int = 17
    test8_survival_grid_points: int = 33
    test9_born_n: int = 600
    test9_doob_n: int = 600
    test9_proc_b_n: int = 200
    test9_proc_c_n: int = 600
    plot_dpi: int = 160


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _counts_hist(counts: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    counts = np.asarray(counts, dtype=int)
    minlength = int(np.max(counts, initial=0)) + 1
    if weights is None:
        hist = np.bincount(counts, minlength=minlength).astype(np.float64)
    else:
        hist = np.bincount(
            counts,
            weights=np.asarray(weights, dtype=np.float64),
            minlength=minlength,
        ).astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def _pad_to_same_size(*arrays: np.ndarray) -> list[np.ndarray]:
    size = max((len(arr) for arr in arrays), default=0)
    return [np.pad(arr, (0, size - len(arr))) for arr in arrays]


def _total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p_pad, q_pad = _pad_to_same_size(np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64))
    return float(0.5 * np.abs(p_pad - q_pad).sum())


def _pooled_mean_sem(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    sem = float(np.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y))) if len(x) > 1 and len(y) > 1 else 0.0
    return mean_x, mean_y, sem


def _weighted_mean_sem(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0, 0.0
    normalized = weights / total
    mean = float(np.sum(normalized * values))
    ess_denom = float(np.sum(normalized**2))
    ess = 1.0 / ess_denom if ess_denom > 0.0 else 1.0
    var = float(np.sum(normalized * (values - mean) ** 2))
    sem = float(np.sqrt(max(var, 0.0) / max(ess, 1.0)))
    return mean, sem


def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    normalized = weights / total
    denom = float(np.sum(normalized**2))
    return 1.0 / denom if denom > 0.0 else 0.0


def _weighted_trajectory_average(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.zeros(samples.shape[1:], dtype=samples.dtype)
    normalized = weights / total
    return np.tensordot(normalized, samples, axes=(0, 0))


def _run_loop(label: str, n: int, fn) -> list:
    """Run fn() n times with progress printed every ~10%."""
    results = []
    step = max(1, n // 10)
    for i in range(n):
        results.append(fn())
        if (i + 1) % step == 0 or i + 1 == n:
            print(f"    {label}: {i + 1}/{n}", flush=True)
    return results


def _save_figure(fig, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def _observe_exact_born_trajectory(
    model,
    T: float,
    rng: np.random.Generator,
    observe_times: np.ndarray | None = None,
    *,
    entropy_cut: int | None = None,
    record_density: bool = False,
) -> dict[str, Any]:
    observe = np.asarray(observe_times if observe_times is not None else [], dtype=np.float64)
    n_obs = observe.size
    entropies = np.zeros(n_obs, dtype=np.float64) if entropy_cut is not None else None
    densities = (
        np.zeros((n_obs, model.dim, model.dim), dtype=np.complex128)
        if record_density
        else None
    )

    state = np.asarray(model.initial_state, dtype=np.complex128).copy()
    t = 0.0
    obs_idx = 0
    jump_times: list[float] = []
    channels: list[int] = []

    def record_state(psi: np.ndarray, idx: int) -> None:
        if entropies is not None:
            entropies[idx] = entanglement_entropy_statevector(psi, model.L, entropy_cut)
        if densities is not None:
            densities[idx] = np.outer(psi, psi.conj())

    while t < T:
        dt = _sample_waiting_time(model, state, rng)
        segment_end = T if not np.isfinite(dt) or t + dt >= T else t + dt

        while obs_idx < n_obs and observe[obs_idx] <= segment_end + 1e-12:
            psi_obs = safe_normalize(_propagate_unnormalized(model, state, float(observe[obs_idx] - t)))
            record_state(psi_obs, obs_idx)
            obs_idx += 1

        if segment_end >= T:
            state = safe_normalize(_propagate_unnormalized(model, state, T - t))
            t = T
            break

        psi_tilde = _propagate_unnormalized(model, state, dt)
        state = safe_normalize(psi_tilde)
        t += dt
        channel = _sample_channel(model, state, rng)
        state = safe_normalize(model.jump_projectors[channel] @ state)
        jump_times.append(t)
        channels.append(channel)

    while obs_idx < n_obs:
        record_state(state, obs_idx)
        obs_idx += 1

    return {
        "n_jumps": len(channels),
        "jump_times": jump_times,
        "channels": channels,
        "final_state": state,
        "entropies": entropies,
        "densities": densities,
    }


def _observe_gaussian_doob_trajectory(
    model,
    backward,
    T: float,
    zeta: float,
    rng: np.random.Generator,
    observe_times: np.ndarray | None = None,
    *,
    entropy_cut: int | None = None,
) -> dict[str, Any]:
    observe = np.asarray(observe_times if observe_times is not None else [], dtype=np.float64)
    n_obs = observe.size
    entropies = np.zeros(n_obs, dtype=np.float64) if entropy_cut is not None else None

    orbitals = np.asarray(model.orbitals0, dtype=np.complex128).copy()
    t = 0.0
    obs_idx = 0
    jump_times: list[float] = []
    channels: list[int] = []

    def record_covariance(covariance: np.ndarray, idx: int) -> None:
        if entropies is not None:
            entropies[idx] = entanglement_entropy(covariance, entropy_cut)

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
                alpha=model.alpha,
                n_monitored=len(model.jump_pairs),
            )
            C_t, z_t = backward.state_at(t + dt)
            numerator = evolution.branch_norm * gaussian_overlap(C_t, evolution.covariance, z_scalar=z_t)
            return float(numerator / denominator)

        if survival_fn(max_dt) > r:
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
            segment_end = t + jump_dt

        while obs_idx < n_obs and observe[obs_idx] <= segment_end + 1e-12:
            evolution_obs = propagate_no_click_orbitals(
                orbitals,
                model.h_effective,
                float(observe[obs_idx] - t),
                alpha=model.alpha,
                n_monitored=len(model.jump_pairs),
            )
            record_covariance(evolution_obs.covariance, obs_idx)
            obs_idx += 1

        if jump_dt is None:
            evolution = propagate_no_click_orbitals(
                orbitals,
                model.h_effective,
                max_dt,
                alpha=model.alpha,
                n_monitored=len(model.jump_pairs),
            )
            orbitals = evolution.orbitals_normalized
            t = T
            break

        evolution = propagate_no_click_orbitals(
            orbitals,
            model.h_effective,
            jump_dt,
            alpha=model.alpha,
            n_monitored=len(model.jump_pairs),
        )
        pre_orbitals = evolution.orbitals_normalized
        pre_covariance = evolution.covariance
        t += jump_dt

        C_jump, z_jump = backward.state_at(t)
        overlap_pre = gaussian_overlap(C_jump, pre_covariance, z_scalar=z_jump)
        rates = np.zeros(len(model.jump_pairs), dtype=np.float64)
        post_covariances: list[np.ndarray] = []
        for idx, jump_pair in enumerate(model.jump_pairs):
            q, post_covariance = apply_projective_jump(pre_covariance, jump_pair)
            overlap_post = gaussian_overlap(C_jump, post_covariance, z_scalar=z_jump)
            rates[idx] = zeta * 2.0 * model.alpha * q * overlap_post / overlap_pre
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
    }


def _single_mode_analytics(q0: float, alpha: float, T: float, zeta: float) -> dict[str, float]:
    survival = (1.0 - q0) + q0 * np.exp(-2.0 * alpha * T)
    partition = (1.0 - q0) + q0 * np.exp(-2.0 * (1.0 - zeta) * alpha * T)
    return {
        "S": float(survival),
        "Z": float(partition),
        "Q_no_click": float(survival / partition),
        "R_no_click": float(survival**zeta),
    }


def _run_test_1(config: Part6ValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = 4
    w = 0.5
    alpha = 0.5
    T = 2.0
    zeta = 1.0
    observe_times = np.linspace(0.0, T, config.test1_obs_points)

    exact = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
    gauss = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    backward_gauss = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)

    max_abs_C = 0.0
    max_abs_z_minus_1 = 0.0
    for t in np.linspace(0.0, T, 9):
        C_t, z_t = backward_gauss.state_at(float(t))
        max_abs_C = max(max_abs_C, float(np.max(np.abs(C_t))))
        max_abs_z_minus_1 = max(max_abs_z_minus_1, abs(z_t - 1.0))

    exact_small = build_exact_spin_chain_model(L=2, w=0.0, alpha=0.5)
    backward_exact = run_exact_backward_pass(exact_small, T=2.0, zeta=1.0)
    doob_seed = config.seed + 11
    born_seed = config.seed + 11
    doob_small = doob_exact_trajectory(exact_small, backward_exact, 2.0, 1.0, _rng(doob_seed))
    born_small = ordinary_quantum_jump_trajectory(exact_small, 2.0, _rng(born_seed))

    exact_counts = np.zeros(config.test1_n_traj, dtype=int)
    exact_entropies = np.zeros((config.test1_n_traj, observe_times.size), dtype=np.float64)
    exact_densities = np.zeros((config.test1_n_traj, observe_times.size, exact.dim, exact.dim), dtype=np.complex128)
    rng_exact = _rng(config.seed + 101)
    _step1 = max(1, config.test1_n_traj // 10)
    for idx in range(config.test1_n_traj):
        record = _observe_exact_born_trajectory(
            exact,
            T,
            rng_exact,
            observe_times,
            entropy_cut=L // 2,
            record_density=True,
        )
        exact_counts[idx] = record["n_jumps"]
        exact_entropies[idx] = record["entropies"]
        exact_densities[idx] = record["densities"]
        if (idx + 1) % _step1 == 0 or idx + 1 == config.test1_n_traj:
            print(f"    exact Born: {idx + 1}/{config.test1_n_traj}", flush=True)

    doob_counts = np.zeros(config.test1_n_traj, dtype=int)
    doob_entropies = np.zeros((config.test1_n_traj, observe_times.size), dtype=np.float64)
    rng_doob = _rng(config.seed + 202)
    for idx in range(config.test1_n_traj):
        record = _observe_gaussian_doob_trajectory(
            gauss,
            backward_gauss,
            T,
            zeta,
            rng_doob,
            observe_times,
            entropy_cut=L // 2,
        )
        doob_counts[idx] = record["n_jumps"]
        doob_entropies[idx] = record["entropies"]
        if (idx + 1) % _step1 == 0 or idx + 1 == config.test1_n_traj:
            print(f"    Doob Gaussian: {idx + 1}/{config.test1_n_traj}", flush=True)

    exact_mean, doob_mean, mean_sem = _pooled_mean_sem(exact_counts, doob_counts)
    exact_pmf = _counts_hist(exact_counts)
    doob_pmf = _counts_hist(doob_counts)
    pmf_tv = _total_variation(exact_pmf, doob_pmf)
    mean_entropy_exact = np.mean(exact_entropies, axis=0)
    mean_entropy_doob = np.mean(doob_entropies, axis=0)
    entropy_rms = float(np.sqrt(np.mean((mean_entropy_exact - mean_entropy_doob) ** 2)))

    lindblad_t, lindblad_rhos = integrate_lindblad(exact, T, t_eval=observe_times)
    mean_rhos = np.mean(exact_densities, axis=0)
    density_fro_errors = np.array(
        [np.linalg.norm(mean_rhos[idx] - lindblad_rhos[idx], ord="fro") for idx in range(observe_times.size)],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(observe_times, mean_entropy_exact, label="Exact Born", lw=1.8)
    ax.plot(observe_times, mean_entropy_doob, label="Doob Gaussian", lw=1.8, linestyle="--")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle S_{L/2}(t)\rangle$")
    ax.set_title("Test 1: zeta=1 Entanglement Recovery")
    ax.legend()
    plot_path = output_dir / "test1_zeta1_entropy.png"
    _save_figure(fig, plot_path, config.plot_dpi)

    passed = bool(
        max_abs_C < 1e-8
        and max_abs_z_minus_1 < 1e-8
        and abs(doob_mean - exact_mean) <= 4.0 * mean_sem + 0.05
        and pmf_tv < 0.12
        and entropy_rms < 0.15
        and float(np.max(density_fro_errors)) < 0.02
        and doob_small.channels == born_small.channels
        and np.allclose(doob_small.jump_times, born_small.jump_times, atol=1e-10)
        and np.array_equal(np.asarray(lindblad_t), observe_times)
    )

    return {
        "passed": passed,
        "parameters": {"L": L, "w": w, "alpha": alpha, "T": T, "zeta": zeta, "n_traj": config.test1_n_traj},
        "metrics": {
            "max_abs_C": max_abs_C,
            "max_abs_z_minus_1": max_abs_z_minus_1,
            "mean_clicks_exact": exact_mean,
            "mean_clicks_doob": doob_mean,
            "mean_clicks_sem": mean_sem,
            "pmf_total_variation": pmf_tv,
            "entropy_rms_difference": entropy_rms,
            "max_ensemble_density_fro_error": float(np.max(density_fro_errors)),
            "mean_ensemble_density_fro_error": float(np.mean(density_fro_errors)),
            "max_density_fro_error": float(np.max(density_fro_errors)),
            "mean_density_fro_error": float(np.mean(density_fro_errors)),
        },
        "artifacts": {"entropy_plot": str(plot_path)},
    }


def _run_test_2(config: Part6ValidationConfig) -> dict[str, Any]:
    L = 4
    w = 0.5
    alpha = 0.5
    T = 1.0
    zeta = config.test2_zeta

    exact = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
    gauss = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)

    rng_born = _rng(config.seed + 301)
    born_counts = np.array(
        _run_loop("Born", config.test2_born_n, lambda: ordinary_quantum_jump_trajectory(exact, T, rng_born).n_jumps),
        dtype=int,
    )
    weights = zeta**born_counts
    exact_mean, exact_mean_sem = _weighted_mean_sem(born_counts.astype(np.float64), weights)
    exact_p0, exact_p0_sem = _weighted_mean_sem((born_counts == 0).astype(np.float64), weights)
    ess = _effective_sample_size(weights)
    survival_born = float(np.mean(born_counts == 0))
    z_partition = float(np.mean(weights))
    p0_analytic = survival_born / z_partition

    rng_doob = _rng(config.seed + 302)
    doob_counts = np.array(
        _run_loop("Doob", config.test2_doob_n, lambda: doob_gaussian_trajectory(gauss, backward, T, zeta, rng_doob).n_jumps),
        dtype=int,
    )
    rng_c = _rng(config.seed + 303)
    proc_c_counts = np.array(
        _run_loop("Proc C", config.test2_proc_c_n, lambda: procedure_c_local_trajectory(exact, T, zeta, rng_c).n_jumps),
        dtype=int,
    )

    doob_mean = float(np.mean(doob_counts))
    doob_mean_sem = float(np.std(doob_counts, ddof=1) / np.sqrt(len(doob_counts))) if len(doob_counts) > 1 else 0.0
    doob_p0 = float(np.mean(doob_counts == 0))
    doob_p0_sem = float(np.sqrt(doob_p0 * (1.0 - doob_p0) / len(doob_counts)))
    proc_c_p0 = float(np.mean(proc_c_counts == 0))

    passed = bool(
        doob_p0 > 0.80
        and doob_mean < 0.20
        and abs(doob_mean - exact_mean) <= 4.0 * max(doob_mean_sem, exact_mean_sem) + 0.03
        and abs(doob_p0 - p0_analytic) <= 4.0 * max(doob_p0_sem, exact_p0_sem) + 0.03
    )

    return {
        "passed": passed,
        "parameters": {
            "L": L,
            "w": w,
            "alpha": alpha,
            "T": T,
            "zeta": zeta,
            "born_n": config.test2_born_n,
            "doob_n": config.test2_doob_n,
            "procedure_c_n": config.test2_proc_c_n,
        },
        "metrics": {
            "mean_clicks_exact_weighted": exact_mean,
            "mean_clicks_doob": doob_mean,
            "p0_exact_weighted": exact_p0,
            "p0_doob": doob_p0,
            "p0_procedure_c": proc_c_p0,
            "p0_analytic_s_over_z": p0_analytic,
            "z_partition_from_born": z_partition,
            "survival_born": survival_born,
            "weighted_born_ess": ess,
        },
    }


def _run_test_3(config: Part6ValidationConfig) -> dict[str, Any]:
    L = 2
    w = 0.0
    alpha = 0.5
    T = 2.0
    zeta = 0.5

    exact = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
    gauss = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    backward_exact = run_exact_backward_pass(exact, T=T, zeta=zeta)
    backward_gauss = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)
    q0 = float(np.real(np.vdot(exact.initial_state, exact.jump_projectors[0] @ exact.initial_state)))
    analytics = _single_mode_analytics(q0=q0, alpha=alpha, T=T, zeta=zeta)

    C0, z0 = backward_gauss.state_at(0.0)
    z_doob = gaussian_overlap(C0, gauss.gamma0, z_scalar=z0)
    z_exact = backward_exact.overlap(0.0, exact.initial_state)

    sigma_errors = []
    for t in np.linspace(0.0, T, 11):
        C_t, _ = backward_gauss.state_at(float(t))
        sigma_expected = np.tanh(alpha * (1.0 - zeta) * (T - t))
        sigma_errors.append(abs(C_t[0, 3] - sigma_expected))
        mask = np.ones_like(C_t, dtype=bool)
        mask[0, 3] = False
        mask[3, 0] = False
        mask[np.diag_indices_from(mask)] = False
        sigma_errors.append(float(np.max(np.abs(C_t[mask]))))

    rng_doob = _rng(config.seed + 401)
    doob_counts = np.array(
        _run_loop("Doob", config.test3_n_traj, lambda: doob_gaussian_trajectory(gauss, backward_gauss, T, zeta, rng_doob).n_jumps),
        dtype=int,
    )
    rng_c = _rng(config.seed + 402)
    proc_c_counts = np.array(
        _run_loop("Proc C", config.test3_n_traj, lambda: procedure_c_local_trajectory(exact, T, zeta, rng_c).n_jumps),
        dtype=int,
    )
    p0_doob = float(np.mean(doob_counts == 0))
    p0_proc_c = float(np.mean(proc_c_counts == 0))

    passed = bool(
        abs(z_exact - analytics["Z"]) < 1e-10
        and abs(z_doob - analytics["Z"]) < 1e-8
        and abs(p0_doob - analytics["Q_no_click"]) < 0.03
        and abs(p0_proc_c - analytics["R_no_click"]) < 0.04
        and abs(analytics["Q_no_click"] - analytics["R_no_click"]) > 0.05
        and max(sigma_errors) < 5e-6
    )

    return {
        "passed": passed,
        "parameters": {"L": L, "w": w, "alpha": alpha, "T": T, "zeta": zeta, "n_traj": config.test3_n_traj},
        "metrics": {
            "q0": q0,
            "analytic_Z": analytics["Z"],
            "gaussian_Z": z_doob,
            "exact_Z": z_exact,
            "analytic_Q_no_click": analytics["Q_no_click"],
            "analytic_R_no_click": analytics["R_no_click"],
            "doob_no_click": p0_doob,
            "procedure_c_no_click": p0_proc_c,
            "max_backward_covariance_error": max(sigma_errors),
        },
    }


def _run_test_4(config: Part6ValidationConfig) -> dict[str, Any]:
    T = 2.0
    zeta = 0.5
    alpha = 0.5
    ls = [4, 6]
    entries: dict[str, Any] = {}
    all_passed = True

    for L in ls:
        exact = build_exact_spin_chain_model(L=L, w=0.0, alpha=alpha)
        gauss = build_gaussian_chain_model(L=L, w=0.0, alpha=alpha)
        backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)

        covariance_errors = []
        alpha_errors = []
        block_diagonal_errors = []
        for t in np.linspace(0.0, T, 11):
            C_t, _ = backward.state_at(float(t))
            sigma_expected = np.tanh(alpha * (1.0 - zeta) * (T - t))
            allowed = set()
            for a, b in gauss.jump_pairs:
                allowed.add((a, b))
                allowed.add((b, a))
                covariance_errors.append(abs(C_t[a, b] - sigma_expected))
                covariance_errors.append(abs(C_t[b, a] + sigma_expected))
            for m in range(C_t.shape[0]):
                for n in range(C_t.shape[1]):
                    if m == n or (m, n) in allowed:
                        continue
                    covariance_errors.append(abs(C_t[m, n]))

            K_t, _ = backward.generator_at(float(t))
            alpha_expected = 2.0 * alpha * (1.0 - zeta) * (T - t)
            for a, b in gauss.jump_pairs:
                alpha_errors.append(abs(K_t[a, b] - alpha_expected))
                alpha_errors.append(abs(K_t[b, a] + alpha_expected))
                for c in range(K_t.shape[0]):
                    if c in (a, b):
                        continue
                    block_diagonal_errors.append(abs(K_t[a, c]))
                    block_diagonal_errors.append(abs(K_t[b, c]))

        rng_born = _rng(config.seed + 500 + 10 * L)
        born_counts = np.array(
            _run_loop(f"L={L} Born", config.test4_born_n, lambda: ordinary_quantum_jump_trajectory(exact, T, rng_born).n_jumps),
            dtype=int,
        )
        weights = zeta**born_counts
        exact_mean, exact_mean_sem = _weighted_mean_sem(born_counts.astype(np.float64), weights)
        ess = _effective_sample_size(weights)

        rng_doob = _rng(config.seed + 600 + 10 * L)
        doob_counts = np.array(
            _run_loop(f"L={L} Doob", config.test4_doob_n, lambda: doob_gaussian_trajectory(gauss, backward, T, zeta, rng_doob).n_jumps),
            dtype=int,
        )
        doob_mean = float(np.mean(doob_counts))
        doob_mean_sem = float(np.std(doob_counts, ddof=1) / np.sqrt(len(doob_counts))) if len(doob_counts) > 1 else 0.0

        h = 1e-2
        backward_plus = run_gaussian_backward_pass(gauss, T=T, zeta=min(zeta + h, 0.999), sample_points=33)
        backward_minus = run_gaussian_backward_pass(gauss, T=T, zeta=max(zeta - h, 1e-6), sample_points=33)
        C_plus, z_plus = backward_plus.state_at(0.0)
        C_minus, z_minus = backward_minus.state_at(0.0)
        Z_plus = gaussian_overlap(C_plus, gauss.gamma0, z_scalar=z_plus)
        Z_minus = gaussian_overlap(C_minus, gauss.gamma0, z_scalar=z_minus)
        mean_from_derivative = float(zeta * (np.log(Z_plus) - np.log(Z_minus)) / (2.0 * h))

        passed = bool(
            max(covariance_errors) < 5e-5
            and max(alpha_errors, default=0.0) < 1e-8
            and max(block_diagonal_errors, default=0.0) < 1e-8
            and abs(doob_mean - exact_mean) <= 4.0 * max(doob_mean_sem, exact_mean_sem) + 0.05
            and abs(doob_mean - mean_from_derivative) < 0.15
        )
        all_passed = all_passed and passed
        entries[f"L={L}"] = {
            "passed": passed,
            "mean_clicks_exact_weighted": exact_mean,
            "mean_clicks_doob": doob_mean,
            "mean_clicks_from_partition_derivative": mean_from_derivative,
            "max_covariance_error": max(covariance_errors),
            "max_generator_alpha_error": max(alpha_errors, default=0.0),
            "max_generator_block_diagonal_error": max(block_diagonal_errors, default=0.0),
            "weighted_born_ess": ess,
        }

    return {
        "passed": all_passed,
        "parameters": {"L_values": ls, "w": 0.0, "alpha": alpha, "T": T, "zeta": zeta},
        "metrics": entries,
    }


def _run_test_5(config: Part6ValidationConfig) -> dict[str, Any]:
    L = 4
    w = 0.5
    alpha = 0.5
    T = 2.0
    zetas = [0.9, 0.7, 0.5, 0.3]

    exact = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
    gauss = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    rng_born = _rng(config.seed + 701)
    born_counts = np.array(
        [ordinary_quantum_jump_trajectory(exact, T, rng_born).n_jumps for _ in range(config.test5_born_n)],
        dtype=int,
    )

    metrics: dict[str, Any] = {}
    all_passed = True
    for idx, zeta in enumerate(zetas):
        backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)
        C0, z0 = backward.state_at(0.0)
        Z_doob = gaussian_overlap(C0, gauss.gamma0, z_scalar=z0)
        weights = zeta**born_counts
        Z_exact = float(np.mean(weights))
        mean_exact, mean_exact_sem = _weighted_mean_sem(born_counts.astype(np.float64), weights)

        rng_doob = _rng(config.seed + 702 + idx)
        doob_counts = np.array(
            _run_loop(f"zeta={zeta} Doob", config.test5_doob_n, lambda: doob_gaussian_trajectory(gauss, backward, T, zeta, rng_doob).n_jumps),
            dtype=int,
        )
        doob_mean = float(np.mean(doob_counts))
        doob_mean_sem = float(np.std(doob_counts, ddof=1) / np.sqrt(len(doob_counts))) if len(doob_counts) > 1 else 0.0

        passed = bool(
            abs(Z_doob - Z_exact) < 0.05
            and abs(doob_mean - mean_exact) <= 4.0 * max(doob_mean_sem, mean_exact_sem) + 0.08
        )
        all_passed = all_passed and passed
        metrics[str(zeta)] = {
            "passed": passed,
            "Z_exact_weighted": Z_exact,
            "Z_doob_overlap": Z_doob,
            "mean_clicks_exact_weighted": mean_exact,
            "mean_clicks_doob": doob_mean,
        }

    return {
        "passed": all_passed,
        "parameters": {"L": L, "w": w, "alpha": alpha, "T": T, "zetas": zetas},
        "metrics": metrics,
    }


def _run_test_6(config: Part6ValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = 4
    w = 0.5
    alpha = 0.5
    T = 2.0
    zeta = 0.5

    exact = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
    gauss = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)

    rng_born = _rng(config.seed + 801)
    born_counts = np.array(
        _run_loop("Born", config.test6_born_n, lambda: ordinary_quantum_jump_trajectory(exact, T, rng_born).n_jumps),
        dtype=int,
    )
    weights = zeta**born_counts
    pmf_weighted = _counts_hist(born_counts, weights=weights)
    pmf_born = _counts_hist(born_counts)

    rng_doob = _rng(config.seed + 802)
    doob_counts = np.array(
        _run_loop("Doob", config.test6_doob_n, lambda: doob_gaussian_trajectory(gauss, backward, T, zeta, rng_doob).n_jumps),
        dtype=int,
    )
    pmf_doob = _counts_hist(doob_counts)

    rng_c = _rng(config.seed + 803)
    proc_c_counts = np.array(
        _run_loop("Proc C", config.test6_proc_c_n, lambda: procedure_c_local_trajectory(exact, T, zeta, rng_c).n_jumps),
        dtype=int,
    )
    pmf_c = _counts_hist(proc_c_counts)

    pmf_weighted, pmf_doob, pmf_born, pmf_c = _pad_to_same_size(pmf_weighted, pmf_doob, pmf_born, pmf_c)
    xs = np.arange(len(pmf_weighted))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(xs - 1.5 * width, pmf_weighted, width=width, label=r"Exact $Q_s$ (weighted Born)")
    ax.bar(xs - 0.5 * width, pmf_doob, width=width, label="Doob WTMC")
    ax.bar(xs + 0.5 * width, pmf_born, width=width, label="Born rule")
    ax.bar(xs + 1.5 * width, pmf_c, width=width, label=r"Procedure C / $R_\zeta$")
    ax.set_xlabel(r"$N_T$")
    ax.set_ylabel("Probability")
    ax.set_title("Test 6: Click-Count Distribution Comparison")
    ax.legend(fontsize=9)
    plot_path = output_dir / "test6_click_count_pmf.png"
    _save_figure(fig, plot_path, config.plot_dpi)

    tv_doob_exact = _total_variation(pmf_doob, pmf_weighted)
    tv_doob_c = _total_variation(pmf_doob, pmf_c)
    passed = bool(tv_doob_exact < 0.12 and tv_doob_c > 0.04)

    return {
        "passed": passed,
        "parameters": {
            "L": L,
            "w": w,
            "alpha": alpha,
            "T": T,
            "zeta": zeta,
            "born_n": config.test6_born_n,
            "doob_n": config.test6_doob_n,
            "procedure_c_n": config.test6_proc_c_n,
        },
        "metrics": {
            "tv_doob_vs_exact_weighted": tv_doob_exact,
            "tv_doob_vs_procedure_c": tv_doob_c,
            "pmf_exact_weighted": pmf_weighted.tolist(),
            "pmf_doob": pmf_doob.tolist(),
            "pmf_born": pmf_born.tolist(),
            "pmf_procedure_c": pmf_c.tolist(),
        },
        "artifacts": {"pmf_plot": str(plot_path)},
    }


def _run_test_7(config: Part6ValidationConfig, output_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha = 0.5
    w = 0.5
    T = 5.0
    zetas = [1.0, 0.7, 0.5, 0.3, 0.1]
    observe_times = np.linspace(0.0, T, config.test7_obs_points)
    large_means: list[float] = []

    for idx, zeta in enumerate(zetas):
        print(f"  L=12 backward pass  zeta={zeta}  ({idx + 1}/{len(zetas)})", flush=True)
        gauss = build_gaussian_chain_model(L=12, w=w, alpha=alpha)
        backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)
        rng_doob = _rng(config.seed + 900 + idx)
        entropies = np.zeros((config.test7_large_n, observe_times.size), dtype=np.float64)
        _step7 = max(1, config.test7_large_n // 5)
        for n in range(config.test7_large_n):
            record = _observe_gaussian_doob_trajectory(
                gauss,
                backward,
                T,
                zeta,
                rng_doob,
                observe_times,
                entropy_cut=gauss.L // 2,
            )
            entropies[n] = record["entropies"]
            if (n + 1) % _step7 == 0 or n + 1 == config.test7_large_n:
                print(f"    L=12 traj: {n + 1}/{config.test7_large_n}", flush=True)
        large_means.append(float(np.mean(entropies)))

    benchmark_metrics: dict[str, Any] = {}
    benchmark_passed = True
    for idx, zeta in enumerate(zetas):
        print(f"  L=8 benchmark  zeta={zeta}  ({idx + 1}/{len(zetas)})", flush=True)
        L = 8
        exact = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
        gauss = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
        backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)

        rng_born = _rng(config.seed + 1000 + idx)
        exact_entropies = np.zeros((config.test7_exact_n, observe_times.size), dtype=np.float64)
        exact_counts = np.zeros(config.test7_exact_n, dtype=int)
        for n in range(config.test7_exact_n):
            record = _observe_exact_born_trajectory(
                exact,
                T,
                rng_born,
                observe_times,
                entropy_cut=L // 2,
                record_density=False,
            )
            exact_entropies[n] = record["entropies"]
            exact_counts[n] = record["n_jumps"]
            print(f"    L=8 exact Born: {n + 1}/{config.test7_exact_n}", flush=True)
        weights = zeta**exact_counts
        exact_timeavg = float(np.mean(_weighted_trajectory_average(exact_entropies, weights)))

        rng_doob = _rng(config.seed + 1100 + idx)
        gauss_entropies = np.zeros((config.test7_exact_n, observe_times.size), dtype=np.float64)
        for n in range(config.test7_exact_n):
            record = _observe_gaussian_doob_trajectory(
                gauss,
                backward,
                T,
                zeta,
                rng_doob,
                observe_times,
                entropy_cut=L // 2,
            )
            gauss_entropies[n] = record["entropies"]
            print(f"    L=8 Doob: {n + 1}/{config.test7_exact_n}", flush=True)
        gauss_timeavg = float(np.mean(gauss_entropies))
        diff = abs(gauss_timeavg - exact_timeavg)
        passed = diff < 0.30
        benchmark_passed = benchmark_passed and passed
        benchmark_metrics[str(zeta)] = {
            "passed": passed,
            "exact_weighted_timeavg_entropy_L8": exact_timeavg,
            "gaussian_doob_timeavg_entropy_L8": gauss_timeavg,
            "absolute_difference": diff,
        }

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(zetas, large_means, marker="o", lw=1.8, label="L=12 Gaussian Doob")
    benchmark_doob = [benchmark_metrics[str(z)]["gaussian_doob_timeavg_entropy_L8"] for z in zetas]
    benchmark_exact = [benchmark_metrics[str(z)]["exact_weighted_timeavg_entropy_L8"] for z in zetas]
    ax.plot(zetas, benchmark_doob, marker="s", lw=1.5, linestyle="--", label="L=8 Gaussian Doob")
    ax.plot(zetas, benchmark_exact, marker="^", lw=1.5, linestyle=":", label="L=8 Exact weighted")
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel("Time-averaged middle-cut entropy")
    ax.set_title("Test 7: Entanglement Scaling under $Q_s$")
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    plot_path = output_dir / "test7_entropy_scaling.png"
    _save_figure(fig, plot_path, config.plot_dpi)

    monotone = all(large_means[idx] >= large_means[idx + 1] - 1e-9 for idx in range(len(large_means) - 1))
    passed = bool(monotone and benchmark_passed)

    return {
        "passed": passed,
        "parameters": {
            "L_large": 12,
            "L_benchmark": 8,
            "w": w,
            "alpha": alpha,
            "T": T,
            "zetas": zetas,
            "large_n": config.test7_large_n,
            "benchmark_n": config.test7_exact_n,
        },
        "metrics": {
            "large_system_timeavg_entropy": dict(zip([str(z) for z in zetas], large_means)),
            "monotone_nonincreasing": monotone,
            "benchmark_L8": benchmark_metrics,
        },
        "artifacts": {"entropy_scaling_plot": str(plot_path)},
    }


def _run_test_8(config: Part6ValidationConfig) -> dict[str, Any]:
    L = 4
    w = 0.5
    alpha = 0.5
    T = 2.0
    zeta = 0.5

    gauss = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)
    trajectory = doob_gaussian_trajectory(
        gauss,
        backward,
        T,
        zeta,
        _rng(config.seed + 1201),
        survival_grid_points=config.test8_survival_grid_points,
    )
    segments = trajectory.diagnostics.get("conditioned_survival_segments", [])
    monotone_flags = []
    min_raw_overlap = float("inf")
    max_positive_diff = -float("inf")
    for segment in segments:
        values = np.asarray(segment["values"], dtype=np.float64)
        denominator = float(segment["denominator"])
        raw = denominator * values
        if raw.size > 1:
            diffs = np.diff(raw)
            monotone_flags.append(bool(np.all(diffs <= 1e-10)))
            max_positive_diff = max(max_positive_diff, float(np.max(diffs)))
        min_raw_overlap = min(min_raw_overlap, float(np.min(raw)))

    final_covariance = covariance_from_orbitals(np.asarray(trajectory.final_state, dtype=np.complex128))
    final_overlap = gaussian_overlap(np.zeros_like(final_covariance), final_covariance, z_scalar=1.0)
    passed = bool(all(monotone_flags) and min_raw_overlap > 0.0 and abs(final_overlap - 1.0) < 1e-12)

    return {
        "passed": passed,
        "parameters": {"L": L, "w": w, "alpha": alpha, "T": T, "zeta": zeta},
        "metrics": {
            "n_segments": len(segments),
            "max_positive_diff_in_raw_overlap": max_positive_diff,
            "min_raw_overlap": min_raw_overlap,
            "final_normalized_overlap_at_T": final_overlap,
        },
    }


def _run_test_9(config: Part6ValidationConfig) -> dict[str, Any]:
    L = 4
    w = 0.0
    alpha = 0.5
    T = 2.0
    zeta = 0.5

    exact = build_exact_spin_chain_model(L=L, w=w, alpha=alpha)
    gauss = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    backward = run_gaussian_backward_pass(gauss, T=T, zeta=zeta, sample_points=65)

    rng_born = _rng(config.seed + 1301)
    born_counts = np.array(
        _run_loop("Born", config.test9_born_n, lambda: ordinary_quantum_jump_trajectory(exact, T, rng_born).n_jumps),
        dtype=int,
    )
    weights = zeta**born_counts
    exact_p0, exact_p0_sem = _weighted_mean_sem((born_counts == 0).astype(np.float64), weights)
    exact_mean, exact_mean_sem = _weighted_mean_sem(born_counts.astype(np.float64), weights)

    rng_doob = _rng(config.seed + 1302)
    doob_counts = np.array(
        _run_loop("Doob", config.test9_doob_n, lambda: doob_gaussian_trajectory(gauss, backward, T, zeta, rng_doob).n_jumps),
        dtype=int,
    )
    doob_p0 = float(np.mean(doob_counts == 0))
    doob_mean = float(np.mean(doob_counts))

    rng_b = _rng(config.seed + 1303)
    proc_b_counts = np.array(
        _run_loop("Proc B", config.test9_proc_b_n, lambda: procedure_b_trajectory(exact, T, zeta, rng_b).n_jumps),
        dtype=int,
    )
    proc_b_p0 = float(np.mean(proc_b_counts == 0))
    proc_b_mean = float(np.mean(proc_b_counts))

    rng_c = _rng(config.seed + 1304)
    proc_c_counts = np.array(
        _run_loop("Proc C", config.test9_proc_c_n, lambda: procedure_c_local_trajectory(exact, T, zeta, rng_c).n_jumps),
        dtype=int,
    )
    proc_c_p0 = float(np.mean(proc_c_counts == 0))
    proc_c_mean = float(np.mean(proc_c_counts))

    doob_p0_sem = float(np.sqrt(doob_p0 * (1.0 - doob_p0) / len(doob_counts)))
    doob_mean_sem = float(np.std(doob_counts, ddof=1) / np.sqrt(len(doob_counts))) if len(doob_counts) > 1 else 0.0
    proc_b_p0_sem = float(np.sqrt(proc_b_p0 * (1.0 - proc_b_p0) / len(proc_b_counts)))
    proc_b_mean_sem = float(np.std(proc_b_counts, ddof=1) / np.sqrt(len(proc_b_counts))) if len(proc_b_counts) > 1 else 0.0

    passed = bool(
        abs(doob_p0 - exact_p0) <= 4.0 * max(doob_p0_sem, exact_p0_sem) + 0.03
        and abs(doob_mean - exact_mean) <= 4.0 * max(doob_mean_sem, exact_mean_sem) + 0.05
        and abs(proc_b_p0 - exact_p0) <= 4.0 * max(proc_b_p0_sem, exact_p0_sem) + 0.04
        and abs(proc_b_mean - exact_mean) <= 4.0 * max(proc_b_mean_sem, exact_mean_sem) + 0.06
        and abs(proc_c_p0 - exact_p0) > 0.03
        and abs(proc_c_mean - exact_mean) > 0.04
    )

    return {
        "passed": passed,
        "parameters": {
            "L": L,
            "w": w,
            "alpha": alpha,
            "T": T,
            "zeta": zeta,
            "born_n": config.test9_born_n,
            "doob_n": config.test9_doob_n,
            "procedure_b_n": config.test9_proc_b_n,
            "procedure_c_n": config.test9_proc_c_n,
        },
        "metrics": {
            "exact_weighted_p0": exact_p0,
            "doob_p0": doob_p0,
            "procedure_b_p0": proc_b_p0,
            "procedure_c_p0": proc_c_p0,
            "exact_weighted_mean_clicks": exact_mean,
            "doob_mean_clicks": doob_mean,
            "procedure_b_mean_clicks": proc_b_mean,
            "procedure_c_mean_clicks": proc_c_mean,
        },
    }


def run_part6_validation(config: Part6ValidationConfig | None = None) -> dict[str, Any]:
    cfg = config or Part6ValidationConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_runners = [
        ("test_1_zeta1_recovery",                  lambda: _run_test_1(cfg, output_dir)),
        ("test_2_small_zeta_concentration",         lambda: _run_test_2(cfg)),
        ("test_3_single_mode_exact_case",           lambda: _run_test_3(cfg)),
        ("test_4_commuting_case",                   lambda: _run_test_4(cfg)),
        ("test_5_partition_and_moments",            lambda: _run_test_5(cfg)),
        ("test_6_click_count_distribution",         lambda: _run_test_6(cfg, output_dir)),
        ("test_7_entanglement_scaling",             lambda: _run_test_7(cfg, output_dir)),
        ("test_8_conditioned_survival_monotonicity",lambda: _run_test_8(cfg)),
        ("test_9_qs_vs_rzeta",                      lambda: _run_test_9(cfg)),
    ]

    tests: dict[str, Any] = {}
    t_total = time.perf_counter()
    for num, (name, runner) in enumerate(test_runners, 1):
        print(f"\n[{num}/9] {name}", flush=True)
        t0 = time.perf_counter()
        result = runner()
        elapsed = time.perf_counter() - t0
        status = "PASS" if result.get("passed") else "FAIL"
        print(f"  → {status}  ({elapsed:.1f}s)", flush=True)
        tests[name] = result

    total_elapsed = time.perf_counter() - t_total
    print(f"\nall tests completed in {total_elapsed:.1f}s", flush=True)

    return {
        "config": asdict(cfg),
        "tests": tests,
        "all_passed": bool(all(result.get("passed", False) for result in tests.values())),
        "notes": [
            "Trajectory observables under Q_s are estimated from exact ordinary trajectories with weights zeta^N_T where appropriate.",
            "That weighted-Born estimator is mathematically identical to Procedure A for ensemble averages and histograms, but much cheaper than rejection sampling.",
        ],
    }
