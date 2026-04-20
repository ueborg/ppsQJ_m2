from __future__ import annotations

import numpy as np

from pps_qj.backward_pass import ExactBackwardData, GaussianBackwardData
from pps_qj.exact_backend import ExactSpinChainModel, _propagate_unnormalized
from pps_qj.gaussian_backend import (
    GaussianChainModel,
    apply_projective_jump,
    covariance_from_orbitals,
    entanglement_entropy,
    jump_probability,
    orbitals_from_covariance,
    propagate_no_click_orbitals,
    topological_entanglement_entropy,
)
from pps_qj.overlaps import gaussian_overlap
from pps_qj.types import JumpTrajectory, Tolerances


def _bounded_bisection(
    fn,
    target: float,
    left: float,
    right: float,
    tolerances: Tolerances,
) -> float:
    f_left = fn(left) - target
    f_right = fn(right) - target
    if f_left < 0.0 or f_right > 0.0:
        raise ValueError("Bisection interval does not bracket a monotone root")

    while right - left > tolerances.rtol * max(1.0, abs(right)):
        mid = 0.5 * (left + right)
        f_mid = fn(mid) - target
        if abs(f_mid) <= tolerances.atol:
            return mid
        if f_mid > 0.0:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


def conditioned_survival_exact(
    model: ExactSpinChainModel,
    backward_data: ExactBackwardData,
    state: np.ndarray,
    t_start: float,
    dt: float,
    denominator: float | None = None,
) -> float:
    psi_tilde = _propagate_unnormalized(model, state, dt)
    numerator = np.vdot(psi_tilde, backward_data.operator_at(t_start + dt) @ psi_tilde)
    denom = denominator if denominator is not None else backward_data.overlap(t_start, state)
    return float(np.real_if_close(numerator, tol=1_000.0).real / denom)


def conditioned_survival_gaussian(
    model: GaussianChainModel,
    backward_data: GaussianBackwardData,
    orbitals: np.ndarray,
    t_start: float,
    dt: float,
    denominator: float | None = None,
) -> float:
    evolution = propagate_no_click_orbitals(
        orbitals,
        model.h_effective,
        dt,
        alpha=model.alpha,
        n_monitored=len(model.jump_pairs),
    )
    C_t, z_t = backward_data.state_at(t_start + dt)
    numerator = evolution.branch_norm * gaussian_overlap(C_t, evolution.covariance, z_scalar=z_t)
    if denominator is None:
        C_0, z_0 = backward_data.state_at(t_start)
        denominator = gaussian_overlap(C_0, covariance_from_orbitals(orbitals), z_scalar=z_0)
    return float(numerator / denominator)


def doob_exact_trajectory(
    model: ExactSpinChainModel,
    backward_data: ExactBackwardData,
    T: float,
    zeta: float,
    rng: np.random.Generator,
    *,
    tolerances: Tolerances | None = None,
    survival_grid_points: int = 0,
) -> JumpTrajectory:
    tol = tolerances or Tolerances()
    state = np.asarray(model.initial_state, dtype=np.complex128).copy()
    t = 0.0
    jump_times: list[float] = []
    channels: list[int] = []
    diagnostics: dict[str, object] = {"conditioned_survival_segments": []}

    while t < T:
        denominator = backward_data.overlap(t, state)
        if denominator <= 0.0:
            raise RuntimeError("Encountered non-positive Doob denominator in exact backend")
        r = float(rng.uniform(0.0, 1.0))
        max_dt = T - t
        survival_fn = lambda dt: conditioned_survival_exact(
            model,
            backward_data,
            state,
            t,
            dt,
            denominator=denominator,
        )

        segment_info = {
            "t_start": float(t),
            "denominator": float(denominator),
            "uniform_threshold": float(r),
            "max_dt": float(max_dt),
        }
        terminal_survival = float(survival_fn(max_dt))
        segment_info["terminal_survival"] = terminal_survival
        if survival_grid_points > 1:
            grid = np.linspace(0.0, max_dt, survival_grid_points)
            segment_info["times"] = list(t + grid)
            segment_info["values"] = [survival_fn(float(dt)) for dt in grid]

        if terminal_survival > r:
            segment_info["realized_dt"] = float(max_dt)
            segment_info["realized_survival"] = terminal_survival
            segment_info["jumped"] = False
            diagnostics["conditioned_survival_segments"].append(segment_info)
            state = _propagate_unnormalized(model, state, max_dt)
            state = state / np.linalg.norm(state)
            t = T
            break

        dt = _bounded_bisection(survival_fn, r, 0.0, max_dt, tol)
        segment_info["realized_dt"] = float(dt)
        segment_info["realized_survival"] = float(survival_fn(dt))
        segment_info["jumped"] = True
        diagnostics["conditioned_survival_segments"].append(segment_info)
        psi_tilde = _propagate_unnormalized(model, state, dt)
        pre_state = psi_tilde / np.linalg.norm(psi_tilde)
        t += dt

        overlap_pre = backward_data.overlap(t, pre_state)
        rates = []
        post_states: list[np.ndarray] = []
        for projector in model.jump_projectors:
            weight = float(np.real(np.vdot(pre_state, projector @ pre_state)))
            if weight <= 1e-14:
                rates.append(0.0)
                post_states.append(pre_state)
                continue
            post_state = projector @ pre_state
            post_state = post_state / np.linalg.norm(post_state)
            post_states.append(post_state)
            overlap_post = backward_data.overlap(t, post_state)
            rates.append(zeta * 2.0 * model.alpha * weight * overlap_post / overlap_pre)

        rates = np.asarray(rates, dtype=np.float64)
        if np.sum(rates) <= 0.0:
            state = pre_state
            continue

        channel = int(rng.choice(len(rates), p=rates / np.sum(rates)))
        state = post_states[channel]
        jump_times.append(t)
        channels.append(channel)

    return JumpTrajectory(
        jump_times=jump_times,
        channels=channels,
        final_time=t,
        final_state=state,
        diagnostics=diagnostics,
    )


def doob_gaussian_trajectory(
    model: GaussianChainModel,
    backward_data: GaussianBackwardData,
    T: float,
    zeta: float,
    rng: np.random.Generator,
    *,
    tolerances: Tolerances | None = None,
    survival_grid_points: int = 0,
) -> JumpTrajectory:
    tol = tolerances or Tolerances()
    orbitals = np.asarray(model.orbitals0, dtype=np.complex128).copy()
    t = 0.0
    jump_times: list[float] = []
    channels: list[int] = []
    diagnostics: dict[str, object] = {"conditioned_survival_segments": []}

    while t < T:
        gamma_now = covariance_from_orbitals(orbitals)
        C_now, z_now = backward_data.state_at(t)
        denominator = gaussian_overlap(C_now, gamma_now, z_scalar=z_now)
        if denominator <= 0.0:
            raise RuntimeError("Encountered non-positive Doob denominator in Gaussian backend")

        r = float(rng.uniform(0.0, 1.0))
        max_dt = T - t
        survival_fn = lambda dt: conditioned_survival_gaussian(
            model,
            backward_data,
            orbitals,
            t,
            dt,
            denominator=denominator,
        )

        segment_info = {
            "t_start": float(t),
            "denominator": float(denominator),
            "uniform_threshold": float(r),
            "max_dt": float(max_dt),
        }
        terminal_survival = float(survival_fn(max_dt))
        segment_info["terminal_survival"] = terminal_survival
        if survival_grid_points > 1:
            grid = np.linspace(0.0, max_dt, survival_grid_points)
            segment_info["times"] = list(t + grid)
            segment_info["values"] = [survival_fn(float(dt)) for dt in grid]

        if terminal_survival > r:
            segment_info["realized_dt"] = float(max_dt)
            segment_info["realized_survival"] = terminal_survival
            segment_info["jumped"] = False
            diagnostics["conditioned_survival_segments"].append(segment_info)
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

        dt = _bounded_bisection(survival_fn, r, 0.0, max_dt, tol)
        segment_info["realized_dt"] = float(dt)
        segment_info["realized_survival"] = float(survival_fn(dt))
        segment_info["jumped"] = True
        diagnostics["conditioned_survival_segments"].append(segment_info)
        evolution = propagate_no_click_orbitals(
            orbitals,
            model.h_effective,
            dt,
            alpha=model.alpha,
            n_monitored=len(model.jump_pairs),
        )
        pre_orbitals = evolution.orbitals_normalized
        pre_covariance = evolution.covariance
        t += dt

        C_jump, z_jump = backward_data.state_at(t)
        overlap_pre = gaussian_overlap(C_jump, pre_covariance, z_scalar=z_jump)
        rates = []
        post_orbitals: list[np.ndarray] = []
        for jump_pair in model.jump_pairs:
            q = jump_probability(pre_covariance, jump_pair)
            if q <= 1e-14:
                rates.append(0.0)
                post_orbitals.append(pre_orbitals)
                continue
            _, post_covariance = apply_projective_jump(pre_covariance, jump_pair)
            overlap_post = gaussian_overlap(C_jump, post_covariance, z_scalar=z_jump)
            rates.append(zeta * 2.0 * model.alpha * q * overlap_post / overlap_pre)
            post_orbitals.append(orbitals_from_covariance(post_covariance))

        rates = np.asarray(rates, dtype=np.float64)
        if np.sum(rates) <= 0.0:
            orbitals = pre_orbitals
            continue

        channel = int(rng.choice(len(rates), p=rates / np.sum(rates)))
        orbitals = post_orbitals[channel]
        jump_times.append(t)
        channels.append(channel)

    return JumpTrajectory(
        jump_times=jump_times,
        channels=channels,
        final_time=t,
        final_state=orbitals,
        diagnostics=diagnostics,
    )


def gaussian_doob_trajectory_observables(
    model: GaussianChainModel,
    backward: GaussianBackwardData,
    T: float,
    zeta: float,
    rng: np.random.Generator,
    **kwargs,
) -> dict:
    """Run a Gaussian Doob trajectory and return entropy observables.

    Returns a dict with keys ``"entropy"``, ``"topo_entropy"``, ``"n_clicks"``,
    and ``"B_L"`` (the product of entanglement and topological entropies).
    """
    traj = doob_gaussian_trajectory(model, backward, T, zeta, rng, **kwargs)
    cov = covariance_from_orbitals(np.asarray(traj.final_state))
    ent = entanglement_entropy(cov, model.L // 2)
    topo = topological_entanglement_entropy(cov)
    return {
        "entropy": ent,
        "topo_entropy": topo,
        "n_clicks": traj.n_jumps,
        "B_L": ent * topo,
    }
