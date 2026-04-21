from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from pps_qj.backward_pass import ExactBackwardData, GaussianBackwardData
from pps_qj.exact_backend import ExactSpinChainModel, _propagate_unnormalized
from pps_qj.gaussian_backend import (
    GaussianChainModel,
    apply_projective_jump,
    covariance_from_orbitals,
    entanglement_entropy,
    orbitals_from_covariance,
    propagate_no_click_orbitals,
    topological_entanglement_entropy,
)
from pps_qj.overlaps import gaussian_overlap, gaussian_post_jump_overlap, log_gaussian_overlap, log_gaussian_overlap_from_orbitals
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
    max_jumps: int | None = None,
) -> JumpTrajectory:
    """Gaussian Doob trajectory with log-space survival arithmetic.

    Uses log-space throughout to avoid underflow when z_scalar is extremely
    small (e.g. 10^-100 at small zeta / large L). The survival function
    S(dt) = exp(log_S(dt)) is still monotone, so brentq applies unchanged.

    Parameters
    ----------
    max_jumps:
        Safety guard against Gaussian-closure breakdown. When the approximation
        is poor the conditioned survival function can collapse to near-zero
        immediately, causing dt_star ≈ 0 on every iteration and an effectively
        non-terminating loop. If the accumulated jump count exceeds max_jumps
        the trajectory is aborted early and flagged with ``degenerate=True``.
        Defaults to ``5 * len(model.jump_pairs)``, which is a generous cap
        well above any physically meaningful jump count at typical T and alpha.
    """
    n_monitored = len(model.jump_pairs)
    n = 2 * model.L
    if max_jumps is None:
        max_jumps = 5 * n_monitored

    # Pre-diagonalise h_effective once for the whole trajectory.
    h_eff = np.asarray(model.h_effective, dtype=np.complex128)
    evals, V = np.linalg.eig(h_eff)
    V_inv = np.linalg.inv(V)

    # Precompute jump-pair index arrays for vectorised q_j extraction.
    _jp = model.jump_pairs
    _ja = np.array([p[0] for p in _jp], dtype=np.intp)
    _jb = np.array([p[1] for p in _jp], dtype=np.intp)

    # Born-rule expected inter-jump interval: 1 / (n_monitored * 2*alpha).
    # Under the tilted measure with zeta < 1, jumps are suppressed so intervals
    # should be LONGER than this. We detect Gaussian-closure breakdown by
    # checking whether dt_star is consistently shorter than a fraction of this
    # Born-rule scale — which should never happen for zeta <= 1.
    _dt_born = 1.0 / (n_monitored * 2.0 * model.alpha + 1e-12)
    _dt_tiny = _dt_born / 10.0   # 10x shorter than Born-rule is already a red flag
    _MAX_CONSEC_TINY = 3          # abort after this many consecutive tiny waits
    _consec_tiny = 0

    orbitals = np.asarray(model.orbitals0, dtype=np.complex128).copy()
    t = 0.0
    jump_times: list[float] = []
    channels: list[int] = []
    diagnostics: dict[str, object] = {"conditioned_survival_segments": [], "degenerate": False}

    while t < T:
        # Hard cap on total jumps.
        if len(jump_times) >= max_jumps:
            diagnostics["degenerate"] = True
            break
        # Backward orbitals W and log_z at current time — O(L²) lookup.
        W_now, log_z_now = backward_data.orbitals_at(t)
        # log_denom = log(Tr(G rho)) using L×L det instead of 2L×2L det.
        log_denom = log_gaussian_overlap_from_orbitals(W_now, orbitals, log_z_now)
        if not np.isfinite(log_denom):
            raise RuntimeError(
                f"Non-finite log-denominator at t={t:.4f}: log_denom={log_denom}"
            )

        r = float(rng.uniform(0.0, 1.0))
        log_r = float(np.log(r)) if r > 0.0 else -np.inf
        max_dt = T - t
        coeffs = V_inv @ orbitals  # (2L, L) — reused in survival + propagation

        def _log_survival(dt: float) -> float:
            if dt <= 0.0:
                return 0.0
            exp_d = np.exp(evals * dt)
            orbs_tilde = V @ (exp_d[:, None] * coeffs)
            q_mat, r_mat = np.linalg.qr(orbs_tilde, mode="reduced")
            diag_abs = np.abs(np.diag(r_mat))
            if np.any(diag_abs <= 1e-300):
                return -np.inf
            # log|det M(dt) V₀| from QR R-factor — already computed.
            log_branch = float(np.sum(np.log(diag_abs))) - model.alpha * n_monitored * dt
            # Backward orbitals and log_z at t+dt — O(L²) interpolation.
            W_t, log_z_t = backward_data.orbitals_at(t + dt)
            # L×L det replaces 2L×2L det — 8× cheaper at L=64.
            log_ov_t = log_gaussian_overlap_from_orbitals(W_t, q_mat, log_z_t)
            return log_branch + log_ov_t - log_denom

        segment_info: dict[str, object] = {
            "t_start": float(t),
            "log_denominator": float(log_denom),
            "uniform_threshold": float(r),
            "max_dt": float(max_dt),
        }
        log_terminal = _log_survival(max_dt)
        segment_info["terminal_survival"] = float(np.exp(log_terminal))

        if survival_grid_points > 1:
            grid = np.linspace(0.0, max_dt, survival_grid_points)
            segment_info["times"] = list(t + grid)
            segment_info["values"] = [float(np.exp(_log_survival(float(s)))) for s in grid]

        if log_terminal > log_r:
            # No jump — propagate to T.
            segment_info["jumped"] = False
            diagnostics["conditioned_survival_segments"].append(segment_info)
            exp_d = np.exp(evals * max_dt)
            orbs_tilde = V @ (exp_d[:, None] * coeffs)
            q_mat, _ = np.linalg.qr(orbs_tilde, mode="reduced")
            orbitals = q_mat
            t = T
            break

        # Find jump time: brentq on log_survival - log_r (still monotone).
        try:
            dt_star = brentq(
                lambda dt: _log_survival(dt) - log_r,
                0.0, max_dt,
                xtol=1e-8, maxiter=50, full_output=False,
            )
        except (ValueError, RuntimeError):
            # ValueError: bracket invalid (survival non-monotone at endpoints).
            # RuntimeError: maxiter exceeded (survival non-monotone throughout).
            # Both indicate Gaussian closure breakdown — mark degenerate.
            diagnostics["degenerate"] = True
            break

        # Fast degeneracy detector: if dt_star is far below the Born-rule mean
        # inter-jump interval for several consecutive jumps, the Gaussian
        # closure has collapsed and the trajectory will never reach T.
        if dt_star < _dt_tiny:
            _consec_tiny += 1
            if _consec_tiny >= _MAX_CONSEC_TINY:
                diagnostics["degenerate"] = True
                break
        else:
            _consec_tiny = 0

        segment_info["realized_dt"] = float(dt_star)
        segment_info["jumped"] = True
        diagnostics["conditioned_survival_segments"].append(segment_info)

        # Propagate to jump time — reuses coeffs.
        exp_d_star = np.exp(evals * dt_star)
        orbs_tilde = V @ (exp_d_star[:, None] * coeffs)
        pre_orbitals, _ = np.linalg.qr(orbs_tilde, mode="reduced")
        pre_covariance = covariance_from_orbitals(pre_orbitals)
        t += dt_star

        # Channel weights — orbital-based L×L det for all overlaps.
        W_jump, log_z_jump = backward_data.orbitals_at(t)
        log_ov_pre = log_gaussian_overlap_from_orbitals(
            W_jump, pre_orbitals, log_z_jump
        )

        probs_q = np.clip(0.5 * (1.0 - pre_covariance[_ja, _jb]), 0.0, 1.0)
        log_rates: list[float] = []
        post_orbitals_list: list[np.ndarray] = []
        for i, jump_pair in enumerate(model.jump_pairs):
            q = float(probs_q[i])
            if q <= 1e-14:
                log_rates.append(-np.inf)
                post_orbitals_list.append(pre_orbitals)
                continue
            _, post_covariance = apply_projective_jump(pre_covariance, jump_pair)
            post_orbs = orbitals_from_covariance(post_covariance)
            log_ov_post = log_gaussian_overlap_from_orbitals(
                W_jump, post_orbs, log_z_jump
            )
            log_rates.append(
                np.log(zeta * 2.0 * model.alpha * q) + log_ov_post - log_ov_pre
            )
            post_orbitals_list.append(post_orbs)

        log_rates_arr = np.asarray(log_rates, dtype=np.float64)
        finite_mask = np.isfinite(log_rates_arr)
        if not np.any(finite_mask):
            # All channel weights non-finite: Gaussian closure has broken down
            # at this point in the trajectory. Continuing is meaningless —
            # the state cannot be evolved forward reliably. Mark degenerate.
            diagnostics["degenerate"] = True
            break

        # Numerically stable softmax-style normalisation.
        log_rates_arr[~finite_mask] = -np.inf
        log_max = log_rates_arr[finite_mask].max()
        weights = np.where(finite_mask, np.exp(log_rates_arr - log_max), 0.0)
        weights /= weights.sum()

        channel = int(rng.choice(n_monitored, p=weights))
        orbitals = post_orbitals_list[channel]
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
