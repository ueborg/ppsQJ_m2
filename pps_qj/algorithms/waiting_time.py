from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pps_qj.backends.base import StateBackend
from pps_qj.core.numerics import bracket_and_bisect, safe_probs
from pps_qj.types import Tolerances, TrajectoryRecord


@dataclass
class RunContext:
    backend: StateBackend
    rng: np.random.Generator
    T: float
    zeta: float
    tol: Tolerances


def sample_waiting_time(ctx: RunContext) -> float:
    r = float(ctx.rng.uniform(0.0, 1.0))
    tau_a = ctx.backend.analytic_waiting_time(r)
    if tau_a is not None:
        return float(tau_a)

    try:
        return bracket_and_bisect(
            fn=ctx.backend.survival,
            target=r,
            x0=0.0,
            x1=1.0,
            tol=ctx.tol,
        )
    except RuntimeError:
        # No root in finite time (dark-state component / no-jump tail).
        return float("inf")


def sample_channel(backend: StateBackend, rng: np.random.Generator) -> int:
    rates = backend.channel_rates()
    probs = safe_probs(rates)
    cdf = np.cumsum(probs)
    u = float(rng.uniform(0.0, 1.0))
    j = int(np.searchsorted(cdf, u, side="right"))
    return min(j, len(rates) - 1)


def run_waiting_time_trajectory(ctx: RunContext) -> TrajectoryRecord:
    t = 0.0
    times: list[float] = [0.0]
    accepted_jump_times: list[float] = []
    candidate_jump_times: list[float] = []
    channels: list[int] = []
    purity_trace: list[float] = [ctx.backend.purity()]

    ctx.backend.normalize()

    while t < ctx.T:
        tau_w = sample_waiting_time(ctx)
        if not np.isfinite(tau_w) or t + tau_w >= ctx.T:
            ctx.backend.propagate_no_click(max(ctx.T - t, 0.0))
            t = ctx.T
            times.append(t)
            purity_trace.append(ctx.backend.purity())
            break

        ctx.backend.propagate_no_click(tau_w)
        t += tau_w
        times.append(t)
        candidate_jump_times.append(t)

        j = sample_channel(ctx.backend, ctx.rng)
        ctx.backend.apply_jump(j)
        channels.append(j)
        accepted_jump_times.append(t)
        purity_trace.append(ctx.backend.purity())

    return TrajectoryRecord(
        times=times,
        accepted_jump_times=accepted_jump_times,
        candidate_jump_times=candidate_jump_times,
        channels=channels,
        n_clicks=len(accepted_jump_times),
        accepted=True,
        observables={"purity_trace": purity_trace},
        final_time=t,
        candidate_count=len(candidate_jump_times),
    )


def run_pps_mc_trajectory(ctx: RunContext) -> TrajectoryRecord:
    t = 0.0
    times: list[float] = [0.0]
    accepted_jump_times: list[float] = []
    candidate_jump_times: list[float] = []
    channels: list[int] = []
    purity_trace: list[float] = [ctx.backend.purity()]

    ctx.backend.normalize()

    while t < ctx.T:
        tau_w = sample_waiting_time(ctx)
        if not np.isfinite(tau_w) or t + tau_w >= ctx.T:
            ctx.backend.propagate_no_click(max(ctx.T - t, 0.0))
            t = ctx.T
            times.append(t)
            purity_trace.append(ctx.backend.purity())
            break

        ctx.backend.propagate_no_click(tau_w)
        t += tau_w
        times.append(t)
        candidate_jump_times.append(t)

        if ctx.zeta >= 1.0:
            accepted = True
        elif ctx.zeta <= 0.0:
            accepted = False
        else:
            u = float(ctx.rng.uniform(0.0, 1.0))
            accepted = u <= ctx.zeta

        if accepted:
            j = sample_channel(ctx.backend, ctx.rng)
            ctx.backend.apply_jump(j)
            channels.append(j)
            accepted_jump_times.append(t)

        purity_trace.append(ctx.backend.purity())

    return TrajectoryRecord(
        times=times,
        accepted_jump_times=accepted_jump_times,
        candidate_jump_times=candidate_jump_times,
        channels=channels,
        n_clicks=len(accepted_jump_times),
        accepted=True,
        observables={"purity_trace": purity_trace},
        final_time=t,
        candidate_count=len(candidate_jump_times),
    )
