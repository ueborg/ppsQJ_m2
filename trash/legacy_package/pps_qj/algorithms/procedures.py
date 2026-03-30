from __future__ import annotations

import numpy as np

from pps_qj.algorithms.waiting_time import RunContext, run_waiting_time_trajectory, sample_channel, sample_waiting_time
from pps_qj.types import TrajectoryRecord


def run_procedure_a(ctx: RunContext) -> TrajectoryRecord:
    # Generate full Born-rule trajectory first.
    born = run_waiting_time_trajectory(
        RunContext(
            backend=ctx.backend,
            rng=ctx.rng,
            T=ctx.T,
            zeta=1.0,
            tol=ctx.tol,
        )
    )

    n = born.n_clicks
    if n == 0:
        born.accepted = True
        born.observables["accept_probability"] = 1.0
        return born

    flips = ctx.rng.uniform(0.0, 1.0, size=n)
    accepted = bool(np.all(flips <= ctx.zeta))
    born.accepted = accepted
    born.observables["accept_probability"] = float(ctx.zeta**n)
    return born


def run_procedure_b(ctx: RunContext) -> TrajectoryRecord:
    # Sequential conditioning: abort on first rejected click.
    t = 0.0
    times: list[float] = [0.0]
    accepted_jump_times: list[float] = []
    candidate_jump_times: list[float] = []
    channels: list[int] = []
    purity_trace: list[float] = [ctx.backend.purity()]

    accepted = True
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

        u = float(ctx.rng.uniform(0.0, 1.0))
        if u > ctx.zeta:
            accepted = False
            purity_trace.append(ctx.backend.purity())
            break

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
        accepted=accepted and (t >= ctx.T),
        observables={"purity_trace": purity_trace},
        final_time=t,
        candidate_count=len(candidate_jump_times),
    )
