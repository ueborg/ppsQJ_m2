from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pps_qj.algorithms import (
    RunContext,
    run_pps_mc_trajectory,
    run_procedure_a,
    run_procedure_b,
    run_waiting_time_trajectory,
)
from pps_qj.backends import ExactStateBackend, GaussianStateBackend, StateBackend
from pps_qj.core.random import make_rng, spawn_rng
from pps_qj.types import ModelSpec, SimulationConfig, TrajectoryRecord


@dataclass
class Simulator:
    def _build_backend(self, model: ModelSpec, config: SimulationConfig) -> StateBackend:
        if config.backend == "exact":
            if model.H is None:
                raise ValueError("Exact backend requires model.H")
            dim = model.H.shape[0]
            psi0 = model.initial_state
            if psi0 is None:
                psi0 = np.zeros(dim, dtype=np.complex128)
                psi0[0] = 1.0
            return ExactStateBackend(model=model, psi=np.asarray(psi0, dtype=np.complex128))

        if config.backend == "gaussian":
            g0 = model.initial_gamma
            if g0 is None:
                n = 2 * model.L
                g0 = np.zeros((n, n), dtype=np.float64)
                for j in range(model.L):
                    a = 2 * j
                    b = 2 * j + 1
                    g0[a, b] = -1.0
                    g0[b, a] = 1.0
            return GaussianStateBackend(model=model, Gamma=np.asarray(g0, dtype=np.float64))

        raise ValueError(f"Unknown backend: {config.backend}")

    def run_trajectory(
        self,
        config: SimulationConfig,
        model: ModelSpec,
        rng: np.random.Generator | None = None,
    ) -> TrajectoryRecord:
        if not (0.0 <= config.zeta <= 1.0):
            raise ValueError("zeta must lie in [0,1]")

        if rng is None:
            rng = make_rng(config.seed)

        backend = self._build_backend(model, config)
        ctx = RunContext(
            backend=backend,
            rng=rng,
            T=config.T,
            zeta=config.zeta,
            tol=config.tolerances,
        )

        if config.method == "waiting_time_mc":
            return run_waiting_time_trajectory(ctx)
        if config.method == "pps_mc":
            return run_pps_mc_trajectory(ctx)
        if config.method == "procedure_a":
            return run_procedure_a(ctx)
        if config.method == "procedure_b":
            return run_procedure_b(ctx)

        raise ValueError(f"Unknown method: {config.method}")

    def run_ensemble(self, config: SimulationConfig, model: ModelSpec) -> list[TrajectoryRecord]:
        rng = make_rng(config.seed)
        out: list[TrajectoryRecord] = []
        for _ in range(config.n_traj):
            trng = spawn_rng(rng)
            out.append(self.run_trajectory(config, model, rng=trng))
        return out
