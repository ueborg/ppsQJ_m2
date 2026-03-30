from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pps_qj.models import free_fermion_gaussian_model, single_projector_model
from pps_qj.observables import acceptance_fraction
from pps_qj.simulator import Simulator
from pps_qj.types import SimulationConfig


def main() -> None:
    sim = Simulator()

    q0 = 0.7
    psi = np.array([np.sqrt(1.0 - q0), np.sqrt(q0)], dtype=np.complex128)
    model = single_projector_model(gamma=1.0, initial_state=psi)

    cfg_a = SimulationConfig(T=1.2, zeta=0.6, n_traj=2000, seed=10, backend="exact", method="procedure_a")
    recs_a = sim.run_ensemble(cfg_a, model)
    empirical = acceptance_fraction(recs_a)
    expected = (1.0 - q0) + q0 * np.exp(-(1.0 - cfg_a.zeta) * cfg_a.T)
    print(f"Procedure A acceptance: empirical={empirical:.4f}, expected={expected:.4f}")

    cfg_c = SimulationConfig(T=1.2, zeta=0.6, n_traj=2000, seed=11, backend="exact", method="pps_mc")
    recs_c = sim.run_ensemble(cfg_c, model)
    print(f"PPS-MC mean accepted clicks: {np.mean([r.n_clicks for r in recs_c]):.4f}")

    model_g = free_fermion_gaussian_model(L=16, w=0.2, gamma=0.5)
    cfg_g = SimulationConfig(T=0.5, zeta=0.7, n_traj=100, seed=21, backend="gaussian", method="pps_mc")
    recs_g = sim.run_ensemble(cfg_g, model_g)
    print(f"Gaussian backend mean accepted clicks: {np.mean([r.n_clicks for r in recs_g]):.4f}")


if __name__ == "__main__":
    main()
