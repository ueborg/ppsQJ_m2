import unittest

import numpy as np

from pps_qj.models import single_projector_model
from pps_qj.observables import acceptance_fraction
from pps_qj.simulator import Simulator
from pps_qj.types import SimulationConfig


class TestEq130AndProcedures(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Simulator()
        q0 = 0.7
        self.q0 = q0
        psi = np.array([np.sqrt(1 - q0), np.sqrt(q0)], dtype=np.complex128)
        self.model = single_projector_model(gamma=1.0, initial_state=psi)

    def test_acceptance_rate_matches_eq130_procedure_a(self) -> None:
        T = 1.4
        zeta = 0.6
        cfg = SimulationConfig(T=T, zeta=zeta, n_traj=2500, seed=101, backend="exact", method="procedure_a")
        recs = self.sim.run_ensemble(cfg, self.model)
        empirical = acceptance_fraction(recs)
        expected = (1.0 - self.q0) + self.q0 * np.exp(-(1.0 - zeta) * T)
        self.assertLess(abs(empirical - expected), 0.04)

    def test_acceptance_rate_matches_eq130_procedure_b(self) -> None:
        T = 1.4
        zeta = 0.6
        cfg = SimulationConfig(T=T, zeta=zeta, n_traj=2500, seed=102, backend="exact", method="procedure_b")
        recs = self.sim.run_ensemble(cfg, self.model)
        empirical = acceptance_fraction(recs)
        expected = (1.0 - self.q0) + self.q0 * np.exp(-(1.0 - zeta) * T)
        self.assertLess(abs(empirical - expected), 0.04)

    def test_distributional_equivalence_abc(self) -> None:
        T = 1.2
        zeta = 0.6
        n = 2500

        cfg_a = SimulationConfig(T=T, zeta=zeta, n_traj=n, seed=201, backend="exact", method="procedure_a")
        cfg_b = SimulationConfig(T=T, zeta=zeta, n_traj=n, seed=202, backend="exact", method="procedure_b")
        cfg_c = SimulationConfig(T=T, zeta=zeta, n_traj=n, seed=203, backend="exact", method="pps_mc")

        ra = self.sim.run_ensemble(cfg_a, self.model)
        rb = self.sim.run_ensemble(cfg_b, self.model)
        rc = self.sim.run_ensemble(cfg_c, self.model)

        ca = np.array([r.n_clicks for r in ra if r.accepted], dtype=int)
        cb = np.array([r.n_clicks for r in rb if r.accepted], dtype=int)
        cc = np.array([r.n_clicks for r in rc], dtype=int)

        kmax = int(max(ca.max(initial=0), cb.max(initial=0), cc.max(initial=0), 1))

        def pmf(c: np.ndarray) -> np.ndarray:
            h = np.bincount(c, minlength=kmax + 1).astype(float)
            return h / h.sum()

        pa = pmf(ca)
        pb = pmf(cb)
        pc = pmf(cc)

        tv_ca = 0.5 * np.abs(pc - pa).sum()
        tv_cb = 0.5 * np.abs(pc - pb).sum()

        self.assertLess(tv_ca, 0.12)
        self.assertLess(tv_cb, 0.12)

    def test_effective_thinning_rate(self) -> None:
        psi = np.array([0.0, 1.0], dtype=np.complex128)
        model = single_projector_model(gamma=1.0, initial_state=psi)
        zeta = 0.35
        cfg = SimulationConfig(T=1.5, zeta=zeta, n_traj=1800, seed=303, backend="exact", method="pps_mc")
        recs = self.sim.run_ensemble(cfg, model)

        total_candidates = sum(r.candidate_count for r in recs)
        total_accepts = sum(r.n_clicks for r in recs)
        ratio = total_accepts / max(total_candidates, 1)
        self.assertLess(abs(ratio - zeta), 0.05)


if __name__ == "__main__":
    unittest.main()
