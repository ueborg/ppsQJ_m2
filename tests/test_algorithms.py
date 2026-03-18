import unittest

import numpy as np

from pps_qj.models import single_projector_model
from pps_qj.observables import average_purity
from pps_qj.simulator import Simulator
from pps_qj.types import SimulationConfig


class TestAlgorithms(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = Simulator()
        psi = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        self.model = single_projector_model(gamma=1.0, initial_state=psi)

    def test_purity_preserved(self) -> None:
        cfg = SimulationConfig(T=1.0, zeta=0.4, n_traj=200, seed=7, backend="exact", method="pps_mc")
        recs = self.sim.run_ensemble(cfg, self.model)
        avg = average_purity(recs)
        self.assertGreater(avg, 1.0 - 1e-10)

    def test_zeta_one_reduces_to_standard_mc(self) -> None:
        cfg_std = SimulationConfig(T=1.0, zeta=1.0, n_traj=1, seed=11, backend="exact", method="waiting_time_mc")
        cfg_pps = SimulationConfig(T=1.0, zeta=1.0, n_traj=1, seed=11, backend="exact", method="pps_mc")

        r_std = self.sim.run_trajectory(cfg_std, self.model)
        r_pps = self.sim.run_trajectory(cfg_pps, self.model)

        self.assertEqual(r_std.n_clicks, r_pps.n_clicks)
        self.assertTrue(np.allclose(r_std.accepted_jump_times, r_pps.accepted_jump_times))
        self.assertEqual(r_std.channels, r_pps.channels)

    def test_zeta_zero_no_accepted_jumps(self) -> None:
        cfg = SimulationConfig(T=1.2, zeta=0.0, n_traj=40, seed=44, backend="exact", method="pps_mc")
        recs = self.sim.run_ensemble(cfg, self.model)
        self.assertTrue(all(r.n_clicks == 0 for r in recs))


if __name__ == "__main__":
    unittest.main()
