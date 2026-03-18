import unittest

import numpy as np

from pps_qj.backends import GaussianStateBackend
from pps_qj.models import free_fermion_gaussian_model, spin_chain_model
from pps_qj.simulator import Simulator
from pps_qj.types import SimulationConfig


class TestGaussianBackend(unittest.TestCase):
    def test_invariants_under_pps(self) -> None:
        """Γ should remain antisymmetric with Γ²=-I after propagation + jump."""
        model = free_fermion_gaussian_model(L=4, w=0.2, gamma=0.8)
        backend = GaussianStateBackend(model=model, Gamma=model.initial_gamma.copy())

        backend.propagate_no_click(0.2)
        rates = backend.channel_rates()
        self.assertTrue(np.all(rates >= 0.0))

        if np.sum(rates) > 0:
            j = int(np.argmax(rates))
            backend.apply_jump(j)

        G = backend.Gamma
        self.assertTrue(np.allclose(G + G.T, 0.0, atol=1e-7))
        I = np.eye(G.shape[0])
        self.assertTrue(np.allclose(-(G @ G), I, atol=5e-5))

    def test_survival_decays(self) -> None:
        """Exact survival probability should decrease monotonically."""
        model = free_fermion_gaussian_model(L=4, w=0.2, gamma=0.8)
        backend = GaussianStateBackend(model=model, Gamma=model.initial_gamma.copy())

        taus = [0.0, 0.1, 0.5, 1.0, 2.0]
        survivals = [backend.survival(tau) for tau in taus]
        for i in range(len(survivals) - 1):
            self.assertGreaterEqual(survivals[i], survivals[i + 1] - 1e-12)
        self.assertAlmostEqual(survivals[0], 1.0, places=10)

    def test_cross_backend_click_statistics(self) -> None:
        """Cross-validate Gaussian and exact backends at L=4 (Remark 81)."""
        sim = Simulator()
        L = 4
        w = 0.3
        gamma = 0.5
        zeta = 0.7
        T = 1.0
        n = 800

        # Exact backend using spin chain model
        exact_model = spin_chain_model(L=L, w=w, gamma=gamma)
        cfg_e = SimulationConfig(
            T=T, zeta=zeta, n_traj=n, seed=77,
            backend="exact", method="pps_mc",
        )
        recs_e = sim.run_ensemble(cfg_e, exact_model)
        mean_e = np.mean([r.n_clicks for r in recs_e])

        # Gaussian backend using free-fermion model
        gauss_model = free_fermion_gaussian_model(L=L, w=w, gamma=gamma)
        cfg_g = SimulationConfig(
            T=T, zeta=zeta, n_traj=n, seed=78,
            backend="gaussian", method="pps_mc",
        )
        recs_g = sim.run_ensemble(cfg_g, gauss_model)
        mean_g = np.mean([r.n_clicks for r in recs_g])

        # The two backends should give similar click statistics
        self.assertLess(abs(mean_e - mean_g), 0.5)


if __name__ == "__main__":
    unittest.main()
