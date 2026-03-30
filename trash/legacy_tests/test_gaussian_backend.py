import unittest

import numpy as np

from pps_qj.backends import ExactStateBackend, GaussianStateBackend
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

    def test_post_jump_occupation(self) -> None:
        """After a d†d jump, q' must equal 1 (mode occupied) for both backends.

        This discriminates number-operator jumps (d†d, correct: q'=1, Γ'_{ab}=-1)
        from annihilation-operator jumps (d, wrong: q'=0, Γ'_{ab}=+1).
        """
        # --- Gaussian backend ---
        model_g = free_fermion_gaussian_model(L=2, w=0.0, gamma=1.0)
        backend_g = GaussianStateBackend(model=model_g, Gamma=model_g.initial_gamma.copy())

        j = 0
        a, b = model_g.majorana_jump_pairs[j]
        q_pre = 0.5 * (1.0 - backend_g.Gamma[a, b])
        if q_pre > 1e-6:
            backend_g.apply_jump(j)
            q_post = 0.5 * (1.0 - backend_g.Gamma[a, b])
            self.assertAlmostEqual(
                q_post, 1.0, places=10,
                msg=f"Gaussian: post-jump q={q_post:.8f}, expected 1.0 "
                    f"(number-operator jump). Γ'_ab = {backend_g.Gamma[a, b]:.6f}, "
                    f"expected -1.0.",
            )

        # --- Exact backend cross-check ---
        model_e = spin_chain_model(L=2, w=0.0, gamma=1.0)
        backend_e = ExactStateBackend(model=model_e, psi=model_e.initial_state.copy())

        rates_e = backend_e.channel_rates()
        if rates_e[0] > 1e-6:
            backend_e.apply_jump(0)
            op = model_e.jump_ops[0]
            q_post_e = float(np.real(np.vdot(backend_e.psi, (op.conj().T @ op) @ backend_e.psi)))
            q_post_e /= model_e.gamma
            self.assertAlmostEqual(
                q_post_e, 1.0, places=10,
                msg=f"Exact: post-jump q={q_post_e:.8f}, expected 1.0",
            )

    def test_cross_backend_mean_clicks_tight(self) -> None:
        """Cross-backend click mean must agree to within 4 standard errors."""
        sim = Simulator()
        L, w, gamma, zeta, T = 4, 0.3, 0.5, 0.7, 1.0
        n = 2000

        exact_model = spin_chain_model(L=L, w=w, gamma=gamma)
        cfg_e = SimulationConfig(
            T=T, zeta=zeta, n_traj=n, seed=77,
            backend="exact", method="pps_mc",
        )
        recs_e = sim.run_ensemble(cfg_e, exact_model)
        clicks_e = np.array([r.n_clicks for r in recs_e])

        gauss_model = free_fermion_gaussian_model(L=L, w=w, gamma=gamma)
        cfg_g = SimulationConfig(
            T=T, zeta=zeta, n_traj=n, seed=78,
            backend="gaussian", method="pps_mc",
        )
        recs_g = sim.run_ensemble(cfg_g, gauss_model)
        clicks_g = np.array([r.n_clicks for r in recs_g])

        mean_e, mean_g = clicks_e.mean(), clicks_g.mean()
        se = np.sqrt(clicks_e.var() / n + clicks_g.var() / n)
        self.assertLess(
            abs(mean_e - mean_g), 4 * se,
            msg=f"Exact mean={mean_e:.4f}, Gaussian mean={mean_g:.4f}, "
                f"4*SE={4*se:.4f}. Backends disagree beyond statistical noise.",
        )


if __name__ == "__main__":
    unittest.main()
