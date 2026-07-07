"""Regression tests for the safeguarded-Newton waiting-time solver.

Newton (solver_method='newton') solves the integrated-hazard equation with the
analytic derivative Lambda'(t) = -Re Tr(Q^dag K Q) + alpha*N, converging in ~3-4
evals/jump and returning the propagated orbitals (no separate QR).  It is a
STATISTICALLY-equivalent (not bit-identical) sampler -- the accepted waiting time
differs from brentq at the ~xtol level -- so these tests check jump-sequence
preservation, ~xtol covariance agreement, and bias-free cloning observables.
"""
import numpy as np
import pytest
from pps_qj.gaussian_backend import build_gaussian_chain_model, gaussian_born_rule_trajectory


@pytest.mark.parametrize("zeta", [0.2, 0.5, 1.0])
def test_newton_matches_brentq_trajectory(zeta):
    L = 64
    m = build_gaussian_chain_model(L, 0.65, 0.35)
    seed = 271
    rb = gaussian_born_rule_trajectory(m, float(L), np.random.default_rng(seed),
                                       proposal_c=zeta, solver_method="brentq")
    rn = gaussian_born_rule_trajectory(m, float(L), np.random.default_rng(seed),
                                       proposal_c=zeta, solver_method="newton")
    assert rb.n_jumps == rn.n_jumps
    assert np.max(np.abs(rb.final_covariance - rn.final_covariance)) < 1e-4
    U = rn.final_orbitals
    assert np.max(np.abs(U.conj().T @ U - np.eye(U.shape[1]))) < 1e-10


@pytest.mark.parametrize("eps", [1e-8, 1e-9, 1e-10])
def test_newton_eps_hazard_robust(eps):
    L = 64
    m = build_gaussian_chain_model(L, 0.65, 0.35)
    seed = 99
    rref = gaussian_born_rule_trajectory(m, float(L), np.random.default_rng(seed),
                                         proposal_c=0.5, solver_method="brentq")
    re = gaussian_born_rule_trajectory(m, float(L), np.random.default_rng(seed),
                                       proposal_c=0.5, solver_method="newton", eps_hazard=eps)
    assert rref.n_jumps == re.n_jumps


def test_newton_cloning_unbiased():
    from pps_qj.cloning import run_cloning
    L, zeta, N_c, T = 32, 0.5, 8, 8.0
    alpha = 0.35
    dtau = 12.0 / max(2 * alpha * (L - 1), 1e-6)
    m = build_gaussian_chain_model(L, 0.65, alpha)

    def go(solver):
        r = run_cloning(m, zeta, T, N_c, np.random.default_rng(7), delta_tau=dtau,
                        record_entropy=True, proposal_c=zeta,
                        jump_update_method="lowrank", solver_method=solver)
        return r.theta_hat, r.S_mean

    tb, sb = go("brentq")
    tn, sn = go("newton")
    assert abs(tb - tn) < 1e-4
    assert abs(sb - sn) < 1e-4
