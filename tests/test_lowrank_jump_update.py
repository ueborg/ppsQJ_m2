"""Regression tests for the low-rank projective-jump orbital update.

The active-subspace update (_lowrank_jump_orbital_update) replaces the per-jump
O(L^3) eigendecomposition in orbitals_from_covariance with an O(L^2) update,
exploiting that a projective Majorana-parity jump changes the occupied projector
by rank <= 4.  It must reproduce the eigh path to machine precision.
"""
import numpy as np
import pytest
from pps_qj.gaussian_backend import (
    build_gaussian_chain_model, gaussian_born_rule_trajectory,
    orbitals_from_covariance, covariance_from_orbitals,
    _lowrank_jump_orbital_update,
)


@pytest.mark.parametrize("L", [32, 64, 96])
def test_lowrank_single_jump_matches_eigh(L):
    model = build_gaussian_chain_model(L, 0.65, 0.35)
    r = gaussian_born_rule_trajectory(model, 8.0, np.random.default_rng(L), proposal_c=0.5)
    cov = r.final_covariance
    U = orbitals_from_covariance(cov)
    for bond in range(0, L - 1, max(1, (L - 1) // 6)):
        jp = model.jump_pairs[bond]
        a, b = jp
        if abs(1.0 - cov[a, b]) < 1e-10:
            continue
        U_lr, Gp = _lowrank_jump_orbital_update(U, cov, jp)
        U_ref = orbitals_from_covariance(Gp)
        assert np.max(np.abs(U_lr @ U_lr.conj().T - U_ref @ U_ref.conj().T)) < 1e-10
        assert np.max(np.abs(covariance_from_orbitals(U_lr) - Gp)) < 1e-9
        assert np.max(np.abs(U_lr.conj().T @ U_lr - np.eye(U_lr.shape[1]))) < 1e-10


@pytest.mark.parametrize("zeta", [0.2, 0.5, 1.0])
def test_lowrank_trajectory_same_seed(zeta):
    L = 48
    model = build_gaussian_chain_model(L, 0.65, 0.35)
    seed = 314
    r_e = gaussian_born_rule_trajectory(model, float(L), np.random.default_rng(seed),
                                        proposal_c=zeta, jump_update_method="eigh")
    r_l = gaussian_born_rule_trajectory(model, float(L), np.random.default_rng(seed),
                                        proposal_c=zeta, jump_update_method="lowrank",
                                        refresh_every=100)
    assert r_e.n_jumps == r_l.n_jumps
    assert np.max(np.abs(r_e.final_covariance - r_l.final_covariance)) < 1e-10
    assert abs(r_e.Lambda - r_l.Lambda) < 1e-9
    U = r_l.final_orbitals
    assert np.max(np.abs(U.conj().T @ U - np.eye(U.shape[1]))) < 1e-10


def test_lowrank_cloning_population_unbiased():
    """Cloning population (theta/S/CMI) is bit-identical eigh vs lowrank."""
    from pps_qj.cloning import run_cloning
    from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
    L, zeta, N_c, T = 32, 0.5, 8, 8.0
    alpha = 0.35
    dtau = 12.0 / max(2 * alpha * (L - 1), 1e-6)
    model = build_gaussian_chain_model(L, 0.65, alpha)

    def go(method):
        res = run_cloning(model, zeta, T, N_c, np.random.default_rng(7), delta_tau=dtau,
                          record_entropy=True, proposal_c=zeta,
                          jump_update_method=method, refresh_every=100)
        cmi = np.nanmean(_batched_compute_B_L(res.final_covs, L)["CMI"])
        return res.theta_hat, res.S_mean, cmi

    t_e, s_e, c_e = go("eigh")
    t_l, s_l, c_l = go("lowrank")
    assert abs(t_e - t_l) < 1e-10
    assert abs(s_e - s_l) < 1e-10
    assert (np.isnan(c_e) and np.isnan(c_l)) or abs(c_e - c_l) < 1e-10
