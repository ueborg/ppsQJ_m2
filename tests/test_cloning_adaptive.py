import numpy as np

from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.cloning_adaptive import run_cloning_adaptive


def test_zeta_one_never_resamples_and_keeps_uniform_weights():
    model = build_gaussian_chain_model(4, 0.6, 0.4)
    r = run_cloning_adaptive(
        model, 1.0, T_total=0.5, N_c=8,
        rng=np.random.default_rng(7), delta_tau=0.1,
        proposal_c=1.0, resampling_mode="adaptive",
        ess_threshold=0.9, jump_update_method="eigh",
        solver_method="brentq",
    )
    assert r.n_resampling_events == 0
    assert np.allclose(r.final_weights, 1.0 / 8.0)
    assert np.isclose(r.eff_sample_size, 8.0)
    assert np.isclose(r.root_genealogical_ess, 8.0)


def test_adaptive_target_normalization_is_finite():
    model = build_gaussian_chain_model(4, 0.65, 0.35)
    r = run_cloning_adaptive(
        model, 0.55, T_total=0.5, N_c=8,
        rng=np.random.default_rng(11), delta_tau=0.1,
        proposal_c=0.55, resampling_mode="adaptive",
        ess_threshold=0.9, jump_update_method="eigh",
        solver_method="brentq",
    )
    assert np.isfinite(r.theta_hat)
    assert np.isclose(np.sum(r.final_weights), 1.0)
    assert 1.0 <= r.eff_sample_size <= 8.0 + 1e-12
    assert 1.0 <= r.root_genealogical_ess <= 8.0 + 1e-12
