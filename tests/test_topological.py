"""Tests for pps_qj.observables.topological and pps_qj.backward_pass_io."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pps_qj.backward_pass import run_gaussian_backward_pass
from pps_qj.backward_pass_io import load_backward_pass, save_backward_pass
from pps_qj.gaussian_backend import (
    build_gaussian_chain_model,
    entanglement_entropy,
    gaussian_born_rule_trajectory,
    neel_covariance,
)
from pps_qj.observables.topological import (
    compute_all_observables,
    dual_topological_entropy,
    subsystem_entropy,
    topological_entropy,
)


# --------------------------------------------------------------------------
# subsystem_entropy + topological_entropy
# --------------------------------------------------------------------------

def test_neel_product_state_has_zero_subsystem_entropy():
    """Neel covariance is a pure product state: every subregion has S = 0."""
    L = 8
    gamma = neel_covariance(L)
    # Any single site, any contiguous region: entropy = 0.
    for region in ([1], [1, 2, 3], list(range(1, L // 2 + 1)), list(range(1, L + 1))):
        assert subsystem_entropy(gamma, region) == pytest.approx(0.0, abs=1e-10)


def test_neel_product_state_has_zero_topological_entropy():
    L = 8
    gamma = neel_covariance(L)
    S_top = topological_entropy(gamma, L)
    assert S_top == pytest.approx(0.0, abs=1e-10)


def test_topological_entropy_nan_when_L_not_divisible_by_4():
    L = 10  # not divisible by 4
    gamma = neel_covariance(L)
    assert np.isnan(topological_entropy(gamma, L))


def test_subsystem_entropy_matches_half_cut_entropy():
    """For L=8, subsystem_entropy(half) must match gaussian_backend's
    entanglement_entropy on an entangled state."""
    L = 8
    model = build_gaussian_chain_model(L=L, w=1.0, alpha=0.3)
    rng = np.random.default_rng(12345)
    # Evolve a short Born-rule trajectory to get a non-trivial covariance.
    traj = gaussian_born_rule_trajectory(model, T=2.0, rng=rng)
    gamma = np.asarray(traj.final_covariance)

    half_sites = list(range(1, L // 2 + 1))  # 1-indexed
    S_ours = subsystem_entropy(gamma, half_sites)
    S_ref = entanglement_entropy(gamma, L // 2)
    assert S_ours == pytest.approx(S_ref, abs=1e-8)


def test_topological_entropy_bounded_on_entangled_state():
    """On a non-trivial Gaussian state S^{top} is some real number — just
    check finiteness and that it did not get confused with the existing
    ABDC implementation in gaussian_backend (they use different partitions
    so generally disagree; here we only sanity check finiteness)."""
    L = 8
    model = build_gaussian_chain_model(L=L, w=1.0, alpha=0.5)
    rng = np.random.default_rng(7)
    traj = gaussian_born_rule_trajectory(model, T=3.0, rng=rng)
    gamma = np.asarray(traj.final_covariance)
    S_top = topological_entropy(gamma, L)
    assert np.isfinite(S_top)
    # Bound: |S_top| <= 2 * max half-cut entropy, which is <= L/2 bits.
    assert abs(S_top) <= L / 2 + 1e-6


# --------------------------------------------------------------------------
# dual_topological_entropy
# --------------------------------------------------------------------------

def test_dual_topological_nan_when_Ld_not_divisible_by_4():
    L = 8  # L_d = L-1 = 7 (not divisible by 4)
    model = build_gaussian_chain_model(L=L, w=1.0, alpha=0.5)
    gamma = neel_covariance(L)
    result = dual_topological_entropy(gamma, list(model.jump_pairs))
    assert np.isnan(result)


def test_dual_topological_finite_when_Ld_divisible_by_4():
    L = 5  # L_d = 4 (divisible by 4)
    model = build_gaussian_chain_model(L=L, w=1.0, alpha=0.5)
    gamma = neel_covariance(L)
    result = dual_topological_entropy(gamma, list(model.jump_pairs))
    assert np.isfinite(result)


# --------------------------------------------------------------------------
# compute_all_observables
# --------------------------------------------------------------------------

def test_compute_all_observables_raises_if_L_not_div_4():
    L = 6
    model = build_gaussian_chain_model(L=L, w=1.0, alpha=0.5)
    gamma = neel_covariance(L)
    with pytest.raises(ValueError):
        compute_all_observables(gamma, L, list(model.jump_pairs))


def test_compute_all_observables_keys_and_nan_behaviour():
    L = 8  # L%4==0, but L_d=7 so B_L_prime is NaN
    model = build_gaussian_chain_model(L=L, w=1.0, alpha=0.5)
    gamma = neel_covariance(L)
    obs = compute_all_observables(gamma, L, list(model.jump_pairs))
    assert set(obs.keys()) == {"S_half", "S_top", "S_top_d", "B_L", "B_L_prime"}
    # Neel → everything zero except dual which is NaN (L_d=7).
    assert obs["S_half"] == pytest.approx(0.0, abs=1e-10)
    assert obs["S_top"] == pytest.approx(0.0, abs=1e-10)
    assert obs["B_L"] == pytest.approx(0.0, abs=1e-10)
    assert np.isnan(obs["S_top_d"])
    assert np.isnan(obs["B_L_prime"])


# --------------------------------------------------------------------------
# backward_pass_io
# --------------------------------------------------------------------------

def test_backward_pass_save_load_roundtrip_and_ZT_at_zeta1():
    """At zeta=1 the backward pass is untilted (scalar_rate = 0), so
    Z_T = Tr(G_0 rho_0) = Tr(identity * rho_0) = 1 up to ODE discretisation."""
    L = 4
    T = 0.5
    model = build_gaussian_chain_model(L=L, w=1.0, alpha=0.5)
    bwd = run_gaussian_backward_pass(model, T=T, zeta=1.0, sample_points=33)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bwd.npz"
        save_backward_pass(
            bwd,
            path,
            metadata=dict(L=L, alpha=0.5, w=1.0, zeta=1.0, T=T, lam=0.5),
        )
        loaded = load_backward_pass(path)

    # Z_T at zeta=1 should be 1 (identity overlap with a pure state).
    assert loaded.Z_T == pytest.approx(1.0, abs=1e-4)
    assert loaded.theta_doob == pytest.approx(0.0, abs=1e-4)
    assert loaded.T == pytest.approx(T)
    assert loaded.zeta == pytest.approx(1.0)

    # state_at matches the original at the sample grid points.
    for t_frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = t_frac * T
        C_orig, z_orig = bwd.state_at(t)
        C_loaded, z_loaded = loaded.state_at(t)
        assert C_loaded.shape == C_orig.shape
        # Interpolation introduces some error; tolerances reflect that.
        np.testing.assert_allclose(C_loaded, C_orig, atol=5e-3)
        assert z_loaded == pytest.approx(z_orig, abs=5e-3)


def test_backward_pass_ZT_nontrivial_at_zeta_less_than_1():
    """At zeta < 1 the ODE tilts log z downward, so Z_T < 1. Check that
    Z_T as computed via gaussian_overlap matches an independent evaluation."""
    L = 4
    T = 0.5
    zeta = 0.5
    model = build_gaussian_chain_model(L=L, w=1.0, alpha=0.5)
    bwd = run_gaussian_backward_pass(model, T=T, zeta=zeta, sample_points=33)

    from pps_qj.overlaps import gaussian_overlap
    C0, z0 = bwd.state_at(0.0)
    Z_T_ref = float(gaussian_overlap(C0, model.gamma0, z_scalar=z0))

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bwd.npz"
        save_backward_pass(
            bwd,
            path,
            metadata=dict(L=L, alpha=0.5, w=1.0, zeta=zeta, T=T, lam=0.5),
        )
        loaded = load_backward_pass(path)

    assert loaded.Z_T == pytest.approx(Z_T_ref, rel=1e-6)
    # Untilted Z_T=1 reference; tilted should differ.
    assert loaded.Z_T < 1.0
    assert loaded.theta_doob == pytest.approx(np.log(Z_T_ref) / T, rel=1e-6)
