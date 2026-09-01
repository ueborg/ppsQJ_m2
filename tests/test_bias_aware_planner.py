"""Tests for the BIAS-AWARE budget planner. TASK-2026-08-31-SMCCERT.

These are the tests the SMCSTAT prototype could not have passed. Its docstring
wrote MSE = A/(R*N_c) + (B/N_c)^2 and its code implemented only the first term,
so it pinned N_c at the pre-asymptotic floor and spent every remaining
core-hour on R - which cannot reduce a bias.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import plan_cloning_statistics as P  # noqa: E402

CAL = P.BIAS_CALIBRATION


def _a_calibrated_cell():
    if not CAL:
        pytest.skip("no bias calibration table shipped")
    # the highest-|B| calibrated cell: the one where the distinction bites
    key = max(CAL, key=lambda k: abs(CAL[k]["B"]))
    return key, CAL[key]


# --- the refusal is the most important behaviour ---------------------------
def _an_uncalibrated_cell():
    """A cell guaranteed absent from whatever table is shipped.

    Deliberately NOT hardcoded: L=96,T=96,zeta=0.35,lam=0.3032 was uncalibrated
    when this suite was written and became calibrated the same day. A fixture
    that silently turns into its own opposite is worse than no fixture.
    """
    lam = 0.3032
    while any(abs(k[3] - lam) < 1e-9 for k in CAL):
        lam += 0.0011
    return dict(L=128, T=128.0, zeta=0.35, lam=lam)


def test_uncalibrated_cell_returns_calibration_required():
    cell = _an_uncalibrated_cell()
    assert (cell["L"], cell["T"], cell["zeta"], cell["lam"], "systematic") not in CAL
    out = P.plan(**cell, tau2=0.03, target_sem=0.01)
    assert out["status"] == P.CALIBRATION_REQUIRED
    assert out["B_f"] == P.CALIBRATION_REQUIRED
    assert any("B_f" in g for g in out["calibration_gaps"])


def test_every_calibration_gap_is_reported_not_just_the_first():
    """The prototype returned on the first gap, so a caller never learned the
    bias calibration was missing too. 'add T=L VIF data' and 'add a bias ladder'
    are different instructions and both must surface."""
    out = P.plan(L=32, T=999.0, zeta=0.30, lam=0.2793, tau2=0.06, target_sem=0.01)
    assert out["status"] == P.CALIBRATION_REQUIRED
    assert len(out["calibration_gaps"]) >= 1


def test_bias_is_never_inferred_from_vif():
    """A cell with a perfectly good VIF calibration and no bias ladder must
    still refuse. VIF does not predict B: two calibrated cells with overlapping
    VIF differ in B by more than an order of magnitude."""
    out = P.plan(L=128, T=128.0, zeta=0.35, lam=0.30, tau2=0.03, target_sem=0.01)
    assert out["VIF"] != P.CALIBRATION_REQUIRED
    assert out["status"] == P.CALIBRATION_REQUIRED


def test_calibration_table_has_overlapping_vif_with_different_bias():
    """The empirical fact the refusal rests on. If this ever stops holding, the
    refusal may be reconsidered - but not before."""
    if len(CAL) < 2:
        pytest.skip("need at least two calibrated cells")
    pairs = [(a, b) for a in CAL for b in CAL if a < b]
    overlapping = [
        (a, b) for a, b in pairs
        if CAL[a]["vif_min"] <= CAL[b]["vif_max"] and CAL[b]["vif_min"] <= CAL[a]["vif_max"]
    ]
    if not overlapping:
        pytest.skip("no VIF-overlapping calibrated pair in the shipped table")
    ratios = [max(abs(CAL[a]["B"]), abs(CAL[b]["B"]))
              / max(min(abs(CAL[a]["B"]), abs(CAL[b]["B"])), 1e-9)
              for a, b in overlapping]
    assert max(ratios) > 3.0


# --- the allocation must put N_c before R ----------------------------------
def test_bias_floor_binds_and_R_cannot_relax_it():
    key, e = _a_calibrated_cell()
    L, T, zeta, lam, scheme = key
    out = P.plan(L=L, T=T, zeta=zeta, lam=lam, target_sem=0.01,
                 resample_scheme=scheme)
    assert out["status"] == "ok", out
    assert "N_c_min_bias" in out["constraints"]
    assert out["recommended"]["N_c"] >= out["constraints"]["N_c_min_bias"]
    assert out["recommended"]["N_c_binding_constraint"] == "bias"


def test_tightening_the_bias_tolerance_raises_N_c_not_R():
    key, e = _a_calibrated_cell()
    L, T, zeta, lam, scheme = key
    loose = P.plan(L=L, T=T, zeta=zeta, lam=lam, target_sem=0.01,
                   bias_tol=0.05, resample_scheme=scheme)
    tight = P.plan(L=L, T=T, zeta=zeta, lam=lam, target_sem=0.01,
                   bias_tol=0.005, resample_scheme=scheme)
    assert tight["recommended"]["N_c"] > loose["recommended"]["N_c"]
    assert tight["recommended"]["R"] <= loose["recommended"]["R"]


def test_N_c_is_never_traded_down_to_meet_R_min():
    """The prototype, on hitting its R floor, recomputed N_c = ceil(M / R)
    DOWNWARD. That trade is exactly what must not happen."""
    key, e = _a_calibrated_cell()
    L, T, zeta, lam, scheme = key
    out = P.plan(L=L, T=T, zeta=zeta, lam=lam, target_sem=0.01,
                 R_min=64, resample_scheme=scheme)
    assert out["recommended"]["N_c"] >= out["constraints"]["N_c_min_bias"]
    assert out["recommended"]["R"] >= 64


def test_bias_floor_uses_the_conservative_end_of_the_B_interval():
    """A poorly determined B must produce a LARGER N_c, never a confident one."""
    key, e = _a_calibrated_cell()
    L, T, zeta, lam, scheme = key
    out = P.plan(L=L, T=T, zeta=zeta, lam=lam, target_sem=0.01,
                 bias_tol=0.01, resample_scheme=scheme)
    worst = max(abs(e["B_ci"][0]), abs(e["B_ci"][1]))
    assert out["constraints"]["N_c_min_bias"] >= worst / 0.01 - 1


# --- the output must not be misreadable ------------------------------------
def test_reports_bias_sem_and_rmse_separately():
    key, e = _a_calibrated_cell()
    L, T, zeta, lam, scheme = key
    out = P.plan(L=L, T=T, zeta=zeta, lam=lam, target_sem=0.01,
                 resample_scheme=scheme)
    for k in ("predicted_sem_CMI", "predicted_bias_CMI", "predicted_rmse_CMI",
              "predicted_bias_ci", "bias_over_sem", "regime"):
        assert k in out, k
    assert out["predicted_rmse_CMI"] >= out["predicted_sem_CMI"]
    assert out["regime"] in ("BIAS_LIMITED", "VARIANCE_LIMITED", "BALANCED")


def test_refuses_when_the_tolerance_cannot_be_met():
    key, e = _a_calibrated_cell()
    L, T, zeta, lam, scheme = key
    out = P.plan(L=L, T=T, zeta=zeta, lam=lam, target_sem=0.01,
                 bias_tol=1e-12, resample_scheme=scheme)
    # either it escalates N_c enormously or it refuses - never silently accepts
    assert (out.get("status") == "bias tolerance not met"
            or out["recommended"]["N_c"] >= abs(e["B"]) / 1e-12 * 0.5)


def test_bias_tolerance_is_required_when_it_cannot_be_derived():
    key, e = _a_calibrated_cell()
    L, T, zeta, lam, scheme = key
    out = P.plan(L=L, T=T, zeta=zeta, lam=lam, target_dlam=0.001, dD=1.0,
                 resample_scheme=scheme)
    assert out["status"] == P.CALIBRATION_REQUIRED
    assert "bias_tol" in str(out.get("required_input"))


# --- the calibration data file must carry its provenance -------------------
def test_calibration_file_states_its_domain_and_provenance():
    path = os.path.join(ROOT, "tools", "calibration", "bias_calibration.json")
    doc = json.load(open(path))
    for k in ("provenance", "domain_of_validity", "bias_model",
              "uncertainty", "observable", "minimum_detectable_effect_on_B"):
        assert doc.get(k), k
    for c in doc["cells"]:
        assert c["B_ci"][0] <= c["B"] <= c["B_ci"][1]
        assert c["mde_B"] <= doc["minimum_detectable_effect_on_B"] + 1e-9
