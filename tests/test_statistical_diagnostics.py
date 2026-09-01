"""Gates for the statistical diagnostics added by TASK-2026-08-30-SMCSTAT.

The whole point of these diagnostics is that they are ADDITIVE: they must
record what the sampler does without changing it. Gate A asserts exactly that,
bitwise, and it can fail.

Run:
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
    .venv/bin/python3 -m pytest tests/test_statistical_diagnostics.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from pps_qj.cloning import run_cloning
from pps_qj.gaussian_backend import (
    build_gaussian_chain_model,
    gaussian_born_rule_trajectory,
)
from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L

L, ZETA, LAM, NC, T = 16, 0.35, 0.30, 24, 8.0
DTAU = 6.0 / (2.0 * LAM * (L - 1))


def _run(seed, *, record_selection_history=False, zeta=ZETA):
    model = build_gaussian_chain_model(L=L, w=1.0 - LAM, alpha=LAM)
    return run_cloning(
        model, zeta=zeta, T_total=T, N_c=NC,
        rng=np.random.default_rng(seed), delta_tau=DTAU,
        record_entropy=False, backend="scalar",
        proposal_c=(zeta if zeta != 1.0 else 1.0),
        jump_update_method="lowrank", refresh_every=100,
        solver_method="brentq", eps_hazard=1e-9,
        record_selection_history=record_selection_history,
    )


# --- Gate A: the diagnostics are additive -----------------------------------

def test_recording_selection_history_changes_nothing_bitwise():
    """Recording the genealogy must consume no randomness and change no result.

    If this ever fails, the diagnostic has become part of the sampler and every
    run taken with it is off the certified baseline.
    """
    a = _run(20260830, record_selection_history=False)
    b = _run(20260830, record_selection_history=True)
    oa = _batched_compute_B_L(a.final_covs, L)
    ob = _batched_compute_B_L(b.final_covs, L)
    for k in ("CMI", "B_L", "S_AB", "S_BC", "S_B", "S_ABC"):
        assert np.array_equal(np.asarray(oa[k]), np.asarray(ob[k]), equal_nan=True), k
    assert np.array_equal(a.ancestor_ids_final, b.ancestor_ids_final)
    assert np.array_equal(np.asarray(a.ess_history), np.asarray(b.ess_history))
    assert a.theta_hat == b.theta_hat


def test_a_different_seed_actually_differs():
    """Negative control for the gate above: it must be capable of failing."""
    a = _run(20260830)
    b = _run(20260831)
    assert not np.array_equal(a.ancestor_ids_final, b.ancestor_ids_final)


# --- Gate B: the lineage ESS is a real, non-saturated diagnostic -------------

def test_lineage_ess_recorded_every_window_and_bounded():
    r = _run(11)
    e = np.asarray(r.ess_lineage_history, dtype=float)
    assert e.size == len(r.ess_history)
    assert np.all(e >= 0.0) and np.all(e <= NC + 1e-9)
    assert np.all(np.isfinite(e))


def test_lineage_ess_sees_what_instantaneous_ess_cannot():
    """The instantaneous ESS is near-saturated by construction under the guided
    proposal (log w = -(1-zeta)*dLambda is narrow). The lineage-accumulated ESS
    is not, and that difference is the whole reason for adding it."""
    r = _run(12)
    inst = float(np.mean(r.ess_history)) / NC
    lin = float(r.ess_lineage_history[-1]) / NC
    assert inst > 0.8, f"instantaneous ESS unexpectedly low ({inst:.3f})"
    assert lin < inst, (
        f"lineage ESS {lin:.3f} did not fall below instantaneous {inst:.3f}; "
        f"the new diagnostic is carrying no extra information"
    )


def test_zeta_one_does_no_selection_and_keeps_full_lineage_ess():
    r = _run(13, zeta=1.0)
    assert r.n_resampling_events == 0
    assert r.n_distinct_ancestors == NC
    assert np.allclose(np.asarray(r.ess_lineage_history), float(NC))


# --- Gate C: the selection history is sufficient to rebuild the genealogy ----

def test_selection_history_shape_and_reconstructs_the_founders():
    r = _run(14, record_selection_history=True)
    sel = np.asarray(r.selection_history)
    assert sel.shape == (r.n_resampling_events, NC)
    assert sel.dtype == np.int32
    assert sel.min() >= 0 and sel.max() < NC
    # Replay the maps forward: this must reproduce ancestor_ids_final exactly.
    anc = np.arange(NC, dtype=np.intp)
    for row in sel:
        anc = anc[row]
    assert np.array_equal(anc, np.asarray(r.ancestor_ids_final))


def test_selection_history_yields_pairwise_mrca():
    """The property nothing else in the output gives: exact pairwise MRCA depth."""
    r = _run(15, record_selection_history=True)
    sel = np.asarray(r.selection_history)
    lab = np.arange(NC, dtype=np.intp)
    M = np.full((NC, NC), -1, dtype=np.int64)
    depth = 0
    for row in sel[::-1]:
        depth += 1
        lab = row[lab]
        same = lab[:, None] == lab[None, :]
        M[same & (M < 0)] = depth
    M[M < 0] = depth + 1
    np.fill_diagonal(M, 0)
    assert (M == M.T).all()
    assert M.max() <= depth + 1
    assert (M[np.triu_indices(NC, 1)] > 0).all()


def test_selection_history_is_off_by_default():
    """It costs ~1.7 MB per realisation at production scale; it must be opt-in."""
    assert np.asarray(_run(16).selection_history).size == 0


# --- Gate D: the uncontrolled solver fallback is counted --------------------

def test_solver_fallback_counter_exists_and_is_sane():
    model = build_gaussian_chain_model(L=L, w=1.0 - LAM, alpha=LAM)
    t = gaussian_born_rule_trajectory(
        model, T=2.0, rng=np.random.default_rng(3), proposal_c=ZETA,
        jump_update_method="lowrank", refresh_every=100,
        solver_method="brentq", eps_hazard=1e-9,
    )
    assert isinstance(t.n_solver_fallbacks, int)
    assert t.n_solver_fallbacks >= 0
    r = _run(17)
    assert isinstance(r.n_solver_fallbacks, int)
    assert r.n_solver_fallbacks >= 0


# --- Gate E: the production output cannot be misread ------------------------

@pytest.mark.parametrize("obs", ["CMI", "B_L"])
def test_production_emits_unambiguous_uncertainty_fields(tmp_path, obs):
    """summary['CMI_std'] and summary['CMI_mean_std'] are DIFFERENT quantities
    whose names differ only by suffix placement. A reader taking the first and
    dividing by sqrt(N_c) understates the uncertainty by sqrt(VIF). The
    unambiguous names must be present, and so must the deliberately ugly
    'DO_NOT_USE' one, so the mistake is visible rather than available."""
    from pps_qj.production.config import ProductionConfig
    from pps_qj.production.run import run_production_cell

    cfg = ProductionConfig(L=16, zeta=0.35, lam=0.30, T=8.0, N_c=24,
                           realizations=6, seed=99, output_dir=str(tmp_path))
    rec = run_production_cell(cfg)
    s = rec["summary"]
    for suffix in ("across_population_sem", "across_population_std",
                   "within_population_clone_std", "t_crit_95",
                   "ci95_halfwidth", "naive_clone_sem_DO_NOT_USE",
                   "variance_inflation_factor"):
        assert f"{obs}_{suffix}" in s, f"missing {obs}_{suffix}"
    assert np.isfinite(s[f"{obs}_across_population_sem"])
    # The interval is Student-t, not normal-z: t_{5} = 2.571 > 1.96.
    assert s[f"{obs}_t_crit_95"] > 1.96
    assert np.isclose(s[f"{obs}_ci95_halfwidth"],
                      s[f"{obs}_t_crit_95"] * s[f"{obs}_across_population_sem"])


def test_production_emits_per_clone_arrays_and_lineage_ess(tmp_path):
    from pps_qj.production.config import ProductionConfig
    from pps_qj.production.run import run_production_cell

    cfg = ProductionConfig(L=16, zeta=0.35, lam=0.30, T=8.0, N_c=24,
                           realizations=4, seed=101, output_dir=str(tmp_path))
    run_production_cell(cfg)
    npz = sorted(tmp_path.glob("*.npz"))[-1]
    z = np.load(npz, allow_pickle=True)
    for k in ("clone_CMI_per_clone", "clone_B_L_per_clone",
              "clone_ess_lineage_history"):
        assert k in z.files, f"{k} missing from the .npz"
    assert z["clone_CMI_per_clone"].shape == (4, 24)
    # The per-clone means must reproduce the stored per-realisation means.
    a = z["clone_CMI_per_clone"]
    assert np.allclose(np.nanmean(a, axis=1), z["real_CMI_mean"], equal_nan=True)


# --- Gate F: the lineage ESS must work in EVERY zeta sector ------------------
# TASK-2026-08-31-SMCCERT. Gate B above tests zeta = 1 and the production
# intermediate zeta, and both passed while the diagnostic was silently broken at
# zeta = 0.0 - the fully post-selected no-click sector, where `log_w` is never
# bound in run_cloning and the original `'log_w' in dir()` guard fell through to
# zeros. The lineage ESS then read exactly N_c at every window on runs that had
# genuinely collapsed. These are the tests that would have caught it.

def _run_zeta(zeta, proposal_c, seed=11, N_c=16):
    model = build_gaussian_chain_model(L=8, w=0.70, alpha=0.30)
    return run_cloning(
        model, zeta=zeta, T_total=2.0, N_c=N_c,
        rng=np.random.default_rng(seed), delta_tau=0.5,
        record_entropy=False, backend="scalar", proposal_c=proposal_c,
        jump_update_method="lowrank", refresh_every=100,
        solver_method="brentq", eps_hazard=1e-9,
    )


def test_lineage_ess_is_not_saturated_at_zeta_zero():
    """At zeta = 0 the weights are (n_jumps == 0) indicators, which is the most
    degenerate regime the sampler supports. A lineage ESS pinned at N_c there is
    the diagnostic failing, not the population being healthy."""
    r = _run_zeta(0.0, 1.0)
    ess = np.asarray(r.ess_lineage_history, dtype=float)
    assert ess.size > 0
    assert np.all(ess >= 0.0) and np.all(ess <= 16.0 + 1e-9)
    # the run really did collapse
    assert r.n_distinct_ancestors < 16
    # so the lineage ESS must not report a full, undegraded population
    assert ess.min() < 16.0 - 1e-9, (
        f"lineage ESS never dropped below N_c at zeta=0 (min {ess.min()}) while "
        f"only {r.n_distinct_ancestors}/16 founders survived - the diagnostic is "
        f"reporting an all-clear on a collapsed population")


def test_lineage_ess_tracks_selection_pressure_across_zeta():
    """Ordering sanity: zeta = 1 does no selection at all and must stay pinned at
    N_c; a sector that actually resamples must not."""
    assert np.asarray(_run_zeta(1.0, 1.0).ess_lineage_history, float).min() == 16.0
    for zeta, pc in ((0.0, 1.0), (0.30, 0.30)):
        ess = np.asarray(_run_zeta(zeta, pc).ess_lineage_history, float)
        assert ess.min() < 16.0 - 1e-9, f"zeta={zeta} lineage ESS pinned at N_c"


def test_lineage_ess_finite_and_defined_for_every_window():
    for zeta, pc in ((0.0, 1.0), (0.30, 0.30), (1.0, 1.0)):
        r = _run_zeta(zeta, pc)
        ess = np.asarray(r.ess_lineage_history, dtype=float)
        assert ess.size == len(r.ess_history), (
            f"zeta={zeta}: {ess.size} lineage entries for {len(r.ess_history)} windows")
        assert np.all(np.isfinite(ess)), f"zeta={zeta}: non-finite lineage ESS"
