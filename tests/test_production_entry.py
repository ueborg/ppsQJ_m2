"""Local correctness certification for the production QJ-PPS entry point.

Gates A-H of TASK-2026-08-20-PRODUCTION-READY §4.  All are small enough to run
on a laptop in a couple of minutes; none is a production-scale validation and
none is claimed to be one.

Statistical gates use a fixed seed set, so they are deterministic reruns rather
than fresh random draws.  Where a gate is statistical its tolerance is stated
in sigma and the measured z is asserted, not a bare tolerance on the mean.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pps_qj.cloning import run_cloning
from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
from pps_qj.production.config import (
    ALGORITHM_VERSION,
    OUTPUT_SCHEMA_VERSION,
    ConfigError,
    ProductionConfig,
)
from pps_qj.production.run import _genealogical_ess, run_production_cell

REPO_ROOT = Path(__file__).resolve().parents[1]

# A tiny but non-trivial cell: L % 4 == 0 so CMI/B_L exist, w != 0 so the
# Kitaev hopping is on, zeta < 1 so resampling is exercised.
TINY = dict(L=8, zeta=0.4, lam=0.3, T=6.0, N_c=64)


def _model(L: int, lam: float):
    return build_gaussian_chain_model(L=L, w=1.0 - lam, alpha=lam)


def _dtau(L: int, lam: float, mult: float = 6.0) -> float:
    return mult / (2.0 * lam * (L - 1))


def _cmi_of(result) -> float:
    return float(np.nanmean(_batched_compute_B_L(result.final_covs, result.L)["CMI"]))


def _cfg(**kw) -> ProductionConfig:
    base = dict(TINY)
    base.update(kw)
    cfg = ProductionConfig.from_dict(base)
    return cfg


# ---------------------------------------------------------------------------
# Gate A — tiny-system exact anchor
# ---------------------------------------------------------------------------

def test_A_born_activity_anchor():
    """At zeta = 1 the sampler must reproduce the exact Born click rate.

    Under Born dynamics (no post-selection, no tilting) the mean activity
    density is exactly

        k_bar = <N_T> / (L*T) = alpha * (L - 1) / L

    because every one of the L-1 bonds carries hazard alpha*(1 - sigma_j) and
    <sigma_j> = 0 under the Born average.  This is an exact identity, not a
    fitted reference, so it is the strongest cheap check available.
    """
    L, lam, T = 8, 0.3, 6.0
    model = _model(L, lam)
    vals = []
    for s in range(6):
        r = run_cloning(
            model, zeta=1.0, T_total=T, N_c=200,
            rng=np.random.default_rng(s), delta_tau=_dtau(L, lam),
            proposal_c=1.0, jump_update_method="lowrank", entropy_stride=4,
        )
        vals.append(r.n_T_mean)
    v = np.asarray(vals)
    predicted = lam * (L - 1) / L
    se = v.std(ddof=1) / np.sqrt(v.size)
    z = (v.mean() - predicted) / se
    assert abs(z) < 3.0, (
        f"Born activity anchor violated: measured {v.mean():.6f} +- {se:.6f} "
        f"vs exact {predicted:.6f} (z = {z:.2f})"
    )


def test_A_zeta_one_has_no_resampling():
    """zeta == 1 is the untilted control: cloning must degenerate to plain
    independent trajectories, i.e. no resampling and no genealogical loss."""
    L, lam = 8, 0.3
    r = run_cloning(
        _model(L, lam), zeta=1.0, T_total=4.0, N_c=32,
        rng=np.random.default_rng(0), delta_tau=_dtau(L, lam),
        proposal_c=1.0, jump_update_method="lowrank", entropy_stride=4,
    )
    assert r.n_resampling_events == 0
    assert r.n_distinct_ancestors == 32


# ---------------------------------------------------------------------------
# Gate B — reference (eigh) vs low-rank jump update
# ---------------------------------------------------------------------------

def test_B_lowrank_matches_eigh():
    """The low-rank projective-jump update is a certified production
    optimisation.  Against the reference eigh path, on the same seed, it must
    agree to numerical noise on every reported quantity."""
    L, lam, zeta = 8, 0.3, 0.4
    model = _model(L, lam)
    kw = dict(
        zeta=zeta, T_total=6.0, N_c=64, delta_tau=_dtau(L, lam),
        proposal_c=zeta, entropy_stride=4,
    )
    ref = run_cloning(model, rng=np.random.default_rng(7),
                      jump_update_method="eigh", **kw)
    low = run_cloning(model, rng=np.random.default_rng(7),
                      jump_update_method="lowrank", refresh_every=100, **kw)

    assert low.n_distinct_ancestors == ref.n_distinct_ancestors
    assert low.n_resampling_events == ref.n_resampling_events
    for name in ("theta_hat", "S_mean", "n_T_mean"):
        a, b = getattr(ref, name), getattr(low, name)
        assert np.isclose(a, b, rtol=1e-9, atol=1e-11), f"{name}: {a} vs {b}"
    assert np.isclose(_cmi_of(ref), _cmi_of(low), rtol=1e-9, atol=1e-11)


# ---------------------------------------------------------------------------
# Gate C — entropy stride contract
# ---------------------------------------------------------------------------

def test_C_entropy_stride_leaves_final_locators_bitwise_identical():
    """entropy_stride is certified under a specific, checkable contract:

    the running-entropy recording consumes NO randomness (cloning.py records it
    from the current state inside an `if` that touches no RNG), so the t=T
    locators CMI and B_L, and theta, must be *bitwise* identical between
    stride 1 and stride 4 on the same seed.  Only the time-averaged
    diagnostics change, and only by sampling fewer windows.
    """
    L, lam, zeta = 8, 0.3, 0.4
    model = _model(L, lam)
    kw = dict(
        zeta=zeta, T_total=6.0, N_c=64, delta_tau=_dtau(L, lam),
        proposal_c=zeta, jump_update_method="lowrank",
    )
    s1 = run_cloning(model, rng=np.random.default_rng(11), entropy_stride=1, **kw)
    s4 = run_cloning(model, rng=np.random.default_rng(11), entropy_stride=4, **kw)

    # Bitwise on the locators.
    assert s1.theta_hat == s4.theta_hat
    assert _cmi_of(s1) == _cmi_of(s4)
    c1 = _batched_compute_B_L(s1.final_covs, L)
    c4 = _batched_compute_B_L(s4.final_covs, L)
    for name in ("CMI", "B_L", "S_AB", "S_BC", "S_B", "S_ABC"):
        np.testing.assert_array_equal(c1[name], c4[name])
    # Genealogy is untouched too.
    assert s1.n_distinct_ancestors == s4.n_distinct_ancestors

    # The documented cost: fewer recorded windows for the time averages.
    assert len(s4.ess_history) == len(s1.ess_history)
    assert np.isfinite(s4.S_mean)


# ---------------------------------------------------------------------------
# Gate D — path measure / compensator regression
# ---------------------------------------------------------------------------

def test_D_guided_compensator_targets_the_same_measure():
    """The guided reduced-rate proposal (intensity c*lambda, c = zeta) plus the
    exact Radon-Nikodym compensator exp[-(1-zeta)*dLambda] must target exactly
    the same tilted measure as the unguided physical proposal with zeta^n
    weights.  A wrong compensator moves the target, so this is the direct
    path-measure regression.

    It also records the variance reduction that motivates the guided path.
    """
    L, lam, zeta = 8, 0.3, 0.4
    model = _model(L, lam)
    seeds = range(12)

    def arm(proposal_c):
        out = []
        for s in seeds:
            r = run_cloning(
                model, zeta=zeta, T_total=6.0, N_c=300,
                rng=np.random.default_rng(s), delta_tau=_dtau(L, lam),
                proposal_c=proposal_c, jump_update_method="lowrank",
                entropy_stride=4,
            )
            out.append(_cmi_of(r))
        return np.asarray(out)

    guided = arm(zeta)
    physical = arm(None)
    diff = guided.mean() - physical.mean()
    se = np.sqrt(guided.var(ddof=1) / guided.size
                 + physical.var(ddof=1) / physical.size)
    z = diff / se
    assert abs(z) < 3.0, (
        f"guided and physical proposals disagree on CMI: "
        f"{guided.mean():.6f} vs {physical.mean():.6f} (z = {z:.2f}) — "
        f"the Radon-Nikodym compensator is suspect"
    )
    # The guided path is meant to be the tighter one; if it ever stops being
    # so at this cell, the reason should be understood before production use.
    assert guided.std(ddof=1) <= physical.std(ddof=1) * 1.5


def test_D_zeta_one_compensator_is_unity():
    """At zeta = 1 the compensator exponent (1 - zeta)*dLambda vanishes and the
    proposal is untilted, so the log-weight accumulator theta must be exactly
    zero — no floating-point drift permitted."""
    L, lam = 8, 0.3
    r = run_cloning(
        _model(L, lam), zeta=1.0, T_total=4.0, N_c=32,
        rng=np.random.default_rng(3), delta_tau=_dtau(L, lam),
        proposal_c=1.0, jump_update_method="lowrank", entropy_stride=4,
    )
    assert abs(r.theta_hat) < 1e-12, f"theta_hat = {r.theta_hat!r} at zeta = 1"


# ---------------------------------------------------------------------------
# Gate E — waiting-time solver
# ---------------------------------------------------------------------------

def test_E_production_solver_is_brentq():
    """The certified production solver is brentq.  newton remains an
    UNCERTIFIED optional candidate and must not be the default."""
    cfg = _cfg()
    assert cfg.solver_method == "brentq"
    assert cfg.deviations_from_certified() == []
    assert "solver_method='newton'" in "".join(
        _cfg(solver_method="newton").deviations_from_certified()
    )


def test_E_newton_candidate_agrees_with_brentq_statistically():
    """Candidate-grade check only.

    newton perturbs accepted waiting times at ~1e-6 and is therefore a
    STATISTICAL change.  This gate records that it is not grossly wrong at a
    tiny cell; it is NOT the production-scale paired-seed validation artifact
    that TASK-2026-08-11-ALGRD found missing, and passing it does not certify
    newton for production.
    """
    L, lam, zeta = 8, 0.3, 0.4
    model = _model(L, lam)
    seeds = range(10)

    def arm(solver):
        return np.asarray([
            _cmi_of(run_cloning(
                model, zeta=zeta, T_total=6.0, N_c=200,
                rng=np.random.default_rng(s), delta_tau=_dtau(L, lam),
                proposal_c=zeta, jump_update_method="lowrank",
                entropy_stride=4, solver_method=solver,
            ))
            for s in seeds
        ])

    b = arm("brentq")
    n = arm("newton")
    diff = b.mean() - n.mean()
    se = np.sqrt(b.var(ddof=1) / b.size + n.var(ddof=1) / n.size)
    assert abs(diff / se) < 3.0, (
        f"newton vs brentq: {b.mean():.6f} vs {n.mean():.6f} "
        f"(z = {diff/se:.2f})"
    )


# ---------------------------------------------------------------------------
# Gate F — genealogy output sanity
# ---------------------------------------------------------------------------

def test_F_genealogical_ess_bounds_and_consistency():
    ids = np.array([0, 0, 0, 0], dtype=np.int64)
    g = _genealogical_ess(ids, 4)
    assert g["gess"] == pytest.approx(1.0)          # one founder -> GESS 1
    assert g["n_distinct_ancestors"] == 1
    assert g["max_family_size"] == 4

    ids = np.array([0, 1, 2, 3], dtype=np.int64)
    g = _genealogical_ess(ids, 4)
    assert g["gess"] == pytest.approx(4.0)          # all distinct -> GESS N_c
    assert g["gess_frac"] == pytest.approx(1.0)
    assert g["max_family_size"] == 1


def test_F_genealogy_recorded_in_a_real_run(tmp_path):
    cfg = _cfg(realizations=2, seed=4242, output_dir=str(tmp_path))
    prov = run_production_cell(cfg)
    gen = prov["genealogy"]

    assert gen["N_c"] == cfg.N_c
    # zeta < 1 resamples every window.
    assert gen["resampling_events_per_realisation"] == pytest.approx(cfg.n_steps)
    assert 1.0 <= gen["gess_mean"] <= cfg.N_c
    assert 0.0 < gen["gess_frac_mean"] <= 1.0
    assert 1 <= gen["n_distinct_ancestors_mean"] <= cfg.N_c
    assert gen["max_family_size_worst"] >= 1
    assert len(gen["per_realisation"]) == 2


# ---------------------------------------------------------------------------
# Gate G — deterministic repeatability
# ---------------------------------------------------------------------------

def test_G_fixed_seed_is_bitwise_repeatable(tmp_path):
    cfg_a = _cfg(realizations=2, seed=31337, output_dir=str(tmp_path / "a"))
    cfg_b = _cfg(realizations=2, seed=31337, output_dir=str(tmp_path / "b"))
    a = run_production_cell(cfg_a)["summary"]
    b = run_production_cell(cfg_b)["summary"]

    for key in ("CMI_mean", "B_L_mean", "S_AB_mean", "theta_hat",
                "n_T_mean", "n_distinct_ancestors"):
        assert a[key] == b[key], f"{key} not repeatable: {a[key]} vs {b[key]}"


def test_G_different_seed_changes_the_answer(tmp_path):
    """Negative control for the repeatability gate: if the seed were ignored,
    test_G above would pass trivially."""
    a = run_production_cell(
        _cfg(realizations=2, seed=1, output_dir=str(tmp_path / "a")))["summary"]
    b = run_production_cell(
        _cfg(realizations=2, seed=2, output_dir=str(tmp_path / "b")))["summary"]
    assert a["CMI_mean"] != b["CMI_mean"]


# ---------------------------------------------------------------------------
# Gate H — output / provenance schema
# ---------------------------------------------------------------------------

REQUIRED_TOP = (
    "provenance_schema_version", "output_schema_version", "algorithm_version",
    "code_version", "entry_point", "status", "timestamp_utc",
    "runtime_seconds", "cpu_time_seconds", "git", "environment", "config",
    "algorithm", "observable_definitions", "genealogy", "per_realisation",
    "summary",
)
REQUIRED_GIT = ("git_commit", "git_dirty", "git_branch")
REQUIRED_ENV = ("hostname", "python_version", "numpy_version",
                "scheduler_job_id", "cpu_count")
REQUIRED_CONFIG = (
    "L", "zeta", "lam", "alpha", "w", "T", "N_c", "realizations", "seed",
    "n_burnin_frac", "n_burnin_steps", "delta_tau", "dtau_mult", "n_steps",
    "entropy_stride", "refresh_every", "realisation_seeds",
)
REQUIRED_ALGO = (
    "family", "target_measure", "proposal_scheme", "proposal_c", "compensator",
    "waiting_time_solver", "jump_update", "low_rank_enabled", "refresh_every",
    "entropy_stride", "resampling", "window", "burn_in",
    "deviations_from_certified_baseline",
)
REQUIRED_GENEALOGY = (
    "ess_mean", "gess_mean", "gess_frac_mean", "n_distinct_ancestors_mean",
    "resampling_events_per_realisation", "max_family_size_worst",
)


def test_H_provenance_schema_complete(tmp_path):
    cfg = _cfg(realizations=1, seed=5150, output_dir=str(tmp_path))
    prov = run_production_cell(cfg)

    for k in REQUIRED_TOP:
        assert k in prov, f"missing provenance key: {k}"
    for k in REQUIRED_GIT:
        assert k in prov["git"], f"missing git key: {k}"
    for k in REQUIRED_ENV:
        assert k in prov["environment"], f"missing environment key: {k}"
    for k in REQUIRED_CONFIG:
        assert k in prov["config"], f"missing config key: {k}"
    for k in REQUIRED_ALGO:
        assert k in prov["algorithm"], f"missing algorithm key: {k}"
    for k in REQUIRED_GENEALOGY:
        assert k in prov["genealogy"], f"missing genealogy key: {k}"

    assert prov["algorithm_version"] == ALGORITHM_VERSION
    assert prov["output_schema_version"] == OUTPUT_SCHEMA_VERSION
    # The four CMI subsystem entropies must be stored separately, not only the
    # assembled four-term difference.
    for k in ("S_AB_mean", "S_BC_mean", "S_B_mean", "S_ABC_mean", "CMI_mean"):
        assert k in prov["summary"]
    # Observable conventions must be pinned by canonical ID.
    assert prov["observable_definitions"]["CMI"]["obs_id"] == "OBS-CMI-001"
    assert prov["observable_definitions"]["B_L"]["obs_id"] == "OBS-BLPROD-001"


def test_H_no_credentials_in_environment_capture(tmp_path):
    """The provenance capture must never dump the general environment."""
    prov = run_production_cell(
        _cfg(realizations=1, seed=1, output_dir=str(tmp_path)))
    blob = json.dumps(prov).lower()
    for needle in ("password", "token", "secret", "api_key", "ssh-rsa",
                   "private_key", "passwd"):
        assert needle not in blob, f"provenance blob contains {needle!r}"


def test_H_outputs_written_and_npz_self_describing(tmp_path):
    cfg = _cfg(realizations=2, seed=77, output_dir=str(tmp_path))
    run_production_cell(cfg)
    run_id = cfg.run_id()

    npz_path = tmp_path / f"{run_id}.npz"
    json_path = tmp_path / f"{run_id}.json"
    assert npz_path.exists() and json_path.exists()

    with np.load(npz_path, allow_pickle=False) as z:
        # A detached .npz still carries its own provenance.
        embedded = json.loads(str(z["provenance_json"]))
        assert embedded["config"]["seed"] == 77
        assert embedded["git"]["git_commit"] is not None
        assert z["real_seeds"].tolist() == [
            cfg.realisation_seed(0), cfg.realisation_seed(1)
        ]
        assert z["real_CMI_mean"].size == 2
        assert "summary_CMI_mean" in z

    on_disk = json.loads(json_path.read_text())
    assert on_disk["config"]["run_id"] == run_id


# ---------------------------------------------------------------------------
# Config-surface guards
# ---------------------------------------------------------------------------

def test_config_rejects_bad_values():
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({**TINY, "zeta": 1.5})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({**TINY, "lam": 0.0})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({**TINY, "L": 6})       # L % 4 != 0 with CMI
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({**TINY, "solver_method": "bisect"})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({**TINY, "nonsense_key": 1})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({**TINY, "compensator": "approximate"})


def test_config_alpha_w_convention():
    cfg = _cfg(lam=0.3)
    assert cfg.alpha == pytest.approx(0.3)
    assert cfg.w == pytest.approx(0.7)
    assert cfg.alpha + cfg.w == pytest.approx(1.0)


def test_entry_point_does_not_read_pps_env_vars(monkeypatch, tmp_path):
    """The production entry point must be configured by file/CLI only.  A stale
    PPS_* variable in the environment must not change a production result."""
    clean = run_production_cell(
        _cfg(realizations=1, seed=606, output_dir=str(tmp_path / "clean")))
    for var, val in (
        ("PPS_SOLVER", "newton"),
        ("PPS_JUMP_METHOD", "eigh"),
        ("PPS_ENTROPY_STRIDE", "1"),
        ("PPS_DTAU_MULT", "48"),
        ("PPS_GUIDED", "0"),
    ):
        monkeypatch.setenv(var, val)
    dirty = run_production_cell(
        _cfg(realizations=1, seed=606, output_dir=str(tmp_path / "dirty")))

    assert clean["summary"]["CMI_mean"] == dirty["summary"]["CMI_mean"]
    assert dirty["algorithm"]["waiting_time_solver"] == "brentq"
    assert dirty["algorithm"]["jump_update"] == "lowrank"
    assert dirty["algorithm"]["entropy_stride"] == 4


def test_cli_print_config_round_trips():
    out = subprocess.run(
        [sys.executable, "-m", "pps_qj.production.run",
         "--L", "8", "--zeta", "0.4", "--lam", "0.3", "--T", "6",
         "--Nc", "64", "--print-config"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), check=True,
    )
    resolved = json.loads(out.stdout)
    assert resolved["alpha"] == pytest.approx(0.3)
    assert resolved["w"] == pytest.approx(0.7)
    assert resolved["algorithm_version"] == ALGORITHM_VERSION
    assert resolved["deviations_from_certified"] == []
