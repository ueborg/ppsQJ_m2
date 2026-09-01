"""THE production entry point for a QJ-PPS Cut B cloning cell.

Usage
-----
From a config file (the recommended production form)::

    python -m pps_qj.production.run --config configs/production/cell.yaml

Fully from the command line (equivalent, and what the tests use)::

    python -m pps_qj.production.run \
        --L 32 --zeta 0.30 --lam 0.2793 --T 32 --Nc 64 \
        --realizations 2 --seed 12345 --output-dir outputs/production

Any command-line parameter overrides the same key from ``--config``.

What this is
------------
A thin, explicit driver around ``pps_qj.cloning.run_cloning`` — the validated
guided Feynman-Kac cloning sampler — plus the batched CMI/B_L reduction from
``pps_qj.parallel.worker_clone_pps``.  No sampler code is reimplemented here.

What it deliberately does NOT do
--------------------------------
It does not read ``PPS_*`` environment variables.  Configuration comes from the
config file and the command line only.  The recorded production failure mode is
a submit script whose environment silently disagreed with its driver
(TASK-2026-08-11-ALGRD §2), and env-var configuration is what made that
possible.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pps_qj.cloning import CloningCollapse, CloningResult, run_cloning
from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
from pps_qj.production.config import ConfigError, ProductionConfig
from pps_qj.production.provenance import build_provenance


# ---------------------------------------------------------------------------
# One realisation
# ---------------------------------------------------------------------------

def _run_one_realisation(payload: dict) -> dict:
    """Run a single cloning realisation.  Returns a plain picklable dict."""
    cfg: ProductionConfig = ProductionConfig.from_dict(payload["cfg"])
    r: int = payload["r"]
    seed = cfg.realisation_seed(r)

    t0 = time.time()
    c0 = time.process_time()
    try:
        rng = np.random.default_rng(seed)
        model = build_gaussian_chain_model(L=cfg.L, w=cfg.w, alpha=cfg.alpha)

        result: CloningResult = run_cloning(
            model,
            zeta=cfg.zeta,
            T_total=cfg.T,
            N_c=cfg.N_c,
            rng=rng,
            delta_tau=cfg.delta_tau,
            n_burnin_frac=cfg.n_burnin_frac,
            record_entropy=True,
            entropy_stride=cfg.entropy_stride,
            show_progress=False,
            backend="scalar",
            record_renyi=cfg.record_renyi,
            proposal_c=cfg.proposal_c,
            jump_update_method=cfg.jump_update_method,
            refresh_every=cfg.refresh_every,
            solver_method=cfg.solver_method,
            eps_hazard=cfg.eps_hazard,
            record_selection_history=cfg.record_selection_history,
        )

        out: dict[str, Any] = {
            "ok": True,
            "realisation": r,
            "seed": seed,
            "S_mean": float(result.S_mean),
            "S_std": float(result.S_std),
            "S_var": float(result.S_var),
            "theta_hat": float(result.theta_hat),
            "eff_sample_size": float(result.eff_sample_size),
            "n_collapses": int(result.n_collapses),
            "n_T_mean": float(result.n_T_mean),
            "chi_k": float(result.chi_k),
            "covar_Sk": float(result.covar_Sk),
            "delta_tau": float(result.delta_tau),
            "n_burnin_steps": int(result.n_burnin_steps),
            # --- genealogy ---
            "min_ess_frac_postburnin": float(result.min_ess_frac_postburnin),
            "n_distinct_ancestors": int(result.n_distinct_ancestors),
            "n_resampling_events": int(result.n_resampling_events),
            "ess_history": np.asarray(result.ess_history, dtype=np.float64),
            "ancestor_ids_final": np.asarray(
                result.ancestor_ids_final, dtype=np.int64
            ),
            # --- statistical diagnostics (TASK-2026-08-30-SMCSTAT) ---
            "ess_lineage_history": np.asarray(
                result.ess_lineage_history, dtype=np.float64
            ),
            "ess_lineage_final": (
                float(result.ess_lineage_history[-1])
                if np.size(result.ess_lineage_history) else float("nan")
            ),
            "n_solver_fallbacks": int(result.n_solver_fallbacks),
        }
        if cfg.record_selection_history:
            out["selection_history"] = np.asarray(
                result.selection_history, dtype=np.int32
            )
        if cfg.record_renyi:
            out.update(
                S_renyi_2=float(result.S_renyi_2_mean),
                S_renyi_3=float(result.S_renyi_3_mean),
                S_renyi_2_std=float(result.S_renyi_2_std),
                S_renyi_3_std=float(result.S_renyi_3_std),
                corr_decay_r=np.asarray(result.corr_decay_r, dtype=np.float64),
                corr_decay_mean=np.asarray(result.corr_decay_mean, dtype=np.float64),
            )

        # --- t=T locators, per clone then averaged (OBS-BLPROD-001 convention) ---
        if cfg.computes_B_L:
            comps = _batched_compute_B_L(result.final_covs, cfg.L)
            # Store the FOUR subsystem entropies separately, not only the
            # assembled CMI: CMI is a four-term cancellation and storing only
            # the difference makes later drift undiagnosable.
            for name in ("CMI", "B_L", "S_AB", "S_BC", "S_B", "S_ABC"):
                arr = np.asarray(comps[name], dtype=np.float64)
                fin = np.isfinite(arr)
                out[f"{name}_mean"] = float(np.mean(arr[fin])) if fin.any() else float("nan")
                out[f"{name}_std"] = float(np.std(arr[fin])) if fin.any() else float("nan")
                out[f"{name}_n_finite"] = int(fin.sum())
                # Store the PER-CLONE array, not only its mean and spread.
                # Without it no one can recompute a variance decomposition,
                # a genealogical variance estimate, or an N_eff after the fact
                # - which is exactly why the historical corpus cannot be
                # re-diagnosed. 6 * N_c float64 is ~6 kB at N_c = 128.
                out[f"{name}_per_clone"] = arr
        out["wall_time"] = time.time() - t0
        out["cpu_time"] = time.process_time() - c0
        return out

    except CloningCollapse as exc:
        return {
            "ok": False, "realisation": r, "seed": seed,
            "error": f"CloningCollapse: {exc}", "n_collapses": 1,
            "wall_time": time.time() - t0, "cpu_time": time.process_time() - c0,
        }
    except Exception as exc:  # noqa: BLE001 - one bad realisation must not kill the cell
        return {
            "ok": False, "realisation": r, "seed": seed,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), "n_collapses": 0,
            "wall_time": time.time() - t0, "cpu_time": time.process_time() - c0,
        }


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def _nanstat(a: np.ndarray) -> tuple[float, float, float, int]:
    """(mean, std, standard error, n) over the finite entries."""
    a = np.asarray(a, dtype=np.float64)
    fin = np.isfinite(a)
    n = int(fin.sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    vals = a[fin]
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
    se = float(s / np.sqrt(n)) if n > 1 else float("nan")
    return m, s, se, n


def _genealogical_ess(ancestor_ids: np.ndarray, N_c: int) -> dict[str, float]:
    """Genealogical ESS and clone-family structure at t = T.

    ``counts[i]`` is the number of the final N_c slots descending from founder
    ``i``.  GESS = (sum counts)^2 / sum counts^2 = N_c^2 / sum counts^2, which
    equals N_c when every founder has exactly one descendant and 1 when the
    whole population descends from a single founder.

    This is the diagnostic that per-step ESS is blind to: campaigns recorded
    per-step ESS/N_c of 0.98 while the entire final population descended from
    ONE founder (TASK-2026-08-11-ALGRD §7 cross-cutting item 3).
    """
    ids = np.asarray(ancestor_ids, dtype=np.int64)
    if ids.size == 0:
        return {
            "gess": float("nan"), "gess_frac": float("nan"),
            "max_family_size": float("nan"), "n_distinct_ancestors": float("nan"),
        }
    counts = np.bincount(ids, minlength=N_c).astype(np.float64)
    nz = counts[counts > 0]
    ssq = float(np.sum(counts ** 2))
    gess = float(ids.size ** 2 / ssq) if ssq > 0 else float("nan")
    return {
        "gess": gess,
        "gess_frac": float(gess / N_c) if N_c else float("nan"),
        "max_family_size": float(nz.max()) if nz.size else float("nan"),
        "n_distinct_ancestors": float(nz.size),
    }


_SCALAR_FIELDS = (
    "S_mean", "S_std", "S_var", "theta_hat", "eff_sample_size",
    "n_T_mean", "chi_k", "covar_Sk",
    "CMI_mean", "B_L_mean", "S_AB_mean", "S_BC_mean", "S_B_mean", "S_ABC_mean",
    "CMI_std", "B_L_std", "S_AB_std", "S_BC_std", "S_B_std", "S_ABC_std",
    "S_renyi_2", "S_renyi_3",
    "min_ess_frac_postburnin", "n_distinct_ancestors", "n_resampling_events",
    "ess_lineage_final", "n_solver_fallbacks",
    "wall_time", "cpu_time",
)


def _aggregate(cfg: ProductionConfig, results: list[dict]) -> tuple[dict, dict, dict]:
    """Reduce per-realisation dicts into (arrays, summary, genealogy)."""
    R = cfg.realizations
    arrays: dict[str, np.ndarray] = {}
    for name in _SCALAR_FIELDS:
        arrays[name] = np.array(
            [float(res.get(name, np.nan)) if res.get("ok") or name in
             ("wall_time", "cpu_time") else np.nan for res in results],
            dtype=np.float64,
        )

    summary: dict[str, float] = {}
    for name in _SCALAR_FIELDS:
        m, s, se, n = _nanstat(arrays[name])
        summary[name] = m
        summary[f"{name}_std"] = s
        summary[f"{name}_err"] = se
        summary[f"{name}_n_valid"] = float(n)

    # Genealogy, per realisation and aggregated.
    gess_rows = []
    for res in results:
        if res.get("ok") and "ancestor_ids_final" in res:
            gess_rows.append(
                _genealogical_ess(res["ancestor_ids_final"], cfg.N_c)
            )
        else:
            gess_rows.append(
                {"gess": np.nan, "gess_frac": np.nan,
                 "max_family_size": np.nan, "n_distinct_ancestors": np.nan}
            )

    def _col(k: str) -> np.ndarray:
        return np.array([row[k] for row in gess_rows], dtype=np.float64)

    genealogy: dict[str, Any] = {
        "N_c": cfg.N_c,
        "resampling_events_per_realisation": _nanstat(
            arrays["n_resampling_events"])[0],
        "ess_mean": _nanstat(arrays["eff_sample_size"])[0],
        "ess_frac_min_postburnin_mean": _nanstat(
            arrays["min_ess_frac_postburnin"])[0],
        "ess_frac_min_postburnin_worst": (
            float(np.nanmin(arrays["min_ess_frac_postburnin"]))
            if np.isfinite(arrays["min_ess_frac_postburnin"]).any() else float("nan")
        ),
        "gess_mean": _nanstat(_col("gess"))[0],
        "gess_frac_mean": _nanstat(_col("gess_frac"))[0],
        "gess_frac_worst": (
            float(np.nanmin(_col("gess_frac")))
            if np.isfinite(_col("gess_frac")).any() else float("nan")
        ),
        "n_distinct_ancestors_mean": _nanstat(_col("n_distinct_ancestors"))[0],
        "n_distinct_ancestors_worst": (
            float(np.nanmin(_col("n_distinct_ancestors")))
            if np.isfinite(_col("n_distinct_ancestors")).any() else float("nan")
        ),
        "max_family_size_worst": (
            float(np.nanmax(_col("max_family_size")))
            if np.isfinite(_col("max_family_size")).any() else float("nan")
        ),
        "per_realisation": [
            {k: (None if not np.isfinite(v) else float(v)) for k, v in row.items()}
            for row in gess_rows
        ],
        "warning": None,
    }
    # ------------------------------------------------------------------
    # UNAMBIGUOUS UNCERTAINTY FIELDS.
    #
    # The generic loop above produces two similarly named quantities that mean
    # completely different things:
    #     summary["CMI_std"]       = mean over realisations of the WITHIN-
    #                                population across-CLONE spread
    #     summary["CMI_mean_std"]  = spread ACROSS realisations of the
    #                                population mean
    # A reader reaching for "the std of CMI" will very reasonably take the
    # first and divide by sqrt(N_c), which understates the uncertainty by
    # sqrt(VIF) - up to a factor of ten in this project's own corpus. That is
    # the OBS-BL-001 failure mode (one label, two quantities) in a new place,
    # so the correct quantities are also emitted under names that cannot be
    # confused, and the naive one is emitted too, explicitly labelled wrong.
    #
    # The interval is STUDENT-t, not normal-z and not a percentile bootstrap
    # over realisations: measured coverage at nominal 0.95 over 1,696 corpus
    # cells is 0.926-0.940 for t, 0.786-0.912 for z, and 0.716-0.894 for the
    # bootstrap, which is WORSE THAN t at every R <= 10.
    from math import sqrt as _sqrt
    try:
        from scipy.stats import t as _tdist
        _tcrit = lambda n: float(_tdist.ppf(0.975, n - 1)) if n > 1 else float("nan")
    except Exception:                                    # pragma: no cover
        _tcrit = lambda n: float("nan")
    for _obs in ("CMI", "B_L", "S_AB", "S_BC", "S_B", "S_ABC"):
        _key = f"{_obs}_mean"
        if _key not in arrays:
            continue
        _n = int(summary.get(f"{_key}_n_valid", 0))
        _sem = summary.get(f"{_key}_err", float("nan"))
        summary[f"{_obs}_across_population_sem"] = _sem
        summary[f"{_obs}_across_population_std"] = summary.get(f"{_key}_std", float("nan"))
        summary[f"{_obs}_within_population_clone_std"] = summary.get(
            f"{_obs}_std", float("nan"))
        summary[f"{_obs}_t_crit_95"] = _tcrit(_n)
        summary[f"{_obs}_ci95_halfwidth"] = (
            _tcrit(_n) * _sem if np.isfinite(_sem) else float("nan"))
        _naive = summary.get(f"{_obs}_std", float("nan"))
        summary[f"{_obs}_naive_clone_sem_DO_NOT_USE"] = (
            float(_naive / _sqrt(cfg.N_c * max(_n, 1)))
            if np.isfinite(_naive) else float("nan"))
        _corr = summary[f"{_obs}_naive_clone_sem_DO_NOT_USE"]
        summary[f"{_obs}_variance_inflation_factor"] = (
            float((_sem / _corr) ** 2) if (np.isfinite(_sem) and _corr) else float("nan"))

    genealogy["ess_lineage_final_mean"] = _nanstat(arrays["ess_lineage_final"])[0]
    genealogy["ess_lineage_frac_final_mean"] = (
        genealogy["ess_lineage_final_mean"] / cfg.N_c if cfg.N_c else float("nan"))
    _fb = float(np.nansum(arrays["n_solver_fallbacks"]))
    genealogy["n_solver_fallbacks_total"] = _fb
    if _fb > 0:
        genealogy["warning"] = (
            (genealogy.get("warning") or "")
            + f" UNCONTROLLED SOLVER FALLBACK fired {int(_fb)} times: the "
              f"brentq waiting-time solve failed and the state advanced to "
              f"0.5*T_rem while the weight was still credited with the exact "
              f"hazard at a root that was not found. This run's deviation from "
              f"the target measure is UNQUANTIFIED; report it, do not pool it."
        ).strip()

    gfm = genealogy["gess_frac_mean"]
    if np.isfinite(gfm) and gfm < 0.05:
        genealogy["warning"] = (
            f"genealogical ESS is {gfm:.3f} of N_c: the final population is "
            f"dominated by very few founders. Per-step ESS does not see this."
        )

    arrays["gess"] = _col("gess")
    arrays["gess_frac"] = _col("gess_frac")
    arrays["max_family_size"] = _col("max_family_size")
    arrays["seeds"] = np.array(
        [cfg.realisation_seed(r) for r in range(R)], dtype=np.int64
    )
    return arrays, summary, genealogy


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_production_cell(cfg: ProductionConfig) -> dict[str, Any]:
    """Run one production cell and write ``<run_id>.npz`` + ``<run_id>.json``.

    Returns the provenance/summary record that was written to JSON.
    """
    cfg.validate()
    started = time.time()
    c0 = time.process_time()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = cfg.run_id()

    dev = cfg.deviations_from_certified()
    print(f"[production] run_id={run_id}", flush=True)
    print(
        f"[production] L={cfg.L} zeta={cfg.zeta} lambda={cfg.lam} "
        f"alpha={cfg.alpha:.6g} w={cfg.w:.6g} T={cfg.T} N_c={cfg.N_c} "
        f"R={cfg.realizations} seed={cfg.seed}",
        flush=True,
    )
    print(
        f"[production] delta_tau={cfg.delta_tau:.6g} n_steps={cfg.n_steps} "
        f"burn_in_steps={cfg.n_burnin_steps} solver={cfg.solver_method} "
        f"jump={cfg.jump_update_method} entropy_stride={cfg.entropy_stride}",
        flush=True,
    )
    if dev:
        print(
            "[production] WARNING: off-baseline configuration: "
            + "; ".join(dev),
            flush=True,
        )

    payloads = [
        {"cfg": cfg.to_dict(), "r": r} for r in range(cfg.realizations)
    ]
    n_workers = max(1, min(cfg.n_workers, cfg.realizations))
    if n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_run_one_realisation, payloads)
    else:
        results = []
        for payload in payloads:
            res = _run_one_realisation(payload)
            tag = "ok" if res.get("ok") else f"FAILED — {res.get('error')}"
            print(
                f"[production]   realisation {res['realisation']} "
                f"(seed {res['seed']}): {tag}",
                flush=True,
            )
            results.append(res)

    arrays, summary, genealogy = _aggregate(cfg, results)
    wall = time.time() - started
    cpu = time.process_time() - c0
    # Worker CPU time is not visible to the parent under multiprocessing, so
    # sum the per-realisation values instead of under-reporting.
    child_cpu = float(np.nansum(arrays["cpu_time"]))
    cpu_total = cpu + (child_cpu if n_workers > 1 else 0.0)

    n_ok = sum(1 for r in results if r.get("ok"))
    status = (
        "complete" if n_ok == cfg.realizations
        else ("partial" if n_ok else "failed")
    )

    per_real = {
        "n_requested": cfg.realizations,
        "n_ok": n_ok,
        "failures": [
            {"realisation": r["realisation"], "seed": r["seed"],
             "error": r.get("error")}
            for r in results if not r.get("ok")
        ],
    }

    prov = build_provenance(
        cfg,
        started_at=started,
        wall_time=wall,
        cpu_time=cpu_total,
        genealogy=genealogy,
        per_realisation=per_real,
        status=status,
    )
    prov["summary"] = {
        k: (None if isinstance(v, float) and not np.isfinite(v) else v)
        for k, v in summary.items()
    }

    # --- per-clone terminal observables and per-window lineage ESS ---------
    # Stacked (realizations, N_c) and (realizations, n_windows). These are the
    # objects that make a run RE-DIAGNOSABLE later: without the per-clone array
    # nobody can recompute a variance decomposition, an N_eff, or a genealogical
    # variance estimate. Their absence is exactly why the 20,355-run historical
    # corpus cannot answer any such question today.
    perclone: dict[str, np.ndarray] = {}
    for _name in ("CMI", "B_L", "S_AB", "S_BC", "S_B", "S_ABC"):
        _k = f"{_name}_per_clone"
        _rows = [res[_k] for res in results if res.get("ok") and _k in res]
        if _rows and all(len(x) == len(_rows[0]) for x in _rows):
            perclone[_k] = np.asarray(_rows, dtype=np.float64)
    _lin = [res["ess_lineage_history"] for res in results
            if res.get("ok") and "ess_lineage_history" in res]
    if _lin and all(len(x) == len(_lin[0]) for x in _lin):
        perclone["ess_lineage_history"] = np.asarray(_lin, dtype=np.float64)
    if cfg.record_selection_history:
        _sel = [res["selection_history"] for res in results
                if res.get("ok") and "selection_history" in res]
        if _sel and all(x.shape == _sel[0].shape for x in _sel):
            perclone["selection_history"] = np.asarray(_sel, dtype=np.int32)

    npz_path = out_dir / f"{run_id}.npz"
    json_path = out_dir / f"{run_id}.json"

    # The .npz carries the numbers AND a copy of the provenance JSON, so a
    # detached .npz is still self-describing.
    np.savez_compressed(
        npz_path,
        provenance_json=np.array(json.dumps(prov, default=str)),
        **{f"real_{k}": v for k, v in arrays.items()},
        **{f"clone_{k}": v for k, v in perclone.items()},
        **{f"summary_{k}": np.float64(v) for k, v in summary.items()},
    )
    json_path.write_text(json.dumps(prov, indent=2, default=str))

    print(
        f"[production] status={status} "
        f"CMI={summary.get('CMI_mean', float('nan')):.6g}"
        f"±{summary.get('CMI_mean_err', float('nan')):.3g} "
        f"B_L={summary.get('B_L_mean', float('nan')):.6g} "
        f"GESS/N_c={genealogy['gess_frac_mean']:.4g} "
        f"wall={wall:.1f}s",
        flush=True,
    )
    if genealogy["warning"]:
        print(f"[production] WARNING: {genealogy['warning']}", flush=True)
    print(f"[production] wrote {npz_path}", flush=True)
    print(f"[production] wrote {json_path}", flush=True)
    return prov


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pps_qj.production.run",
        description="Production QJ-PPS Cut B cloning cell (guided Feynman-Kac SMC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="YAML or JSON production config file")

    g = p.add_argument_group("physics")
    g.add_argument("--L", type=int, help="chain length (L %% 4 == 0 for CMI/B_L)")
    g.add_argument("--zeta", type=float, help="partial post-selection parameter, (0,1]")
    g.add_argument("--lam", type=float, help="lambda; alpha = lam, w = 1 - lam")
    g.add_argument("--T", type=float, help="total evolution time")
    g.add_argument("--Nc", dest="N_c", type=int, help="clone population size")
    g.add_argument("--realizations", type=int, help="independent realisations")
    g.add_argument("--seed", type=int, help="base seed")
    g.add_argument("--burn-in-frac", dest="n_burnin_frac", type=float,
                   help="burn-in fraction for time-averaged diagnostics")

    s = p.add_argument_group("sampler")
    s.add_argument("--dtau-mult", dest="dtau_mult", type=float,
                   help="delta_tau = dtau_mult / (2*alpha*(L-1))")
    s.add_argument("--proposal-scheme", dest="proposal_scheme", type=str,
                   choices=["guided_reduced_rate", "physical"])
    s.add_argument("--jump-update", dest="jump_update_method", type=str,
                   choices=["lowrank", "eigh"])
    s.add_argument("--refresh-every", dest="refresh_every", type=int,
                   help="full-eigh refresh interval for the low-rank update")
    s.add_argument("--entropy-stride", dest="entropy_stride", type=int,
                   help="record running entropy every N windows")
    s.add_argument("--solver", dest="solver_method", type=str,
                   choices=["brentq", "newton"],
                   help="waiting-time solver ('newton' is UNCERTIFIED)")

    o = p.add_argument_group("output")
    o.add_argument("--observables", type=str, default=None,
                   help="comma-separated: CMI,B_L,entropy,activity,renyi,corr_decay")
    o.add_argument("--output-dir", dest="output_dir", type=str)
    o.add_argument("--run-label", dest="run_label", type=str)
    o.add_argument("--n-workers", dest="n_workers", type=int,
                   help="realisation-level parallelism")
    o.add_argument("--notes", type=str)
    o.add_argument("--print-config", action="store_true",
                   help="resolve and print the config, then exit without running")
    return p


def config_from_args(argv: Optional[list[str]] = None) -> tuple[ProductionConfig, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config:
        base = ProductionConfig.from_file(args.config).to_dict()
    else:
        base = {}

    overrides: dict[str, Any] = {}
    for key in (
        "L", "zeta", "lam", "T", "N_c", "realizations", "seed", "n_burnin_frac",
        "dtau_mult", "proposal_scheme", "jump_update_method", "refresh_every",
        "entropy_stride", "solver_method", "output_dir", "run_label",
        "n_workers", "notes",
    ):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    if args.observables:
        overrides["observables"] = tuple(
            x.strip() for x in args.observables.split(",") if x.strip()
        )

    merged = {**base, **overrides}
    required = ("L", "zeta", "lam", "T", "N_c")
    missing = [k for k in required if k not in merged or merged[k] is None]
    if missing:
        parser.error(
            "missing required parameter(s): "
            + ", ".join(missing)
            + " — supply them via --config or on the command line"
        )
    return ProductionConfig.from_dict(merged), args


def main(argv: Optional[list[str]] = None) -> int:
    try:
        cfg, args = config_from_args(argv)
    except ConfigError as exc:
        print(f"[production] config error: {exc}", file=sys.stderr)
        return 2

    if args.print_config:
        print(json.dumps(cfg.resolved_dict(), indent=2, default=str))
        return 0

    try:
        prov = run_production_cell(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"[production] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0 if prov["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
