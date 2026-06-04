"""Cloning worker for a single (L, lambda, zeta) grid point.

Entry point::

    python -m pps_qj.parallel.worker_clone_pps <task_id> <output_dir>

Runs N_REAL=5 independent realisations per task.  When PPS_N_WORKERS > 1
(set by the SLURM script via the environment variable), realisations are
dispatched in parallel using a multiprocessing.Pool created once per task.

Checkpoint / resume
-------------------
For serial runs (n_workers=1), a partial progress file
``clone_{task_id:05d}_partial.pkl`` is written after each realisations
completes.  If the task is interrupted and restarted, the completed
realisations are loaded from this file and skipped, so only the remaining
realisations are re-run.  The partial file is deleted on successful
completion.  For parallel runs (n_workers>1), pool.map returns all results
simultaneously so no meaningful partial state exists; the existing task-level
idempotency guard (checking for the final .npz) applies as usual.

New observables (v2, cloning.py)
---------------------------------
Each realization now returns:
  n_T_mean  — mean activity density k̄ = <N_T>/(L·T) under tilted measure
  chi_k     — activity susceptibility χ_k = Var(N_T^window)/(L·δτ)
  S_var     — variance Var_ζ(S_L/2) under tilted measure
  covar_Sk  — activity-entropy covariance C_{S,k}/(L·δτ)
These are aggregated across realisations and saved in the .npz output.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from pps_qj.cloning import CloningCollapse, CloningResult, run_cloning
from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.parallel.grid_pps import (
    task_params_clone,
    task_params_clone_slope,
)

N_REAL = 5


def _n_workers_from_env() -> int:
    """Read PPS_N_WORKERS or SLURM_CPUS_PER_TASK from the environment."""
    for var in ("PPS_N_WORKERS", "SLURM_CPUS_PER_TASK"):
        v = os.environ.get(var)
        if v is not None and v.strip().isdigit() and int(v) >= 1:
            return int(v)
    return 1


# ---------------------------------------------------------------------------
# Realisation-level parallelism
# ---------------------------------------------------------------------------

def _run_one_realisation(args: dict) -> dict:
    """Run one full cloning realisation and return a plain result dict."""
    L       = int(args["L"])
    w       = float(args["w"])
    alpha   = float(args["alpha"])
    zeta    = float(args["zeta"])
    T       = float(args["T"])
    N_c     = int(args["N_c"])
    seed    = int(args["seed"])

    rng = np.random.default_rng(seed)
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)

    # Optional dτ multiplier from env-var (default 1.0 = unchanged behaviour).
    # Set PPS_DTAU_MULT=2.0 to halve n_steps; validated safe at ζ ≥ 0.3
    # (mult=2.0) and ζ ≥ 0.5 (mult=3.0) at L=32, with statistical agreement
    # of ⟨S⟩ vs baseline within 3σ across all tested configurations.
    dtau_mult = float(os.environ.get("PPS_DTAU_MULT", "1.0"))
    if dtau_mult != 1.0:
        dtau_default = 1.0 / max(2.0 * alpha * (L - 1), 1e-6)
        delta_tau    = dtau_mult * dtau_default
        cloning_kw   = {"delta_tau": delta_tau}
    else:
        cloning_kw   = {}

    # Optional Renyi + correlation-function extraction from env var.
    # Set PPS_RECORD_RENYI=1 to enable. Backward-compatible: default off.
    record_renyi = os.environ.get("PPS_RECORD_RENYI", "0") not in ("0", "", "false", "False")

    # Backend selector.  PPS_BACKEND=scalar forces the original per-clone
    # loop (bit-exact w.r.t. historical data).  PPS_BACKEND=batched uses
    # the vectorised path (1.1–2× faster, statistically equivalent).
    # Default is 'batched' for new runs; set PPS_BACKEND=scalar to reproduce
    # old results exactly or to A/B against new data.
    # NOTE: the batched backend is deliberately *not* used on existing rescue
    # or dense runs -- those were submitted before this env var existed and
    # their already-running processes will not pick up this change.
    backend = os.environ.get("PPS_BACKEND", "batched")

    # Per-step entropy stride.  Setting PPS_ENTROPY_STRIDE=4 computes the
    # running entropy every 4 steps instead of every step, saving ~6–8% of
    # wall time.  The final B_L / CMI are computed from final_covs at t=T
    # and are unaffected.  S_mean / S_std have slightly lower sample counts
    # but remain unbiased estimates.  Default 1 = every step (old behaviour).
    entropy_stride = max(1, int(os.environ.get("PPS_ENTROPY_STRIDE", "1")))

    try:
        result: CloningResult = run_cloning(
            model, zeta=zeta, T_total=T, N_c=N_c, rng=rng,
            show_progress=False, record_renyi=record_renyi,
            backend=backend, entropy_stride=entropy_stride,
            **cloning_kw,
        )
        return {
            "ok":             True,
            "S_mean":         float(result.S_mean),
            "S_std":          float(result.S_std),
            "theta_hat":      float(result.theta_hat),
            "eff_sample_size": float(result.eff_sample_size),
            "n_collapses":    int(result.n_collapses),
            "n_js_fallbacks": int(result.n_js_fallbacks),
            # New observables
            "n_T_mean":       float(result.n_T_mean),
            "chi_k":          float(result.chi_k),
            "S_var":          float(result.S_var),
            "covar_Sk":       float(result.covar_Sk),
            # Renyi entropies + correlation function (NaN/empty if PPS_RECORD_RENYI=0)
            "S_renyi_2":      float(result.S_renyi_2_mean),
            "S_renyi_3":      float(result.S_renyi_3_mean),
            "S_renyi_2_std":  float(result.S_renyi_2_std),
            "S_renyi_3_std":  float(result.S_renyi_3_std),
            "corr_decay_r":   np.asarray(result.corr_decay_r, dtype=np.float64),
            "corr_decay_mean":np.asarray(result.corr_decay_mean, dtype=np.float64),
            # Covariance matrices for B_L
            "final_covs": [np.asarray(c, dtype=np.float64) for c in result.final_covs],
        }
    except CloningCollapse as exc:
        return {"ok": False, "error": str(exc), "n_collapses": 1}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "n_collapses": 0}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_partial(partial_file: Path) -> dict[int, dict]:
    """Load completed-realisations dict from partial file. Returns {} on failure."""
    try:
        with open(partial_file, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    try:
        partial_file.unlink(missing_ok=True)
    except Exception:
        pass
    return {}


def _save_partial(partial_file: Path, completed: dict[int, dict]) -> None:
    """Atomically write partial file."""
    tmp = partial_file.with_suffix(".pkl.tmp")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(completed, f, protocol=4)
        tmp.replace(partial_file)
    except Exception:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Batched B_L
# ---------------------------------------------------------------------------

def _batched_compute_B_L(covs_list: list[np.ndarray], L: int) -> dict:
    """Batched CMI components and B_L across the final clone population.

    Tripartition on Majorana indices [0, 3L/2):
        A = [0,   L/2)
        B = [L/2, L)
        C = [L,   3L/2)

    Returns
    -------
    dict with per-clone arrays of shape (N_c,):
        S_AB, S_BC, S_B, S_ABC : subsystem entropies for the tripartition
        CMI  = S_AB + S_BC - S_B - S_ABC        (CMI of A:C | B)
        B_L  = CMI * S_AB                       (original Binder-like proxy)

    Returns dict of NaN-filled arrays on L not divisible by 4 or on
    LinAlgError (with B_L recoverable from the single-cov fallback path).
    """
    N_c = len(covs_list)
    _keys = ("S_AB", "S_BC", "S_B", "S_ABC", "CMI", "B_L")

    def _nan_out() -> dict:
        return {k: np.full(N_c, np.nan, dtype=np.float64) for k in _keys}

    if L % 4 != 0:
        return _nan_out()

    gamma_batch = np.stack(
        [np.asarray(c, dtype=np.float64) for c in covs_list], axis=0
    )
    hL = L // 2
    tL = 3 * L // 2

    sub_AB  = gamma_batch[:, :L,       :L      ]   # A∪B : modes [0,    L)
    sub_BC  = gamma_batch[:, hL:hL+L,  hL:hL+L ]   # B∪C : modes [L/2,  3L/2)
    sub_B   = gamma_batch[:, hL:L,     hL:L    ]   # B   : modes [L/2,  L)
    sub_ABC = gamma_batch[:, :tL,      :tL     ]   # A∪B∪C: modes [0,   3L/2)

    def _batch_entropy(sub: np.ndarray) -> np.ndarray:
        ell = sub.shape[-1] // 2
        eigs = np.linalg.eigvalsh((1j * sub).astype(np.complex128))
        nus  = np.clip(np.abs(eigs[:, ell:]), 0.0, 1.0)
        p_p  = np.clip(0.5 * (1.0 + nus), 1e-15, 1.0 - 1e-15)
        p_m  = np.clip(0.5 * (1.0 - nus), 1e-15, 1.0 - 1e-15)
        return -np.sum(p_p * np.log2(p_p) + p_m * np.log2(p_m), axis=-1)

    try:
        S_AB  = _batch_entropy(sub_AB).astype(np.float64)
        S_BC  = _batch_entropy(sub_BC).astype(np.float64)
        S_B   = _batch_entropy(sub_B).astype(np.float64)
        S_ABC = _batch_entropy(sub_ABC).astype(np.float64)
        CMI   = (S_AB + S_BC - S_B - S_ABC).astype(np.float64)
        B_L   = (CMI * S_AB).astype(np.float64)
        return {
            "S_AB": S_AB, "S_BC": S_BC, "S_B": S_B, "S_ABC": S_ABC,
            "CMI":  CMI,  "B_L":  B_L,
        }
    except np.linalg.LinAlgError:
        # Fallback: per-clone via single-cov observables. We only know B_L
        # from this path; mark component entropies as NaN so downstream
        # analysis won't silently mix the two paths.
        from pps_qj.observables import compute_all_observables
        out = _nan_out()
        for i, cov in enumerate(covs_list):
            try:
                out["B_L"][i] = float(compute_all_observables(
                    np.asarray(cov, dtype=np.float64), L, []
                )["B_L"])
            except Exception:
                out["B_L"][i] = np.nan
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_summary_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    tmp.replace(path)


def _nanstat(a: np.ndarray):
    a = a[~np.isnan(a)]
    n = int(a.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.mean(a)), float(np.std(a)), float(np.std(a) / np.sqrt(n)), n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 2:
        raise SystemExit(
            "usage: python -m pps_qj.parallel.worker_clone_pps <task_id> <output_dir> [--grid slope]"
        )
    task_id    = int(argv[0])
    output_dir = Path(argv[1])

    # Optional grid selector: --grid slope dispatches to the slope-test grid.
    # Default (no flag) uses the original v1 clone grid for backward compat.
    grid_name = "v1"
    remaining = argv[2:]
    for i, arg in enumerate(remaining):
        if arg == "--grid" and i + 1 < len(remaining):
            grid_name = remaining[i + 1]
            break

    _GRID_DISPATCH = {
        "v1":    task_params_clone,
        "slope": task_params_clone_slope,
    }
    if grid_name not in _GRID_DISPATCH:
        raise SystemExit(f"Unknown --grid value '{grid_name}'. Choices: {list(_GRID_DISPATCH)}")
    _task_params = _GRID_DISPATCH[grid_name]

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file  = output_dir / f"clone_{task_id:05d}.npz"
    summary_file = output_dir / f"summary_clone_{task_id:05d}.json"
    partial_file = output_dir / f"clone_{task_id:05d}_partial.pkl"

    force_rerun = os.environ.get("PPS_FORCE_RERUN", "0") not in ("0", "", "false", "False")
    if output_file.exists() and not force_rerun:
        print(f"clone task {task_id}: already done, skipping", flush=True)
        return 0
    if output_file.exists() and force_rerun:
        print(f"clone task {task_id}: PPS_FORCE_RERUN=1 — overwriting existing output", flush=True)

    t_start = time.time()
    task   = _task_params(task_id)
    L      = int(task["L"])
    lam    = float(task["lam"])
    alpha  = float(task["alpha"])
    w      = float(task["w"])
    zeta   = float(task["zeta"])
    T      = float(task["T"])
    N_c    = int(task["N_c"])
    seed   = int(task["seed"])
    n_workers = _n_workers_from_env()

    print(
        f"\n=== clone task {task_id}: L={L}, λ={lam:.3f} (α={alpha:.3f}), "
        f"ζ={zeta:.2f}, T={T:.0f}, N_c={N_c}, n_real={N_REAL}, "
        f"n_workers={n_workers} ===",
        flush=True,
    )

    try:
        real_args = [
            dict(L=L, w=w, alpha=alpha, zeta=zeta, T=T, N_c=N_c,
                 seed=seed + r * 999_983)
            for r in range(N_REAL)
        ]

        if n_workers > 1:
            # Parallel: pool.map returns all results at once; no partial saving.
            with mp.Pool(processes=min(n_workers, N_REAL)) as pool:
                real_results = pool.map(_run_one_realisation, real_args)
        else:
            # Serial: save partial checkpoint after each realization.
            completed = _load_partial(partial_file)
            if completed:
                print(
                    f"  Resuming from partial file: "
                    f"{len(completed)}/{N_REAL} realisations already done.",
                    flush=True,
                )

            real_results = [None] * N_REAL
            for r in range(N_REAL):
                if r in completed:
                    real_results[r] = completed[r]
                    print(f"  real {r+1}: loaded from checkpoint", flush=True)
                    continue
                res = _run_one_realisation(real_args[r])
                real_results[r] = res
                completed[r] = res
                _save_partial(partial_file, completed)

        # --- Aggregate results ---
        S_means       = np.full(N_REAL, np.nan)
        S_stds        = np.full(N_REAL, np.nan)
        thetas        = np.full(N_REAL, np.nan)
        ESSs          = np.full(N_REAL, np.nan)
        B_L_means     = np.full(N_REAL, np.nan)
        B_L_stds      = np.full(N_REAL, np.nan)
        # CMI tripartition components (per-realisation means/stds across clones)
        CMI_means     = np.full(N_REAL, np.nan)
        CMI_stds      = np.full(N_REAL, np.nan)
        S_AB_means    = np.full(N_REAL, np.nan)
        S_AB_stds     = np.full(N_REAL, np.nan)
        S_BC_means    = np.full(N_REAL, np.nan)
        S_BC_stds     = np.full(N_REAL, np.nan)
        S_B_means     = np.full(N_REAL, np.nan)
        S_B_stds      = np.full(N_REAL, np.nan)
        S_ABC_means   = np.full(N_REAL, np.nan)
        S_ABC_stds    = np.full(N_REAL, np.nan)
        n_T_means     = np.full(N_REAL, np.nan)
        chi_ks        = np.full(N_REAL, np.nan)
        S_renyi_2s    = np.full(N_REAL, np.nan)
        S_renyi_3s    = np.full(N_REAL, np.nan)
        S_renyi_2_stds= np.full(N_REAL, np.nan)
        S_renyi_3_stds= np.full(N_REAL, np.nan)
        corr_decays_list = []   # list of (r_array, c_array) tuples per realisation
        S_vars        = np.full(N_REAL, np.nan)
        covar_Sks     = np.full(N_REAL, np.nan)
        n_collapses_total    = 0
        n_js_fallbacks_total = 0
        can_compute_B_L = (L % 4 == 0)

        for r, res in enumerate(real_results):
            n_collapses_total += int(res.get("n_collapses", 0))
            if not res.get("ok", False):
                print(f"  real {r}: FAILED — {res.get('error', '?')}", flush=True)
                continue

            S_means[r]   = float(res["S_mean"])
            S_stds[r]    = float(res["S_std"])
            thetas[r]    = float(res["theta_hat"])
            ESSs[r]      = float(res["eff_sample_size"])
            n_T_means[r] = float(res.get("n_T_mean", float("nan")))
            chi_ks[r]    = float(res.get("chi_k",    float("nan")))
            S_renyi_2s[r]    = float(res.get("S_renyi_2", float("nan")))
            S_renyi_3s[r]    = float(res.get("S_renyi_3", float("nan")))
            S_renyi_2_stds[r]= float(res.get("S_renyi_2_std", float("nan")))
            S_renyi_3_stds[r]= float(res.get("S_renyi_3_std", float("nan")))
            r_arr  = np.asarray(res.get("corr_decay_r",    []), dtype=np.float64)
            cd_arr = np.asarray(res.get("corr_decay_mean", []), dtype=np.float64)
            if r_arr.size > 0:
                corr_decays_list.append((r_arr, cd_arr))
            S_vars[r]    = float(res.get("S_var",    float("nan")))
            covar_Sks[r] = float(res.get("covar_Sk", float("nan")))
            n_js_fallbacks_total += int(res.get("n_js_fallbacks", 0))

            if can_compute_B_L:
                comps = _batched_compute_B_L(res["final_covs"], L)
                bl = comps["B_L"]
                fm = np.isfinite(bl)
                if fm.any():
                    B_L_means[r] = float(np.mean(bl[fm]))
                    B_L_stds[r]  = float(np.std(bl[fm]))
                # CMI tripartition components — mean/std across clones
                for _name, _arr_means, _arr_stds in (
                    ("CMI",   CMI_means,   CMI_stds),
                    ("S_AB",  S_AB_means,  S_AB_stds),
                    ("S_BC",  S_BC_means,  S_BC_stds),
                    ("S_B",   S_B_means,   S_B_stds),
                    ("S_ABC", S_ABC_means, S_ABC_stds),
                ):
                    _val = comps[_name]
                    _fm  = np.isfinite(_val)
                    if _fm.any():
                        _arr_means[r] = float(np.mean(_val[_fm]))
                        _arr_stds[r]  = float(np.std(_val[_fm]))

            print(
                f"  real {r+1}: S={S_means[r]:.4f}, θ={thetas[r]:.4f}, "
                f"ESS={ESSs[r]:.0f}, k̄={n_T_means[r]:.4f}, "
                f"χ_k={chi_ks[r]:.4f}, C_Sk={covar_Sks[r]:.4f}",
                flush=True,
            )

        S_mean,   S_std_r,   S_err,   nS   = _nanstat(S_means)
        theta_mean,  _, theta_err, nth      = _nanstat(thetas)
        ESS_mean,    _, _,        _         = _nanstat(ESSs)
        B_L_mean, _,  B_L_err,   nBL       = _nanstat(B_L_means)
        # CMI tripartition reductions
        CMI_mean,  _, CMI_err,  _           = _nanstat(CMI_means)
        S_AB_mean, _, S_AB_err, _           = _nanstat(S_AB_means)
        S_BC_mean, _, S_BC_err, _           = _nanstat(S_BC_means)
        S_B_mean,  _, S_B_err,  _           = _nanstat(S_B_means)
        S_ABC_mean,_, S_ABC_err,_           = _nanstat(S_ABC_means)
        n_T_mean, _,  n_T_err,   _         = _nanstat(n_T_means)
        chi_k_mean, _, chi_k_err, _         = _nanstat(chi_ks)
        S_var_mean, _, S_var_err, _         = _nanstat(S_vars)
        covar_Sk_mean, _, covar_Sk_err, _   = _nanstat(covar_Sks)

        wall_time = time.time() - t_start
        dtau_mult_used = float(os.environ.get("PPS_DTAU_MULT", "1.0"))

        np.savez(
            output_file,
            task_id=task_id, L=L, lam=lam, alpha=alpha, w=w, zeta=zeta,
            T=T, N_c=N_c, n_real=N_REAL,
            dtau_mult=dtau_mult_used,
            S_mean=S_mean, S_std=S_std_r, S_err=S_err,
            theta_mean=theta_mean, theta_err=theta_err,
            ESS_mean=ESS_mean,
            B_L_mean=B_L_mean, B_L_err=B_L_err,
            # CMI tripartition components — for cleaner Binder-like crossings
            CMI_mean=CMI_mean, CMI_err=CMI_err,
            S_AB_mean=S_AB_mean, S_AB_err=S_AB_err,
            S_BC_mean=S_BC_mean, S_BC_err=S_BC_err,
            S_B_mean=S_B_mean,   S_B_err=S_B_err,
            S_ABC_mean=S_ABC_mean, S_ABC_err=S_ABC_err,
            # New observables
            n_T_mean=n_T_mean, n_T_err=n_T_err,
            chi_k_mean=chi_k_mean, chi_k_err=chi_k_err,
            S_var_mean=S_var_mean, S_var_err=S_var_err,
            covar_Sk_mean=covar_Sk_mean, covar_Sk_err=covar_Sk_err,
            # Per-realization arrays
            S_means_all=S_means, S_stds_all=S_stds,
            thetas_all=thetas, ESSs_all=ESSs,
            B_L_means_all=B_L_means, B_L_stds_all=B_L_stds,
            # Per-realization CMI components
            CMI_means_all=CMI_means,    CMI_stds_all=CMI_stds,
            S_AB_means_all=S_AB_means,  S_AB_stds_all=S_AB_stds,
            S_BC_means_all=S_BC_means,  S_BC_stds_all=S_BC_stds,
            S_B_means_all=S_B_means,    S_B_stds_all=S_B_stds,
            S_ABC_means_all=S_ABC_means, S_ABC_stds_all=S_ABC_stds,
            n_T_means_all=n_T_means, chi_ks_all=chi_ks,
            S_renyi_2_mean=float(np.nanmean(S_renyi_2s)) if not np.all(np.isnan(S_renyi_2s)) else float("nan"),
            S_renyi_3_mean=float(np.nanmean(S_renyi_3s)) if not np.all(np.isnan(S_renyi_3s)) else float("nan"),
            S_renyi_2_err=float(np.nanstd(S_renyi_2s)/max(1, np.sum(~np.isnan(S_renyi_2s))**0.5)) if not np.all(np.isnan(S_renyi_2s)) else float("nan"),
            S_renyi_3_err=float(np.nanstd(S_renyi_3s)/max(1, np.sum(~np.isnan(S_renyi_3s))**0.5)) if not np.all(np.isnan(S_renyi_3s)) else float("nan"),
            S_renyi_2s_all=S_renyi_2s, S_renyi_3s_all=S_renyi_3s,
            # Translation-averaged correlation: average across realisations, store as ragged.
            # If any realisation has data, save a single (r_array, c_array) using the longest.
            corr_decay_r=(corr_decays_list[0][0] if corr_decays_list else np.asarray([])),
            corr_decay_mean=(
                np.mean(np.stack([cd for _, cd in corr_decays_list], axis=0), axis=0)
                if len(corr_decays_list) > 0 else np.asarray([])
            ),
            S_vars_all=S_vars, covar_Sks_all=covar_Sks,
            n_collapses=n_collapses_total,
            n_valid_S=nS, n_valid_theta=nth, n_valid_B_L=nBL,
            n_js_fallbacks=n_js_fallbacks_total,
            wall_time=wall_time,
        )
        _write_summary_atomic(summary_file, dict(
            task_id=task_id, L=L, lam=lam, zeta=zeta, T=T, N_c=N_c,
            n_real=N_REAL, n_workers=n_workers,
            S_mean=S_mean, S_err=S_err,
            theta_mean=theta_mean, theta_err=theta_err,
            B_L_mean=B_L_mean, B_L_err=B_L_err,
            CMI_mean=CMI_mean, CMI_err=CMI_err,
            S_AB_mean=S_AB_mean, S_AB_err=S_AB_err,
            n_T_mean=n_T_mean, chi_k_mean=chi_k_mean,
            S_var_mean=S_var_mean, covar_Sk_mean=covar_Sk_mean,
            ESS_mean=ESS_mean, n_collapses=n_collapses_total,
            n_js_fallbacks=n_js_fallbacks_total,
            wall_time=wall_time, status="complete",
        ))

        # Clean up partial file on successful completion
        partial_file.unlink(missing_ok=True)

        print(
            f"clone task {task_id}: L={L}, λ={lam:.3f}, ζ={zeta:.2f}, "
            f"<S>={S_mean:.4f}±{S_err:.4f}, <B_L>={B_L_mean:.4f}±{B_L_err:.4f}, "
            f"<CMI>={CMI_mean:.4f}±{CMI_err:.4f}, "
            f"θ={theta_mean:.4f}±{theta_err:.4f}, "
            f"k̄={n_T_mean:.4f}, χ_k={chi_k_mean:.4f}, "
            f"C_Sk={covar_Sk_mean:.4f}, t={wall_time:.1f}s",
            flush=True,
        )
        return 0

    except Exception as exc:
        wall_time = time.time() - t_start
        try:
            _write_summary_atomic(summary_file, dict(
                task_id=task_id, L=L, lam=lam, zeta=zeta, T=T,
                status="failed", error=str(exc), wall_time=wall_time,
            ))
        except Exception:
            pass
        print(f"clone task {task_id}: FAILED — {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
