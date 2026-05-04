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
from pps_qj.parallel.grid_pps import task_params_clone

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

    try:
        result: CloningResult = run_cloning(
            model, zeta=zeta, T_total=T, N_c=N_c, rng=rng,
            show_progress=False,
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

def _batched_compute_B_L(covs_list: list[np.ndarray], L: int) -> np.ndarray:
    """Batched B_L = S_topo * S_half across the final clone population."""
    N_c = len(covs_list)
    if L % 4 != 0:
        return np.full(N_c, np.nan, dtype=np.float64)

    gamma_batch = np.stack(
        [np.asarray(c, dtype=np.float64) for c in covs_list], axis=0
    )
    hL = L // 2
    tL = 3 * L // 2

    sub_AB  = gamma_batch[:, :L,       :L      ]
    sub_BD  = gamma_batch[:, hL:hL+L,  hL:hL+L]
    sub_B   = gamma_batch[:, hL:L,     hL:L    ]
    sub_ABD = gamma_batch[:, :tL,      :tL     ]

    def _batch_entropy(sub: np.ndarray) -> np.ndarray:
        ell = sub.shape[-1] // 2
        eigs = np.linalg.eigvalsh((1j * sub).astype(np.complex128))
        nus  = np.clip(np.abs(eigs[:, ell:]), 0.0, 1.0)
        p_p  = np.clip(0.5 * (1.0 + nus), 1e-15, 1.0 - 1e-15)
        p_m  = np.clip(0.5 * (1.0 - nus), 1e-15, 1.0 - 1e-15)
        return -np.sum(p_p * np.log2(p_p) + p_m * np.log2(p_m), axis=-1)

    try:
        S_AB  = _batch_entropy(sub_AB)
        S_BD  = _batch_entropy(sub_BD)
        S_B   = _batch_entropy(sub_B)
        S_ABD = _batch_entropy(sub_ABD)
        return ((S_AB + S_BD - S_B - S_ABD) * S_AB).astype(np.float64)
    except np.linalg.LinAlgError:
        from pps_qj.observables import compute_all_observables
        out = np.empty(N_c, dtype=np.float64)
        for i, cov in enumerate(covs_list):
            try:
                out[i] = float(compute_all_observables(
                    np.asarray(cov, dtype=np.float64), L, []
                )["B_L"])
            except Exception:
                out[i] = np.nan
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
            "usage: python -m pps_qj.parallel.worker_clone_pps <task_id> <output_dir>"
        )
    task_id    = int(argv[0])
    output_dir = Path(argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file  = output_dir / f"clone_{task_id:05d}.npz"
    summary_file = output_dir / f"summary_clone_{task_id:05d}.json"
    partial_file = output_dir / f"clone_{task_id:05d}_partial.pkl"

    if output_file.exists():
        print(f"clone task {task_id}: already done, skipping", flush=True)
        return 0

    t_start = time.time()
    task   = task_params_clone(task_id)
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
        n_T_means     = np.full(N_REAL, np.nan)
        chi_ks        = np.full(N_REAL, np.nan)
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
            S_vars[r]    = float(res.get("S_var",    float("nan")))
            covar_Sks[r] = float(res.get("covar_Sk", float("nan")))
            n_js_fallbacks_total += int(res.get("n_js_fallbacks", 0))

            if can_compute_B_L:
                bl = _batched_compute_B_L(res["final_covs"], L)
                fm = np.isfinite(bl)
                if fm.any():
                    B_L_means[r] = float(np.mean(bl[fm]))
                    B_L_stds[r]  = float(np.std(bl[fm]))

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
        n_T_mean, _,  n_T_err,   _         = _nanstat(n_T_means)
        chi_k_mean, _, chi_k_err, _         = _nanstat(chi_ks)
        S_var_mean, _, S_var_err, _         = _nanstat(S_vars)
        covar_Sk_mean, _, covar_Sk_err, _   = _nanstat(covar_Sks)

        wall_time = time.time() - t_start

        np.savez(
            output_file,
            task_id=task_id, L=L, lam=lam, alpha=alpha, w=w, zeta=zeta,
            T=T, N_c=N_c, n_real=N_REAL,
            S_mean=S_mean, S_std=S_std_r, S_err=S_err,
            theta_mean=theta_mean, theta_err=theta_err,
            ESS_mean=ESS_mean,
            B_L_mean=B_L_mean, B_L_err=B_L_err,
            # New observables
            n_T_mean=n_T_mean, n_T_err=n_T_err,
            chi_k_mean=chi_k_mean, chi_k_err=chi_k_err,
            S_var_mean=S_var_mean, S_var_err=S_var_err,
            covar_Sk_mean=covar_Sk_mean, covar_Sk_err=covar_Sk_err,
            # Per-realization arrays
            S_means_all=S_means, S_stds_all=S_stds,
            thetas_all=thetas, ESSs_all=ESSs,
            B_L_means_all=B_L_means, B_L_stds_all=B_L_stds,
            n_T_means_all=n_T_means, chi_ks_all=chi_ks,
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
