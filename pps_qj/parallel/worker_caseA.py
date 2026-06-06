"""Case A cloning worker for a single (L, lambda_A, zeta) grid point.

Entry point::

    python -m pps_qj.parallel.worker_caseA <task_id> <output_dir>

Mirrors ``worker_clone_pps`` but drives the Case A backend
(``run_cloning_caseA`` + ``build_caseA_model``) on the ``grid_caseA`` task table.
Reuses the Case B observable/aggregation helpers verbatim -- ``_batched_compute_B_L``
is backend-agnostic, so B_L = CMI * S_AB is computed identically to Case B.

zeta = 1 fast path: delta_tau is set to T (one cloning window = one full
independent trajectory per clone), eliminating the per-window QR/eig overhead.
There is no resampling at zeta = 1, so this is exact, not an approximation.

Runs N_REAL independent realisations per task (serial with a resume checkpoint,
or a multiprocessing pool when PPS_N_WORKERS > 1).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from pps_qj.cloning import CloningCollapse, CloningResult
from pps_qj.cloning_caseA import run_cloning_caseA
from pps_qj.gaussian_backend_caseA import build_caseA_model
from pps_qj.parallel.grid_caseA import task_params_caseA, N_REAL
from pps_qj.parallel.worker_clone_pps import (
    _batched_compute_B_L,
    _nanstat,
    _write_summary_atomic,
    _load_partial,
    _save_partial,
    _n_workers_from_env,
)


def _run_one_realisation_caseA(args: dict) -> dict:
    """Run one full Case A cloning realisation; return a plain result dict."""
    L = int(args["L"])
    gamma_rate = float(args["gamma_rate"])
    alpha_rate = float(args["alpha_rate"])
    zeta = float(args["zeta"])
    T = float(args["T"])
    N_c = int(args["N_c"])
    seed = int(args["seed"])

    rng = np.random.default_rng(seed)
    model = build_caseA_model(L, gamma_rate, alpha_rate)

    entropy_stride = max(1, int(os.environ.get("PPS_ENTROPY_STRIDE", "1")))
    # zeta = 1: one window = one full trajectory per clone (no resampling).
    delta_tau = T if zeta >= 1.0 else None

    try:
        result: CloningResult = run_cloning_caseA(
            model, zeta=zeta, T_total=T, N_c=N_c, rng=rng,
            delta_tau=delta_tau, entropy_stride=entropy_stride,
        )
        return {
            "ok": True,
            "S_mean": float(result.S_mean),
            "S_std": float(result.S_std),
            "theta_hat": float(result.theta_hat),
            "eff_sample_size": float(result.eff_sample_size),
            "n_collapses": int(result.n_collapses),
            "n_T_mean": float(result.n_T_mean),
            "chi_k": float(result.chi_k),
            "S_var": float(result.S_var),
            "covar_Sk": float(result.covar_Sk),
            "min_ess_frac": float(result.min_ess_frac_postburnin),
            "n_ancestors": int(result.n_distinct_ancestors),
            "final_covs": [np.asarray(c, dtype=np.float64) for c in result.final_covs],
        }
    except CloningCollapse as exc:
        return {"ok": False, "error": str(exc), "n_collapses": 1}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "n_collapses": 0}


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 2:
        raise SystemExit(
            "usage: python -m pps_qj.parallel.worker_caseA <task_id> <output_dir>"
        )
    task_id = int(argv[0])
    output_dir = Path(argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"caseA_{task_id:05d}.npz"
    summary_file = output_dir / f"summary_caseA_{task_id:05d}.json"
    partial_file = output_dir / f"caseA_{task_id:05d}_partial.pkl"

    force_rerun = os.environ.get("PPS_FORCE_RERUN", "0") not in ("0", "", "false", "False")
    if output_file.exists() and not force_rerun:
        print(f"caseA task {task_id}: already done, skipping", flush=True)
        return 0

    t_start = time.time()
    task = task_params_caseA(task_id)
    L = int(task["L"])
    lam = float(task["lam"])
    gamma_rate = float(task["gamma_rate"])
    alpha_rate = float(task["alpha_rate"])
    zeta = float(task["zeta"])
    T = float(task["T"])
    N_c = int(task["N_c"])
    seed = int(task["seed"])
    # Calibration / trust-gate overrides. Point these runs at a SEPARATE
    # output_dir (they reuse the production filename caseA_<id>.npz):
    #   PPS_T_OVERRIDE=<T>    -- T-saturation sweep (one L=128 lam=0.5 z=0.10 pt)
    #   PPS_NC_OVERRIDE=<N_c> -- N_c 2-rung check (L=128 z=0.30 lam=0.50)
    _nc_ov = os.environ.get("PPS_NC_OVERRIDE")
    if _nc_ov not in (None, "", "0"):
        N_c = int(_nc_ov)
    _t_ov = os.environ.get("PPS_T_OVERRIDE")
    if _t_ov not in (None, ""):
        T = float(_t_ov)
    n_workers = _n_workers_from_env()

    print(
        f"\n=== caseA task {task_id}: L={L}, lambda_A={lam:.3f} "
        f"(gamma={gamma_rate:.3f}, alpha={alpha_rate:.3f}), zeta={zeta:.2f}, "
        f"T={T:.0f}, N_c={N_c}, n_real={N_REAL}, n_workers={n_workers} ===",
        flush=True,
    )
    if L % 4 != 0:
        print(f"  WARNING: L={L} not divisible by 4 -> B_L tripartition is NaN.", flush=True)

    try:
        real_args = [
            dict(L=L, gamma_rate=gamma_rate, alpha_rate=alpha_rate,
                 zeta=zeta, T=T, N_c=N_c, seed=seed + r * 999_983)
            for r in range(N_REAL)
        ]

        if n_workers > 1:
            with mp.Pool(processes=min(n_workers, N_REAL)) as pool:
                real_results = pool.map(_run_one_realisation_caseA, real_args)
        else:
            completed = _load_partial(partial_file)
            if completed:
                print(f"  Resuming: {len(completed)}/{N_REAL} realisations done.", flush=True)
            real_results = [None] * N_REAL
            for r in range(N_REAL):
                if r in completed:
                    real_results[r] = completed[r]
                    continue
                res = _run_one_realisation_caseA(real_args[r])
                real_results[r] = res
                completed[r] = res
                _save_partial(partial_file, completed)

        # --- Aggregate across realisations ---
        S_means = np.full(N_REAL, np.nan)
        thetas = np.full(N_REAL, np.nan)
        ESSs = np.full(N_REAL, np.nan)
        B_L_means = np.full(N_REAL, np.nan)
        CMI_means = np.full(N_REAL, np.nan)
        S_AB_means = np.full(N_REAL, np.nan)
        S_BC_means = np.full(N_REAL, np.nan)
        S_B_means = np.full(N_REAL, np.nan)
        S_ABC_means = np.full(N_REAL, np.nan)
        n_T_means = np.full(N_REAL, np.nan)
        chi_ks = np.full(N_REAL, np.nan)
        S_vars = np.full(N_REAL, np.nan)
        covar_Sks = np.full(N_REAL, np.nan)
        min_ess_fracs = np.full(N_REAL, np.nan)
        n_ancestors_all = np.full(N_REAL, np.nan)
        n_collapses_total = 0
        can_compute_B_L = (L % 4 == 0)

        for r, res in enumerate(real_results):
            n_collapses_total += int(res.get("n_collapses", 0))
            if not res.get("ok", False):
                print(f"  real {r}: FAILED -- {res.get('error', '?')}", flush=True)
                continue
            S_means[r] = res["S_mean"]
            thetas[r] = res["theta_hat"]
            ESSs[r] = res["eff_sample_size"]
            n_T_means[r] = res["n_T_mean"]
            chi_ks[r] = res["chi_k"]
            S_vars[r] = res["S_var"]
            covar_Sks[r] = res["covar_Sk"]
            min_ess_fracs[r] = res.get("min_ess_frac", np.nan)
            n_ancestors_all[r] = res.get("n_ancestors", np.nan)
            if can_compute_B_L:
                comps = _batched_compute_B_L(res["final_covs"], L)
                for _name, _arr in (("B_L", B_L_means), ("CMI", CMI_means),
                                    ("S_AB", S_AB_means), ("S_BC", S_BC_means),
                                    ("S_B", S_B_means), ("S_ABC", S_ABC_means)):
                    v = comps[_name]
                    fm = np.isfinite(v)
                    if fm.any():
                        _arr[r] = float(np.mean(v[fm]))
            print(f"  real {r+1}: S={S_means[r]:.4f} theta={thetas[r]:+.4f} "
                  f"B_L={B_L_means[r]:.4f} ESS={ESSs[r]:.0f} "
                  f"minESS={min_ess_fracs[r]:.2f}", flush=True)

        S_mean, _, S_err, nS = _nanstat(S_means)
        theta_mean, _, theta_err, _ = _nanstat(thetas)
        ESS_mean, _, _, _ = _nanstat(ESSs)
        B_L_mean, _, B_L_err, nBL = _nanstat(B_L_means)
        CMI_mean, _, CMI_err, _ = _nanstat(CMI_means)
        S_AB_mean, _, S_AB_err, _ = _nanstat(S_AB_means)
        S_BC_mean, _, S_BC_err, _ = _nanstat(S_BC_means)
        S_B_mean, _, S_B_err, _ = _nanstat(S_B_means)
        S_ABC_mean, _, S_ABC_err, _ = _nanstat(S_ABC_means)
        n_T_mean, _, n_T_err, _ = _nanstat(n_T_means)
        chi_k_mean, _, chi_k_err, _ = _nanstat(chi_ks)
        S_var_mean, _, S_var_err, _ = _nanstat(S_vars)
        covar_Sk_mean, _, covar_Sk_err, _ = _nanstat(covar_Sks)
        min_ess_frac_mean, _, _, _ = _nanstat(min_ess_fracs)
        wall_time = time.time() - t_start

        np.savez(
            output_file,
            task_id=task_id, L=L, lam=lam,
            gamma_rate=gamma_rate, alpha_rate=alpha_rate,
            zeta=zeta, T=T, N_c=N_c, n_real=N_REAL,
            S_mean=S_mean, S_err=S_err,
            theta_mean=theta_mean, theta_err=theta_err,
            ESS_mean=ESS_mean,
            B_L_mean=B_L_mean, B_L_err=B_L_err,
            CMI_mean=CMI_mean, CMI_err=CMI_err,
            S_AB_mean=S_AB_mean, S_AB_err=S_AB_err,
            S_BC_mean=S_BC_mean, S_BC_err=S_BC_err,
            S_B_mean=S_B_mean, S_B_err=S_B_err,
            S_ABC_mean=S_ABC_mean, S_ABC_err=S_ABC_err,
            n_T_mean=n_T_mean, n_T_err=n_T_err,
            chi_k_mean=chi_k_mean, chi_k_err=chi_k_err,
            S_var_mean=S_var_mean, S_var_err=S_var_err,
            covar_Sk_mean=covar_Sk_mean, covar_Sk_err=covar_Sk_err,
            min_ess_frac_mean=min_ess_frac_mean,
            # per-realisation arrays
            S_means_all=S_means, thetas_all=thetas, ESSs_all=ESSs,
            B_L_means_all=B_L_means, CMI_means_all=CMI_means,
            S_AB_means_all=S_AB_means, min_ess_fracs_all=min_ess_fracs,
            n_ancestors_all=n_ancestors_all,
            n_collapses=n_collapses_total,
            n_valid_S=nS, n_valid_B_L=nBL,
            wall_time=wall_time,
        )
        _write_summary_atomic(summary_file, dict(
            task_id=task_id, L=L, lam=lam, zeta=zeta, T=T, N_c=N_c,
            n_real=N_REAL, n_workers=n_workers,
            S_mean=S_mean, S_err=S_err,
            theta_mean=theta_mean, theta_err=theta_err,
            B_L_mean=B_L_mean, B_L_err=B_L_err,
            CMI_mean=CMI_mean, CMI_err=CMI_err,
            min_ess_frac_mean=min_ess_frac_mean,
            n_collapses=n_collapses_total,
            wall_time=wall_time, status="complete",
        ))
        partial_file.unlink(missing_ok=True)

        print(
            f"caseA task {task_id}: L={L}, lambda={lam:.3f}, zeta={zeta:.2f}, "
            f"<S>={S_mean:.4f}+/-{S_err:.4f}, <B_L>={B_L_mean:.4f}+/-{B_L_err:.4f}, "
            f"theta={theta_mean:+.4f}, minESS={min_ess_frac_mean:.2f}, "
            f"t={wall_time:.1f}s",
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
        print(f"caseA task {task_id}: FAILED -- {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
