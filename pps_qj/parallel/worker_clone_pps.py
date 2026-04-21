"""Cloning-cross-validation worker for a single (L, lambda, zeta) grid point.

Entry point::

    python -m pps_qj.parallel.worker_clone_pps <task_id> <output_dir>

For each task, runs ``n_real`` = 5 independent cloning realisations (distinct
seeds) with parameters drawn from the clone grid, then records:
  * S_mean and S_std per realisation (already averaged over clones with
    weighting),
  * theta_hat per realisation,
  * eff_sample_size per realisation,
  * n_collapses total,
  * final ensemble-averaged S across realisations.

``CloningResult`` does not expose per-clone final covariances, so only the
scalars S_mean and theta_hat are recorded (B_L cannot be computed from the
cloning output in the current API — this is documented in the spec).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from pps_qj.cloning import CloningCollapse, run_cloning
from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.parallel.grid_pps import task_params_clone


N_REAL = 5  # independent realisations per grid point


def _write_summary_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    tmp.replace(path)


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 2:
        raise SystemExit(
            "usage: python -m pps_qj.parallel.worker_clone_pps <task_id> <output_dir>"
        )
    task_id = int(argv[0])
    output_dir = Path(argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"clone_{task_id:05d}.npz"
    summary_file = output_dir / f"summary_clone_{task_id:05d}.json"

    if output_file.exists():
        print(f"clone task {task_id}: already done, skipping", flush=True)
        return 0

    t_start = time.time()
    task = task_params_clone(task_id)
    L = int(task["L"])
    lam = float(task["lam"])
    alpha = float(task["alpha"])
    w = float(task["w"])
    zeta = float(task["zeta"])
    T = float(task["T"])
    N_c = int(task["N_c"])
    seed = int(task["seed"])

    try:
        model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)

        S_means = np.full(N_REAL, np.nan, dtype=np.float64)
        S_stds = np.full(N_REAL, np.nan, dtype=np.float64)
        thetas = np.full(N_REAL, np.nan, dtype=np.float64)
        ESSs = np.full(N_REAL, np.nan, dtype=np.float64)
        n_collapses_total = 0

        print(
            f"\n=== clone task {task_id}: L={L}, λ={lam:.3f} (α={alpha:.3f}, w={w:.3f}), "
            f"ζ={zeta:.2f}, T={T:.0f}, N_c={N_c}, n_real={N_REAL} ===",
            flush=True,
        )

        with tqdm(total=N_REAL, desc=f"L={L} λ={lam:.2f} ζ={zeta:.2f}", unit="real") as pbar:
            for r in range(N_REAL):
                rng_r = np.random.default_rng(seed + r * 999_983)
                try:
                    result = run_cloning(
                        model, zeta=zeta, T_total=T, N_c=N_c, rng=rng_r,
                        show_progress=True,
                        progress_desc=f"  real {r+1}/{N_REAL}",
                    )
                    S_means[r] = float(result.S_mean)
                    S_stds[r] = float(result.S_std)
                    thetas[r] = float(result.theta_hat)
                    ESSs[r] = float(result.eff_sample_size)
                    n_collapses_total += int(result.n_collapses)
                    pbar.set_postfix({
                        "S": f"{result.S_mean:.3f}",
                        "θ": f"{result.theta_hat:+.3f}",
                        "ESS": f"{result.eff_sample_size:.0f}",
                    })
                except CloningCollapse:
                    n_collapses_total += 1
                pbar.update(1)

        # Aggregate across realisations.
        def _nanstat(a: np.ndarray) -> tuple[float, float, float, int]:
            a = a[~np.isnan(a)]
            n = int(a.size)
            if n == 0:
                return float("nan"), float("nan"), float("nan"), 0
            return (
                float(np.mean(a)),
                float(np.std(a)),
                float(np.std(a) / np.sqrt(n)),
                n,
            )

        S_mean, S_std_across, S_err, n_valid_S = _nanstat(S_means)
        theta_mean, theta_std_across, theta_err, n_valid_theta = _nanstat(thetas)
        ESS_mean, _, _, _ = _nanstat(ESSs)

        wall_time = time.time() - t_start

        np.savez(
            output_file,
            task_id=task_id, L=L, lam=lam, alpha=alpha, w=w, zeta=zeta,
            T=T, N_c=N_c, n_real=N_REAL,
            S_mean=S_mean, S_std=S_std_across, S_err=S_err,
            theta_mean=theta_mean, theta_std=theta_std_across, theta_err=theta_err,
            ESS_mean=ESS_mean,
            S_means_all=S_means, S_stds_all=S_stds,
            thetas_all=thetas, ESSs_all=ESSs,
            n_collapses=n_collapses_total,
            n_valid_S=n_valid_S, n_valid_theta=n_valid_theta,
            wall_time=wall_time,
        )

        summary = dict(
            task_id=task_id, L=L, lam=lam, zeta=zeta, T=T, N_c=N_c, n_real=N_REAL,
            S_mean=S_mean, S_err=S_err,
            theta_mean=theta_mean, theta_err=theta_err,
            ESS_mean=ESS_mean, n_collapses=n_collapses_total,
            wall_time=wall_time, status="complete",
        )
        _write_summary_atomic(summary_file, summary)

        print(
            f"clone task {task_id}: L={L}, λ={lam:.3f}, ζ={zeta:.2f}, T={T:.0f}, "
            f"N_c={N_c}, <S>={S_mean:.4f}±{S_err:.4f}, "
            f"θ={theta_mean:.4f}±{theta_err:.4f}, n_coll={n_collapses_total}, "
            f"t={wall_time:.1f}s",
            flush=True,
        )
        return 0

    except Exception as exc:
        wall_time = time.time() - t_start
        err_summary = dict(
            task_id=task_id, L=L, lam=lam, zeta=zeta, T=T,
            status="failed", error=str(exc), wall_time=wall_time,
        )
        try:
            _write_summary_atomic(summary_file, err_summary)
        except Exception:
            pass
        print(f"clone task {task_id}: FAILED — {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
