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
  * B_L = S_top * S_half averaged over the final clone population.

Jack-Sollich feedback control is enabled automatically when a pre-computed
backward pass file exists at ``<output_dir>/backward_<task_id:05d>.npz``.
Pre-compute it with ``scripts/habrok/precompute_backward_pass.py``.
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
from pps_qj.parallel.grid_pps import task_params_clone, NREAL_FOR_L


# ---------------------------------------------------------------------------
# Batched B_L computation
# ---------------------------------------------------------------------------

def _batched_compute_B_L(
    covs_list: list[np.ndarray],
    L: int,
) -> np.ndarray:
    """Batched B_L = S_top * S_half for an entire clone population.

    Replaces a serial loop of N_c calls to ``compute_all_observables``
    (5 eigvalsh calls each) with four batched eigvalsh operations — one per
    subsystem partition. The speedup is roughly N_c-fold for the eigvalsh
    stage, which dominates at large L and N_c.

    The partition follows Kells et al. (SciPost 2023) with 1-indexed sites and
    zero-based Majorana indices eta_{2(j-1)}, eta_{2(j-1)+1} for site j:

        A   = sites  1 .. L/4      Majorana  0        .. L/2 - 1   (size  L/2)
        B   = sites  L/4+1 .. L/2  Majorana  L/2      .. L   - 1   (size  L/2)
        D   = sites  L/2+1 .. 3L/4 Majorana  L        .. 3L/2 - 1  (size  L/2)
        AB  = A ∪ B                Majorana  0        .. L   - 1   (size  L  )
        BD  = B ∪ D                Majorana  L/2      .. 3L/2 - 1  (size  L  )
        ABD = A ∪ B ∪ D            Majorana  0        .. 3L/2 - 1  (size 3L/2)

    Note that all four subsystems have *contiguous* Majorana indices, so each
    can be extracted as a rectangular subblock slice — no fancy indexing needed.

    Parameters
    ----------
    covs_list : list of (2L, 2L) float64 arrays — clone covariance matrices
    L         : int — chain length; must be divisible by 4

    Returns
    -------
    (N_c,) float64 array; entries are np.nan where L % 4 != 0 or on failure.
    """
    N_c = len(covs_list)
    if L % 4 != 0:
        return np.full(N_c, np.nan, dtype=np.float64)

    # Stack all covariance matrices: (N_c, 2L, 2L)
    gamma_batch = np.stack(
        [np.asarray(c, dtype=np.float64) for c in covs_list], axis=0
    )

    # Majorana slice boundaries  (all contiguous, so plain slicing works)
    hL  = L // 2       # half-L = L/2
    tL  = 3 * L // 2   # three-halves-L = 3L/2

    # Subblock extraction (batch over N_c axis)
    sub_AB  = gamma_batch[:, :L,   :L  ]   # (N_c, L,    L   ) — also "half-chain"
    sub_BD  = gamma_batch[:, hL:hL+L, hL:hL+L]  # (N_c, L,    L   )
    sub_B   = gamma_batch[:, hL:L,  hL:L  ]   # (N_c, L/2,  L/2 )
    sub_ABD = gamma_batch[:, :tL,  :tL   ]   # (N_c, 3L/2, 3L/2)

    def _batch_entropy(sub: np.ndarray) -> np.ndarray:
        """Batched entropy from stacked (N_c, m, m) real antisymmetric matrices.

        Uses eigvalsh on iC (complex Hermitian) — same path as the optimised
        ``_batched_entanglement_entropy`` in cloning.py.
        """
        m   = sub.shape[-1]
        ell = m // 2
        eigs = np.linalg.eigvalsh((1j * sub).astype(np.complex128))  # (N_c, m) ascending
        nus  = np.abs(eigs[:, ell:])  # positive half (N_c, ell)
        nus  = np.clip(nus, 0.0, 1.0)
        p_p  = np.clip(0.5 * (1.0 + nus), 1e-15, 1.0 - 1e-15)
        p_m  = np.clip(0.5 * (1.0 - nus), 1e-15, 1.0 - 1e-15)
        return -np.sum(p_p * np.log2(p_p) + p_m * np.log2(p_m), axis=-1)  # (N_c,)

    try:
        S_AB  = _batch_entropy(sub_AB)   # = S_half
        S_BD  = _batch_entropy(sub_BD)
        S_B   = _batch_entropy(sub_B)
        S_ABD = _batch_entropy(sub_ABD)
        S_top = S_AB + S_BD - S_B - S_ABD
        return (S_top * S_AB).astype(np.float64)
    except np.linalg.LinAlgError:
        # Fallback to serial computation on numerical failure
        from pps_qj.observables import compute_all_observables
        jump_pairs_fallback: list = []  # B_L doesn't use jump_pairs
        result = np.empty(N_c, dtype=np.float64)
        for i, cov in enumerate(covs_list):
            try:
                obs = compute_all_observables(
                    np.asarray(cov, dtype=np.float64), L, jump_pairs_fallback
                )
                result[i] = float(obs["B_L"])
            except Exception:
                result[i] = np.nan
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_summary_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    tmp.replace(path)


def _try_load_backward_pass(output_dir: Path, task_id: int):
    """Return a LoadedBackwardPass if the pre-computed file exists, else None."""
    bwd_path = output_dir / f"backward_{task_id:05d}.npz"
    if not bwd_path.exists():
        return None
    try:
        from pps_qj.backward_pass_io import load_backward_pass
        bp = load_backward_pass(bwd_path)
        print(f"  [JS] loaded backward pass from {bwd_path.name}", flush=True)
        return bp
    except Exception as exc:
        print(f"  [JS] failed to load backward pass ({exc}) — using standard cloning", flush=True)
        return None


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

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
    # Realisation count is L-dependent (see NREAL_FOR_L in grid_pps.py).
    # L=64 uses N_REAL=2 to keep all tasks within a 48h wall-time limit.
    N_REAL = NREAL_FOR_L.get(L, 5)

    try:
        model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)

        # Attempt to load a pre-computed backward pass for Jack-Sollich control.
        # Falls back to standard cloning silently if the file is absent.
        backward_data = _try_load_backward_pass(output_dir, task_id)
        using_js = backward_data is not None

        S_means = np.full(N_REAL, np.nan, dtype=np.float64)
        S_stds = np.full(N_REAL, np.nan, dtype=np.float64)
        thetas = np.full(N_REAL, np.nan, dtype=np.float64)
        ESSs = np.full(N_REAL, np.nan, dtype=np.float64)
        B_L_means = np.full(N_REAL, np.nan, dtype=np.float64)
        B_L_stds = np.full(N_REAL, np.nan, dtype=np.float64)
        n_collapses_total = 0
        n_js_fallbacks_total = 0

        can_compute_B_L = (L % 4 == 0)

        print(
            f"\n=== clone task {task_id}: L={L}, λ={lam:.3f} (α={alpha:.3f}, w={w:.3f}), "
            f"ζ={zeta:.2f}, T={T:.0f}, N_c={N_c}, n_real={N_REAL}"
            + (" [JS]" if using_js else "")
            + " ===",
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
                        backward_data=backward_data,
                    )
                    S_means[r] = float(result.S_mean)
                    S_stds[r] = float(result.S_std)
                    thetas[r] = float(result.theta_hat)
                    ESSs[r] = float(result.eff_sample_size)
                    n_collapses_total += int(result.n_collapses)
                    n_js_fallbacks_total += int(result.n_js_fallbacks)

                    # Compute B_L across the final clone population using the
                    # batched eigvalsh path — avoids N_c serial eigvalsh calls.
                    if can_compute_B_L and result.final_covs:
                        bl_vals = _batched_compute_B_L(
                            [np.asarray(c, dtype=np.float64) for c in result.final_covs],
                            L,
                        )
                        finite_mask = np.isfinite(bl_vals)
                        if finite_mask.any():
                            B_L_means[r] = float(np.mean(bl_vals[finite_mask]))
                            B_L_stds[r]  = float(np.std(bl_vals[finite_mask]))

                    pbar.set_postfix({
                        "S": f"{result.S_mean:.3f}",
                        "θ": f"{result.theta_hat:+.3f}",
                        "ESS": f"{result.eff_sample_size:.0f}",
                        "B_L": f"{B_L_means[r]:.3f}" if np.isfinite(B_L_means[r]) else "nan",
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
        B_L_mean, B_L_std_across, B_L_err, n_valid_B_L = _nanstat(B_L_means)

        wall_time = time.time() - t_start

        np.savez(
            output_file,
            task_id=task_id, L=L, lam=lam, alpha=alpha, w=w, zeta=zeta,
            T=T, N_c=N_c, n_real=N_REAL,
            S_mean=S_mean, S_std=S_std_across, S_err=S_err,
            theta_mean=theta_mean, theta_std=theta_std_across, theta_err=theta_err,
            ESS_mean=ESS_mean,
            B_L_mean=B_L_mean, B_L_std=B_L_std_across, B_L_err=B_L_err,
            S_means_all=S_means, S_stds_all=S_stds,
            thetas_all=thetas, ESSs_all=ESSs,
            B_L_means_all=B_L_means, B_L_stds_all=B_L_stds,
            n_collapses=n_collapses_total,
            n_valid_S=n_valid_S, n_valid_theta=n_valid_theta, n_valid_B_L=n_valid_B_L,
            using_js=using_js,
            n_js_fallbacks=n_js_fallbacks_total,
            wall_time=wall_time,
        )

        summary = dict(
            task_id=task_id, L=L, lam=lam, zeta=zeta, T=T, N_c=N_c, n_real=N_REAL,
            S_mean=S_mean, S_err=S_err,
            theta_mean=theta_mean, theta_err=theta_err,
            B_L_mean=B_L_mean, B_L_err=B_L_err,
            ESS_mean=ESS_mean, n_collapses=n_collapses_total,
            using_js=using_js, n_js_fallbacks=n_js_fallbacks_total,
            wall_time=wall_time, status="complete",
        )
        _write_summary_atomic(summary_file, summary)

        print(
            f"clone task {task_id}: L={L}, λ={lam:.3f}, ζ={zeta:.2f}, T={T:.0f}, "
            f"N_c={N_c}, <S>={S_mean:.4f}±{S_err:.4f}, "
            f"<B_L>={B_L_mean:.4f}±{B_L_err:.4f}, "
            f"θ={theta_mean:.4f}±{theta_err:.4f}, n_coll={n_collapses_total}"
            + (f", js_fallbacks={n_js_fallbacks_total}" if using_js else "")
            + f", t={wall_time:.1f}s",
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
