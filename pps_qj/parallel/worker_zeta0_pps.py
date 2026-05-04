"""Worker for the ζ=0 no-click (postselected) benchmark scan.

Entry point::

    python -m pps_qj.parallel.worker_zeta0_pps <task_id> <output_dir>

Physical context
----------------
At ζ=0 the PPS measure concentrates on no-click trajectories.  The
conditional state evolves deterministically under the non-Hermitian
effective Hamiltonian H_eff = H - (iα/2) Σ P_j (renormalized to unit
norm at each step), converging to the ground state of H_eff as T→∞.

This is fully deterministic — no random numbers, N_REAL=1.  The
dominant cost per task is O(L³) (one matrix exponentiation to build
the one-step propagator M = exp(h_eff · δt), then n_steps matrix-
vector products with QR renormalization).

Key theoretical benchmark
-------------------------
Finite-size crossing of S_L/2 as a function of λ should satisfy

    λ_cross(L, ζ=0) ~ L^{-1/2}

from the postselected correlation length ξ_ps ~ (w/α)² = x^{-2}.
This is the cleanest analytic check in the whole dataset.

Convergence check
-----------------
S is computed at T/2 and T.  If |S(T) - S(T/2)| > 0.05 a warning is
printed and the `converged` flag is set False in the output.  Typical
convergence for λ ≤ 0.30 (topological phase, gap O(w)) is T ~ 5/α,
which is the default time_horizon_zeta0 formula.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import expm

from pps_qj.gaussian_backend import (
    build_gaussian_chain_model,
    covariance_from_orbitals,
    entanglement_entropy,
)
from pps_qj.parallel.grid_pps import task_params_zeta0


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------

def run_postselected_steadystate(
    model,
    T: float,
    dt: float = 1.0,
) -> dict:
    """Evolve the Gaussian state under H_eff for time T, renormalizing at
    each step, and return steady-state observables.

    The one-step propagator M = exp(h_eff · dt) is computed once with scipy
    expm (Padé approximation; numerically stable for any T via QR renorm).
    Orbitals are QR-normalized after each step to prevent overflow.

    Returns a dict with keys:
        S_final     : S_L/2 at t=T
        S_half      : S_L/2 at t=T/2  (convergence diagnostic)
        converged   : |S_final - S_half| < 0.05
        log_norm_T  : accumulated log-norm (proxy for partition function)
        n_steps     : total propagation steps taken
    """
    n_steps   = max(1, int(round(T / dt)))
    dt_eff    = T / n_steps
    half_step = n_steps // 2

    # Precompute M_dt = exp(h_eff * dt) once — O(L³) with large constant
    M_dt = expm(model.h_effective * dt_eff)

    orbitals     = model.orbitals0.copy()
    log_norm_acc = 0.0
    S_half       = float("nan")

    for k in range(n_steps):
        orbitals = M_dt @ orbitals
        Q, R     = np.linalg.qr(orbitals, mode="reduced")
        diag_abs = np.abs(np.diag(R))
        log_norm_acc += float(np.sum(np.log(np.maximum(diag_abs, 1e-300))))
        orbitals = Q

        if k == half_step - 1:
            cov_half = covariance_from_orbitals(orbitals)
            S_half   = entanglement_entropy(cov_half, model.L // 2)

    cov_final = covariance_from_orbitals(orbitals)
    S_final   = entanglement_entropy(cov_final, model.L // 2)
    converged = abs(S_final - S_half) < 0.05

    return dict(
        S_final=float(S_final),
        S_half=float(S_half),
        converged=bool(converged),
        log_norm_T=float(log_norm_acc),
        n_steps=int(n_steps),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_summary_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 2:
        raise SystemExit(
            "usage: python -m pps_qj.parallel.worker_zeta0_pps <task_id> <output_dir>"
        )
    task_id    = int(argv[0])
    output_dir = Path(argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file  = output_dir / f"zeta0_{task_id:04d}.npz"
    summary_file = output_dir / f"summary_zeta0_{task_id:04d}.json"

    if output_file.exists():
        print(f"zeta0 task {task_id}: already done, skipping", flush=True)
        return 0

    t_start = time.time()
    task    = task_params_zeta0(task_id)
    L       = int(task["L"])
    lam     = float(task["lam"])
    alpha   = float(task["alpha"])
    w       = float(task["w"])
    T       = float(task["T"])

    print(
        f"\n=== zeta0 task {task_id}: L={L}, λ={lam:.3f} (α={alpha:.3f}), "
        f"T={T:.0f} ===",
        flush=True,
    )

    try:
        model  = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
        result = run_postselected_steadystate(model, T=T)

        wall_time = time.time() - t_start

        if not result["converged"]:
            print(
                f"  WARNING: not converged — S(T/2)={result['S_half']:.4f}, "
                f"S(T)={result['S_final']:.4f}, ΔS={abs(result['S_final']-result['S_half']):.4f}",
                flush=True,
            )

        np.savez(
            output_file,
            task_id=task_id, L=L, lam=lam, alpha=alpha, w=w, T=T,
            S_final=result["S_final"],
            S_half=result["S_half"],
            converged=result["converged"],
            log_norm_T=result["log_norm_T"],
            n_steps=result["n_steps"],
            wall_time=wall_time,
        )
        _write_summary_atomic(summary_file, dict(
            task_id=task_id, L=L, lam=lam, alpha=alpha, w=w, T=T,
            S_final=result["S_final"],
            S_half=result["S_half"],
            converged=result["converged"],
            log_norm_T=result["log_norm_T"],
            wall_time=wall_time,
            status="complete",
        ))

        print(
            f"zeta0 task {task_id}: L={L}, λ={lam:.3f}, "
            f"S={result['S_final']:.4f} (S_half={result['S_half']:.4f}), "
            f"converged={result['converged']}, t={wall_time:.2f}s",
            flush=True,
        )
        return 0

    except Exception as exc:
        wall_time = time.time() - t_start
        try:
            _write_summary_atomic(summary_file, dict(
                task_id=task_id, L=L, lam=lam, T=T,
                status="failed", error=str(exc), wall_time=wall_time,
            ))
        except Exception:
            pass
        print(f"zeta0 task {task_id}: FAILED — {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
