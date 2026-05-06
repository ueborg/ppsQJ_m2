"""Single-task worker for T-cap validation.

Runs one cloning realisation at extended T with no burn-in, recording the
full S_history and ESS_history arrays for offline saturation analysis.

Entry point::

    python -m pps_qj.tools.validate_tcap_worker \\
        <L> <lam> <zeta> <T> <seed> <N_c> <outdir>

Output is a compressed .npz file (S_history can be ~10⁴ entries at L=128).

Called in parallel by submit_validate_tcap.sh via ``xargs -L 1``.

After the job completes, run aggregate_tcap.py to find the saturation time
t_sat such that |S(t_sat) - S(T)| < tol·S(T) and quote
T_safe = ceiling(2 * t_sat / round_to(0.1)) for production use.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.cloning import run_cloning, CloningCollapse


def main() -> int:
    if len(sys.argv) < 8:
        raise SystemExit(
            "usage: python -m pps_qj.tools.validate_tcap_worker "
            "<L> <lam> <zeta> <T> <seed> <N_c> <outdir>"
        )
    L      = int(sys.argv[1])
    lam    = float(sys.argv[2])
    zeta   = float(sys.argv[3])
    T      = float(sys.argv[4])
    seed   = int(sys.argv[5])
    N_c    = int(sys.argv[6])
    outdir = sys.argv[7]

    alpha = lam
    w     = 1.0 - lam
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)

    fname = (
        f"tcap_L{L:03d}_lam{lam:.3f}_z{zeta:.2f}"
        f"_T{T:.0f}_s{seed:06d}.npz"
    )
    fpath = os.path.join(outdir, fname)
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()

    try:
        result = run_cloning(
            model,
            zeta=zeta,
            T_total=T,
            N_c=N_c,
            rng=rng,
            n_burnin_frac=0.0,         # full S_history retained
            record_entropy=True,
            show_progress=False,
        )
        wall = time.perf_counter() - t0
        np.savez_compressed(
            fpath,
            L=L, lam=lam, zeta=zeta, T=T, seed=seed, N_c=N_c,
            delta_tau=float(result.delta_tau),
            S_history=result.S_history,
            S_sq_history=result.S_sq_history,
            ess_history=result.ess_history,
            n_T_mean_history=result.n_T_mean_history,
            wall_time=float(wall),
            S_mean_full=float(result.S_mean),    # full-window mean (no burn-in)
            theta_hat=float(result.theta_hat),
            min_ess_frac=float(result.min_ess_frac_postburnin),
            n_distinct_ancestors=int(result.n_distinct_ancestors),
            n_collapses=int(result.n_collapses),
            ok=True,
        )
        print(
            f"[OK]   L={L:3d} λ={lam:.2f} ζ={zeta:.2f} T={T:5.0f} s={seed}: "
            f"S_full={result.S_mean:.4f} "
            f"min_ESS/N={result.min_ess_frac_postburnin:.3f} "
            f"anc={result.n_distinct_ancestors:>3d}/{N_c} "
            f"wall={wall:.1f}s",
            flush=True,
        )

    except CloningCollapse as exc:
        wall = time.perf_counter() - t0
        np.savez_compressed(
            fpath,
            L=L, lam=lam, zeta=zeta, T=T, seed=seed, N_c=N_c,
            ok=False, error=str(exc), wall_time=float(wall),
        )
        print(
            f"[FAIL] L={L:3d} λ={lam:.2f} ζ={zeta:.2f} T={T:5.0f} s={seed}: "
            f"collapsed ({exc})",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
