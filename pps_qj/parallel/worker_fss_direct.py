"""Direct (L, lam, zeta) worker for the FSS test runs.

Entry point::

    python -m pps_qj.parallel.worker_fss_direct <L> <lam> <zeta> <output_dir>

Purpose: run the cloning algorithm for an arbitrary (L, lam, zeta) triple
that is NOT in any pre-defined grid, using the same N_c/T/seed logic as
the FST grid (worker_clone_v2_fst_pps.py).

This exists because the FST grid (submit_clone_v2_fst.sh) was designed to
test the now-retracted zeta~0.143 separatrix and does not include zeta=0.30,
which is the key point for the 1/L^2 finite-size scaling test.

The output .npz filename encodes (L, lam, zeta) rather than a task ID so it
cannot collide with the main or FST grids.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from pps_qj.cloning import CloningCollapse, CloningResult, run_cloning
from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.parallel.grid_pps import (
    _seed,
    nc_for_L_fst,
    time_horizon_fst,
)
from pps_qj.parallel.worker_clone_pps import (
    N_REAL,
    _n_workers_from_env,
    _run_one_realisation,
    _nanstat,
    _write_summary_atomic,
)

# ── output filename ──────────────────────────────────────────────────────────

def _output_stem(L: int, lam: float, zeta: float) -> str:
    return f"fss_L{L:03d}_lam{lam:.4f}_zeta{zeta:.4f}".replace(".", "p")


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 4:
        raise SystemExit(
            "usage: python -m pps_qj.parallel.worker_fss_direct "
            "<L> <lam> <zeta> <output_dir>"
        )

    L          = int(argv[0])
    lam        = float(argv[1])
    zeta       = float(argv[2])
    output_dir = Path(argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    stem         = _output_stem(L, lam, zeta)
    output_file  = output_dir / f"{stem}.npz"
    summary_file = output_dir / f"summary_{stem}.json"

    if output_file.exists():
        print(f"fss task L={L} lam={lam} zeta={zeta}: already done, skipping",
              flush=True)
        return 0

    alpha     = lam                   # our convention: lam = alpha, w = 1 - lam
    w         = 1.0 - lam
    T         = time_horizon_fst(L, alpha)
    N_c       = nc_for_L_fst(L)
    seed      = _seed(L, lam, zeta)
    n_workers = _n_workers_from_env()

    print(
        f"\n=== fss direct: L={L}, λ={lam:.4f} (α={alpha:.4f}, w={w:.4f}), "
        f"ζ={zeta:.4f}, T={T:.0f}, N_c={N_c}, n_real={N_REAL}, "
        f"n_workers={n_workers} ===",
        flush=True,
    )

    t_start = time.time()

    try:
        real_args = [
            dict(L=L, w=w, alpha=alpha, zeta=zeta, T=T, N_c=N_c,
                 seed=seed + r * 999_983)
            for r in range(N_REAL)
        ]

        if n_workers > 1:
            with mp.Pool(processes=min(n_workers, N_REAL)) as pool:
                real_results = pool.map(_run_one_realisation, real_args)
        else:
            real_results = [_run_one_realisation(a) for a in real_args]

    except CloningCollapse as exc:
        print(f"COLLAPSE: {exc}", flush=True)
        return 1

    # ── aggregate across realisations ───────────────────────────────────────
    def _agg(key: str):
        return _nanstat(np.array([r[key] for r in real_results
                                  if r.get(key) is not None]))

    S_mean, S_std_r, S_err, _ = _agg("S_mean")
    S_var_m, *_               = _agg("S_var")
    n_T_mean, *_              = _agg("n_T_mean")
    chi_k_m, *_               = _agg("chi_k")

    elapsed = time.time() - t_start
    print(
        f"  done: S={S_mean:.4f}±{S_err:.4f}  elapsed={elapsed/3600:.2f}h",
        flush=True,
    )

    np.savez_compressed(
        output_file,
        L=L, lam=lam, alpha=alpha, w=w, zeta=zeta, T=T, N_c=N_c,
        S_mean=S_mean, S_std=S_std_r, S_err=S_err,
        S_var=S_var_m, n_T_mean=n_T_mean, chi_k=chi_k_m,
        elapsed=elapsed,
    )

    _write_summary_atomic(summary_file, dict(
        L=L, lam=lam, alpha=alpha, w=w, zeta=zeta, T=T, N_c=N_c,
        S_mean=float(S_mean), S_err=float(S_err), elapsed=elapsed,
    ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
