"""Single-task worker for dτ-doubling validation.

Runs one cloning realisation at given (L, λ, ζ, dtau_mult, seed) parameters
and writes a JSON result file with the diagnostics needed to decide whether
``delta_tau = mult * default_delta_tau`` is safe at scale.

Entry point::

    python -m pps_qj.tools.validate_dtau_worker \\
        <L> <lam> <zeta> <dtau_mult> <seed> <T> <N_c> <outdir>

Where ``dtau_mult=1.0`` reproduces the production value
(``1/(2*alpha*(L-1))``) and ``dtau_mult=2.0`` doubles it (halving n_steps).

The per-task JSON contains ``S_mean``, ``theta_hat``, ``min_ess_frac_postburnin``,
``n_distinct_ancestors`` and ``wall_time``; the aggregator script combines these
across seeds to produce a comparison table.

Called in parallel by submit_validate_dtau.sh via ``xargs -L 1``.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.cloning import run_cloning, CloningCollapse


def main() -> int:
    if len(sys.argv) < 9:
        raise SystemExit(
            "usage: python -m pps_qj.tools.validate_dtau_worker "
            "<L> <lam> <zeta> <dtau_mult> <seed> <T> <N_c> <outdir>"
        )
    L         = int(sys.argv[1])
    lam       = float(sys.argv[2])
    zeta      = float(sys.argv[3])
    dtau_mult = float(sys.argv[4])
    seed      = int(sys.argv[5])
    T         = float(sys.argv[6])
    N_c       = int(sys.argv[7])
    outdir    = sys.argv[8]

    alpha = lam
    w     = 1.0 - lam
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)

    dtau_default = 1.0 / max(2.0 * alpha * (L - 1), 1e-6)
    dtau_used    = dtau_mult * dtau_default

    fname = (
        f"dtau_L{L:03d}_lam{lam:.3f}_z{zeta:.2f}"
        f"_m{dtau_mult:.2f}_s{seed:06d}.json"
    )
    fpath = os.path.join(outdir, fname)

    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()

    try:
        result = run_cloning(
            model,
            zeta=zeta,
            T_total=T,
            N_c=N_c,
            rng=rng,
            delta_tau=dtau_used,
            show_progress=False,
        )
        wall = time.perf_counter() - t0
        out = {
            "ok":                       True,
            "L":                        L,
            "lam":                      lam,
            "zeta":                     zeta,
            "dtau_mult":                dtau_mult,
            "seed":                     seed,
            "T":                        T,
            "N_c":                      N_c,
            "dtau_default":             float(dtau_default),
            "dtau_used":                float(result.delta_tau),
            "n_steps":                  int(round(T / result.delta_tau)),
            "S_mean":                   float(result.S_mean),
            "S_std":                    float(result.S_std),
            "S_var":                    float(result.S_var),
            "theta_hat":                float(result.theta_hat),
            "eff_sample_size_final":    float(result.eff_sample_size),
            "min_ess_frac_postburnin":  float(result.min_ess_frac_postburnin),
            "n_distinct_ancestors":     int(result.n_distinct_ancestors),
            "n_collapses":              int(result.n_collapses),
            "n_T_mean":                 float(result.n_T_mean),
            "chi_k":                    float(result.chi_k),
            "wall_time":                float(wall),
        }
        msg = (
            f"[OK]   L={L:3d} λ={lam:.2f} ζ={zeta:.2f} "
            f"mult={dtau_mult:.2f} s={seed}: "
            f"S={result.S_mean:.4f} θ={result.theta_hat:+.3f} "
            f"min_ESS/N={result.min_ess_frac_postburnin:.3f} "
            f"anc={result.n_distinct_ancestors:>3d}/{N_c} "
            f"wall={wall:.1f}s"
        )

    except CloningCollapse as exc:
        wall = time.perf_counter() - t0
        out = {
            "ok":           False,
            "L":            L,
            "lam":          lam,
            "zeta":         zeta,
            "dtau_mult":    dtau_mult,
            "seed":         seed,
            "T":            T,
            "N_c":          N_c,
            "dtau_default": float(dtau_default),
            "dtau_used":    float(dtau_used),
            "error":        str(exc),
            "wall_time":    float(wall),
        }
        msg = (
            f"[FAIL] L={L:3d} λ={lam:.2f} ζ={zeta:.2f} "
            f"mult={dtau_mult:.2f} s={seed}: collapsed ({exc})"
        )

    os.makedirs(outdir, exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(out, f)
    print(msg, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
