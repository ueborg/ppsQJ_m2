#!/usr/bin/env python
"""Drive the L=4 first-pass sweep sequentially (40 points).

For each (λ, ζ) in the primary grid, invokes the single-point driver via
the in-process Python API (not subprocess, to save repeated BLAS init
overhead). Each point runs on the full worker count given.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from scripts.run_exact_benchmark import run_benchmark


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--N-traj", dest="N_traj", type=int, default=2000)
    p.add_argument("--T", type=float, default=20.0)
    p.add_argument("--n-workers", dest="n_workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--seed-base", dest="seed_base", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("sweep_l4")

    lambdas = np.linspace(0.1, 0.9, 9)
    zetas = np.arange(0.1, 1.01, 0.1)
    total = len(lambdas) * len(zetas)

    t_start = time.perf_counter()
    for j, lam in enumerate(lambdas):
        for k, zeta in enumerate(zetas):
            idx = j * len(zetas) + k
            out = args.output_root / f"L{args.L}" / f"l{lam:.2f}_z{zeta:.2f}.npz"
            log.info("[%d/%d] L=%d λ=%.2f ζ=%.2f → %s",
                     idx + 1, total, args.L, lam, zeta, out)
            ns = argparse.Namespace(
                L=args.L,
                lambda_=float(lam),
                zeta=float(zeta),
                T=args.T,
                N_traj=args.N_traj,
                method="exact-doob",
                backward_method="sector",
                backward_samples=64,
                output=out,
                n_workers=args.n_workers,
                seed=args.seed_base + idx,
                checkpoint_every=500,
                resume=args.resume,
            )
            run_benchmark(ns)
    log.info("sweep complete in %.1fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
