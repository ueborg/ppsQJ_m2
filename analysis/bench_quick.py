#!/usr/bin/env python
"""Quick interactive backend benchmark (short window, ~5-10 min).

Answers two questions fast, at and near the *production shape*, without
running a full production task:

  1. scalar vs batched wall time at (L, N_c) = production shape, with a SHORT
     T, then linearly extrapolated to the production T to project per-task hours.
  2. where the time goes at L=128 (cProfile bucket fractions) -- the D4 question
     that gates the rank-2 QR rewrite.

This reuses the exact run/profile workers from diagnostic_suite, so the timings
reflect the real run_cloning path.

Usage (Habrok interactive node):
    cd ~/pps_qj && source ~/venvs/pps_qj/bin/activate
    python analysis/bench_quick.py                 # default shapes
    python analysis/bench_quick.py --T-short 2 --reps 1 --no-profile   # fastest
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from pathlib import Path

# Make the repo root importable whether run as a script or with -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from analysis.diagnostic_suite import _one_run, _profile_run, LAM_DEFAULT, ZETA_DEFAULT


def _bench_pair(L, N_c, T_short, reps, T_prod, n_workers_prod, N_real_prod):
    """Time scalar & batched at (L,N_c,T_short); project to per-task prod hours."""
    times = {"scalar": [], "batched": []}
    for backend in ("scalar", "batched"):
        for rep in range(reps):
            r = _one_run(dict(L=L, lam=LAM_DEFAULT, zeta=ZETA_DEFAULT, T=T_short,
                              N_c=N_c, seed=10 + rep, backend=backend,
                              record_entropy=False))
            if not r.get("ok"):
                print(f"    !! {backend} L={L} N_c={N_c} failed: {r.get('error')}")
                times[backend].append(float("nan"))
            else:
                times[backend].append(r["wall"])
    sm = float(np.nanmean(times["scalar"])); bm = float(np.nanmean(times["batched"]))
    speedup = sm / bm if bm > 0 else float("nan")
    # one realisation extrapolated to production T:
    scale = T_prod / T_short
    real_h_scalar = sm * scale / 3600.0
    real_h_batched = bm * scale / 3600.0
    # per-task wall = ceil(N_real / n_workers) * per-realisation wall
    waves = int(np.ceil(N_real_prod / max(1, n_workers_prod)))
    return dict(L=L, N_c=N_c, scalar_s=sm, batched_s=bm, speedup=speedup,
                real_h_scalar=real_h_scalar, real_h_batched=real_h_batched,
                task_h_scalar=real_h_scalar * waves,
                task_h_batched=real_h_batched * waves, waves=waves)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Quick backend benchmark")
    ap.add_argument("--shapes", type=str, default="96x200,128x250",
                    help="comma list of LxN_c shapes to time")
    ap.add_argument("--T-short", type=float, default=3.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--T-prod", type=float, default=100.0)
    ap.add_argument("--n-workers-prod", type=int, default=5,
                    help="CPUS_PER_TASK used in production (for task-hour estimate)")
    ap.add_argument("--N-real-prod", type=int, default=5,
                    help="realisations per production task")
    ap.add_argument("--profile", dest="profile", action="store_true", default=True)
    ap.add_argument("--no-profile", dest="profile", action="store_false")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    shapes = [(int(a), int(b)) for a, b in
              (s.split("x") for s in args.shapes.split(","))]

    print("=" * 72)
    print(f"QUICK BACKEND BENCHMARK   T_short={args.T_short}  reps={args.reps}  "
          f"-> projecting to T_prod={args.T_prod}")
    print(f"  (per-task hours assume {args.N_real_prod} realisations across "
          f"{args.n_workers_prod} workers)")
    print("=" * 72)
    print(f"{'L':>4} {'N_c':>5} {'scalar_s':>10} {'batched_s':>10} {'speedup':>8} "
          f"{'1real_h_sc':>11} {'task_h_sc':>10} {'task_h_ba':>10}")
    for (L, N_c) in shapes:
        r = _bench_pair(L, N_c, args.T_short, args.reps, args.T_prod,
                        args.n_workers_prod, args.N_real_prod)
        tag = ("batched FASTER" if r["speedup"] > 1.05 else
               "scalar faster" if r["speedup"] < 0.95 else "~equal")
        print(f"{L:>4} {N_c:>5} {r['scalar_s']:>10.2f} {r['batched_s']:>10.2f} "
              f"{r['speedup']:>7.2f}x {r['real_h_scalar']:>11.1f} "
              f"{r['task_h_scalar']:>10.1f} {r['task_h_batched']:>10.1f}   {tag}",
              flush=True)

    if args.profile:
        print("\n" + "=" * 72)
        print("HOT-PATH PROFILE at L=128 (scalar, N_c=30, T=2)  -- the D4 question")
        print("=" * 72)
        pr = _profile_run(dict(L=128, lam=LAM_DEFAULT, zeta=ZETA_DEFAULT,
                               T=2.0, N_c=30, seed=7))
        fr = pr["fractions"]
        for k in sorted(fr, key=lambda x: -fr[x]):
            if fr[k] > 0.005:
                print(f"  {k:<34} {fr[k]:6.1%}")
        orb = fr.get("orbitals_from_covariance", 0.0)
        verdict = ("WORTH the rank-2 QR rewrite (>30% in the post-jump eigh)"
                   if orb > 0.30 else
                   "NOT clearly worth the rewrite (post-jump eigh < 30% of wall)")
        print(f"\n  orbitals_from_covariance = {orb:.1%}  ->  {verdict}")
    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
