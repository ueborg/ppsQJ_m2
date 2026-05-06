"""Benchmark and validate the batched cloning backend.

Three things this script does:

1. Statistically validates that ``backend='batched'`` gives the same
   distribution of S_mean and theta_hat as ``backend='scalar'``.  Runs
   ``n_seeds`` independent cloning realisations under each backend and
   checks that the two seed-mean estimates agree within a tolerance set
   by the seed-to-seed standard deviation.  Bit-exact equality is NOT
   expected — batched LAPACK accumulates floating point in a different
   order than serial calls.

2. Times each backend at the requested L values and reports the wall-time
   ratio.  The expected speedup is parameter-dependent: ~2x at low jump
   rate (low λ or low ζ), ~1.1–1.4x near the MIPT, ~1.0x at high jump
   rate.  Use this script before swapping the production default.

3. Reports the new diagnostics (per-step min ESS, number of distinct
   surviving ancestors) so you can sanity-check the run health independent
   of the speed comparison.

Usage
-----

    python -m pps_qj.tools.benchmark_batched \\
        --L 32 64 --N-c 200 --T 8.0 --n-seeds 8

Run on the cluster after pulling, with the venv active.  At L=64 with
N_c=200 and T=8.0, expect ~1–2 minutes per backend per seed.
"""
from __future__ import annotations

import argparse
import time
import numpy as np

from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.cloning import run_cloning


def _fmt_time(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1.0:
        return f"{s * 1e3:.1f} ms"
    if s < 60.0:
        return f"{s:.2f} s"
    return f"{s / 60.0:.2f} min"


def _run_one(model, zeta, T, N_c, seed, backend):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    res = run_cloning(
        model, zeta=zeta, T_total=T, N_c=N_c, rng=rng,
        record_entropy=True, show_progress=False,
        backend=backend,
    )
    elapsed = time.perf_counter() - t0
    return res, elapsed


def compare_backends(
    L: int,
    lam: float,
    zeta: float,
    N_c: int,
    T: float,
    n_seeds: int,
    base_seed: int = 12345,
) -> None:
    """Run scalar and batched n_seeds times each and compare."""
    alpha = float(lam)
    w_param = float(1.0 - lam)
    model = build_gaussian_chain_model(L=L, w=w_param, alpha=alpha)

    print(f"\n[compare] L={L} λ={lam:.2f} ζ={zeta:.2f} "
          f"N_c={N_c} T={T} n_seeds={n_seeds}")
    print(f"          α={alpha:.3f} w={w_param:.3f}")

    S_scalar = np.zeros(n_seeds)
    S_batched = np.zeros(n_seeds)
    th_scalar = np.zeros(n_seeds)
    th_batched = np.zeros(n_seeds)
    t_scalar = np.zeros(n_seeds)
    t_batched = np.zeros(n_seeds)
    ess_scalar = np.zeros(n_seeds)
    ess_batched = np.zeros(n_seeds)
    anc_scalar = np.zeros(n_seeds, dtype=int)
    anc_batched = np.zeros(n_seeds, dtype=int)

    for k in range(n_seeds):
        seed = base_seed + 1000 * k

        res_s, t_s = _run_one(model, zeta, T, N_c, seed, "scalar")
        res_b, t_b = _run_one(model, zeta, T, N_c, seed, "batched")

        S_scalar[k]   = res_s.S_mean
        S_batched[k]  = res_b.S_mean
        th_scalar[k]  = res_s.theta_hat
        th_batched[k] = res_b.theta_hat
        t_scalar[k]   = t_s
        t_batched[k]  = t_b
        ess_scalar[k]  = res_s.min_ess_frac_postburnin
        ess_batched[k] = res_b.min_ess_frac_postburnin
        anc_scalar[k]  = res_s.n_distinct_ancestors
        anc_batched[k] = res_b.n_distinct_ancestors

        print(f"  seed {k:2d} (s={seed:>8d}): "
              f"S_sc={res_s.S_mean:.4f}  S_ba={res_b.S_mean:.4f}    "
              f"θ_sc={res_s.theta_hat:+.4f}  θ_ba={res_b.theta_hat:+.4f}    "
              f"t_sc={_fmt_time(t_s):>8s}  t_ba={_fmt_time(t_b):>8s}    "
              f"ratio={t_s / t_b if t_b > 0 else float('nan'):.2f}x")

    # --- Statistical agreement: |Δ⟨S⟩| vs sqrt(σ_s² + σ_b²)/sqrt(n) ---
    S_diff   = float(np.mean(S_batched)  - np.mean(S_scalar))
    th_diff  = float(np.mean(th_batched) - np.mean(th_scalar))
    S_se     = float(np.sqrt(np.var(S_scalar,  ddof=1) + np.var(S_batched,  ddof=1)) / np.sqrt(n_seeds))
    th_se    = float(np.sqrt(np.var(th_scalar, ddof=1) + np.var(th_batched, ddof=1)) / np.sqrt(n_seeds))
    S_z      = abs(S_diff)  / S_se  if S_se  > 0.0 else float("inf")
    th_z     = abs(th_diff) / th_se if th_se > 0.0 else float("inf")

    print(f"\n  --- statistical agreement (n={n_seeds}) ---")
    print(f"  ⟨S_batched⟩ - ⟨S_scalar⟩ = {S_diff:+.5f}   "
          f"SE = {S_se:.5f}   |z| = {S_z:.2f}  "
          f"{'PASS' if S_z < 3.0 else 'FAIL (>3σ)'}")
    print(f"  ⟨θ_batched⟩ - ⟨θ_scalar⟩ = {th_diff:+.5f}   "
          f"SE = {th_se:.5f}   |z| = {th_z:.2f}  "
          f"{'PASS' if th_z < 3.0 else 'FAIL (>3σ)'}")

    # --- Wall time ---
    print(f"\n  --- wall time ---")
    print(f"  scalar:  mean={_fmt_time(t_scalar.mean())}   "
          f"min={_fmt_time(t_scalar.min())}")
    print(f"  batched: mean={_fmt_time(t_batched.mean())}   "
          f"min={_fmt_time(t_batched.min())}")
    if t_batched.mean() > 0:
        print(f"  speedup: {t_scalar.mean() / t_batched.mean():.2f}x "
              f"(min/min: {t_scalar.min() / t_batched.min():.2f}x)")

    # --- New diagnostics (independent of backend choice) ---
    print(f"\n  --- run health (averaged over seeds) ---")
    print(f"  min ESS / N_c      :  scalar={np.nanmean(ess_scalar):.3f}   "
          f"batched={np.nanmean(ess_batched):.3f}")
    print(f"  distinct ancestors :  scalar={anc_scalar.mean():.1f}/{N_c}   "
          f"batched={anc_batched.mean():.1f}/{N_c}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=int, nargs="+", default=[32, 64],
                   help="System sizes to benchmark.")
    p.add_argument("--lam", type=float, default=0.40,
                   help="α/(α+w); default 0.40 sits in the typical phase boundary region.")
    p.add_argument("--zeta", type=float, default=0.50,
                   help="Post-selection parameter; default 0.50.")
    p.add_argument("--N-c", type=int, default=200,
                   help="Number of clones.")
    p.add_argument("--T", type=float, default=8.0,
                   help="Total simulation time.")
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Number of independent seeds per backend.")
    p.add_argument("--base-seed", type=int, default=12345,
                   help="Base RNG seed.")
    args = p.parse_args()

    print("=" * 64)
    print("Batched-cloning validation and speed benchmark")
    print("=" * 64)
    print(f"params: λ={args.lam:.2f}  ζ={args.zeta:.2f}  "
          f"N_c={args.N_c}  T={args.T}  n_seeds={args.n_seeds}")

    for L in args.L:
        compare_backends(
            L=L, lam=args.lam, zeta=args.zeta,
            N_c=args.N_c, T=args.T, n_seeds=args.n_seeds,
            base_seed=args.base_seed,
        )

    print("\n" + "=" * 64)
    print("done.")
    print("=" * 64)


if __name__ == "__main__":
    main()
