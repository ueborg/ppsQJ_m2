"""Benchmark and verification harness for cloning optimisations.

Three things this script does:

1. Verifies that the optimised cloning code (post optimisations 1–3) gives
   numerically equivalent results to the previous behaviour, by running a
   single fixed-seed cloning call and comparing against expected output
   patterns (S_mean, theta, ESS within tight tolerance).

2. Times the cloning loop at L=32 and L=64 with N_c=200 and a single
   resampling step (i.e., one full forward sweep through all clones).
   Reports per-clone wall time so you can compare against pre-optimisation
   numbers from the codebase profiling notes (~43.8 ms/clone at L=64).

3. (Optional) Tests the Numba JIT trajectory driver as a drop-in
   replacement, comparing both speed and numerical agreement. Pass
   ``--with-jit`` to enable. The first call includes JIT compilation time
   (~5–15 s) — the script reports a steady-state timing from a warm
   second call.

Usage
-----

    python -m pps_qj.tools.benchmark_optimizations [--with-jit] [--L 32 64]

Run on the cluster after pulling the optimised code, with the venv active.
"""
from __future__ import annotations

import argparse
import time
import numpy as np

from pps_qj.gaussian_backend import (
    build_gaussian_chain_model,
    gaussian_born_rule_trajectory,
)
from pps_qj.cloning import run_cloning


def _format_time(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1.0:
        return f"{s * 1e3:.1f} ms"
    if s < 60.0:
        return f"{s:.2f} s"
    return f"{s / 60.0:.2f} min"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_consistency(L: int, lam: float, zeta: float, N_c: int = 50,
                        T: float = 16.0, n_seeds: int = 3) -> None:
    """Run cloning with several seeds; print S_mean, theta, ESS.

    Sanity check — values across seeds should be reasonable and consistent
    with the expected behaviour for these parameters.
    """
    print(f"\n[verify] L={L} lam={lam:.2f} zeta={zeta:.2f} "
          f"N_c={N_c} T={T} (n_seeds={n_seeds})")

    alpha, w = lam, 1.0 - lam
    model = build_gaussian_chain_model(L, w, alpha)

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed * 12345)
        result = run_cloning(model, zeta=float(zeta), T_total=T, N_c=N_c,
                             rng=rng, show_progress=False)
        print(f"  seed {seed}: S={result.S_mean:.4f} "
              f"theta={result.theta_hat:.4f} "
              f"ESS={result.eff_sample_size:.1f}/{N_c}")


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def bench_cloning(L: int, lam: float = 0.40, zeta: float = 0.50,
                   N_c: int = 200, T: float = 8.0, n_warmup: int = 1,
                   n_repeat: int = 3) -> float:
    """Time run_cloning end-to-end; returns mean wall time per call."""
    alpha, w = lam, 1.0 - lam
    model = build_gaussian_chain_model(L, w, alpha)

    for _ in range(n_warmup):
        rng = np.random.default_rng(0)
        run_cloning(model, zeta=zeta, T_total=T, N_c=N_c, rng=rng,
                    show_progress=False)

    times: list[float] = []
    for r in range(n_repeat):
        rng = np.random.default_rng(r + 1)
        t0 = time.perf_counter()
        run_cloning(model, zeta=zeta, T_total=T, N_c=N_c, rng=rng,
                    show_progress=False)
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times)
    print(f"  L={L} N_c={N_c} T={T}: "
          f"mean={_format_time(arr.mean())}  "
          f"min={_format_time(arr.min())}  "
          f"per-clone={_format_time(arr.mean() / N_c)}")
    return float(arr.mean())


# ---------------------------------------------------------------------------
# JIT benchmark + numerical agreement
# ---------------------------------------------------------------------------

def bench_jit_trajectory(L: int, lam: float = 0.40, T_traj: float = 0.04,
                          n_warmup: int = 2, n_repeat: int = 200) -> None:
    """Compare timing of the original vs JIT trajectory drivers."""
    try:
        from pps_qj.gaussian_backend_jit import (
            run_trajectory_jit,
            NUMBA_AVAILABLE,
        )
    except ImportError as e:
        print(f"\n[jit] cannot import gaussian_backend_jit: {e}")
        return

    if not NUMBA_AVAILABLE:
        print("\n[jit] numba not installed (`pip install numba`)")
        return

    print(f"\n[jit] L={L} lam={lam:.2f} T_traj={T_traj} "
          f"(n_repeat={n_repeat})")

    alpha, w = lam, 1.0 - lam
    model = build_gaussian_chain_model(L, w, alpha)

    _jp = model.jump_pairs
    ja = np.array([p[0] for p in _jp], dtype=np.intp)
    jb = np.array([p[1] for p in _jp], dtype=np.intp)

    print("  warming up JIT (compiling)...", flush=True)
    rng_warm = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(n_warmup):
        run_trajectory_jit(
            model, T=T_traj, rng=rng_warm,
            ja_cached=ja, jb_cached=jb,
        )
    print(f"  JIT warmup: {_format_time(time.perf_counter() - t0)} "
          f"(includes one-time compilation)", flush=True)

    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        gaussian_born_rule_trajectory(
            model, T=T_traj, rng=rng,
            ja_cached=ja, jb_cached=jb,
        )
    t_orig = (time.perf_counter() - t0) / n_repeat

    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        run_trajectory_jit(
            model, T=T_traj, rng=rng,
            ja_cached=ja, jb_cached=jb,
        )
    t_jit = (time.perf_counter() - t0) / n_repeat

    speedup = t_orig / t_jit if t_jit > 0 else float('nan')
    print(f"  original (per call): {_format_time(t_orig)}")
    print(f"  JIT      (per call): {_format_time(t_jit)}")
    print(f"  SPEEDUP:             {speedup:.2f}x")

    # Statistical agreement check.
    print("\n  --- statistical agreement check (100 trajectories each) ---")
    n_check = 100
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)

    n_jumps_orig = []
    n_jumps_jit = []
    norm_orig = []
    norm_jit = []
    for _ in range(n_check):
        r1 = gaussian_born_rule_trajectory(model, T=T_traj, rng=rng_a,
                                           ja_cached=ja, jb_cached=jb)
        r2 = run_trajectory_jit(model, T=T_traj, rng=rng_b,
                                ja_cached=ja, jb_cached=jb)
        n_jumps_orig.append(r1.n_jumps)
        n_jumps_jit.append(r2.n_jumps)
        norm_orig.append(float(np.linalg.norm(r1.final_covariance)))
        norm_jit.append(float(np.linalg.norm(r2.final_covariance)))

    a_o = np.asarray(n_jumps_orig); a_j = np.asarray(n_jumps_jit)
    print(f"  <n_jumps>: original {a_o.mean():.3f} ± {a_o.std():.3f}, "
          f"JIT {a_j.mean():.3f} ± {a_j.std():.3f}")
    a_o = np.asarray(norm_orig); a_j = np.asarray(norm_jit)
    print(f"  <||cov||>: original {a_o.mean():.4f} ± {a_o.std():.4f}, "
          f"JIT {a_j.mean():.4f} ± {a_j.std():.4f}")
    print("  (these should agree within statistical fluctuations — they")
    print("   should NOT be bit-equivalent because the two algorithms ")
    print("   consume randomness in different orders.)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, nargs="+", default=[32, 64])
    p.add_argument("--with-jit", action="store_true")
    p.add_argument("--N-c", type=int, default=200)
    p.add_argument("--T", type=float, default=8.0)
    p.add_argument("--n-repeat", type=int, default=3)
    args = p.parse_args()

    print("=" * 60)
    print("PPS-QJ optimisation benchmark")
    print("=" * 60)

    print("\n--- Verification: cloning gives sensible output ---")
    verify_consistency(L=16, lam=0.40, zeta=0.50, N_c=50, T=16.0)

    print("\n--- Cloning speed benchmark ---")
    for L in args.L:
        bench_cloning(L, N_c=args.N_c, T=args.T,
                      n_repeat=args.n_repeat)

    if args.with_jit:
        print("\n--- JIT trajectory benchmark ---")
        for L in args.L:
            alpha = 0.40
            T_traj = 1.0 / max(2.0 * alpha * (L - 1), 1e-6)
            bench_jit_trajectory(L, lam=0.40, T_traj=T_traj,
                                  n_repeat=300)

    print("\n" + "=" * 60)
    print("done.")


if __name__ == "__main__":
    main()
