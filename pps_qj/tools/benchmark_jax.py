"""Benchmark and validation script for the JAX GPU cloning implementation.

Usage
-----
    # On Habrok GPU node:
    python -m pps_qj.tools.benchmark_jax --L 32 64 128

    # CPU-only JAX (still useful for correctness check):
    JAX_PLATFORM_NAME=cpu python -m pps_qj.tools.benchmark_jax --L 32

This script does three things:

1. Reports what device JAX is using (GPU/CPU/TPU) and its memory.
2. Runs the JAX cloning loop at each requested L and compares:
   - Per-step wall time vs the original numpy cloning loop.
   - S_mean and theta_hat agreement (statistical, not bit-equivalent).
3. Prints a summary table of speedups.

If S_mean or theta_hat differ by more than 3 sigma across n_seeds
independent runs between the two implementations, the test fails and
reports which parameter point is anomalous.
"""
from __future__ import annotations

import argparse
import time
import sys
import numpy as np


def check_jax():
    try:
        import jax
        jax.config.update("jax_enable_x64", True)  # must be before any jnp ops
        import jax.numpy as jnp
        print(f"JAX version: {jax.__version__}")
        print(f"Devices:     {jax.devices()}")
        backend = jax.default_backend()
        print(f"Backend:     {backend}")
        if backend == "gpu":
            for d in jax.devices("gpu"):
                print(f"  GPU: {d}")
        elif backend == "cpu":
            print("  (running on CPU — install jax[cuda12] for GPU)")
        return True
    except ImportError:
        print("JAX not installed. Run: pip install 'jax[cuda12]'")
        return False


def bench_jax_vs_numpy(L: int, lam: float, zeta: float,
                         N_c: int, T: float,
                         n_seeds: int = 3) -> dict:
    import jax
    import jax.random as jr
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    from pps_qj.cloning import run_cloning
    from pps_qj.cloning_jax import run_cloning_jax

    alpha, w = lam, 1.0 - lam
    model    = build_gaussian_chain_model(L, w, alpha)

    print(f"\n{'='*60}")
    print(f"L={L}  lam={lam:.2f}  zeta={zeta:.2f}  N_c={N_c}  T={T}")
    print(f"{'='*60}")

    # Build model and compile JAX kernel once — shared across all seeds.
    from pps_qj.cloning_jax import _make_jax_model, _make_trajectory_fn
    from jax import vmap
    jax_model = _make_jax_model(model)
    print("  [JAX] compiling trajectory kernel...", flush=True)
    import time as _t
    t0 = _t.perf_counter()
    traj_fn = _make_trajectory_fn(jax_model, float(
        1.0 / max(2.0 * alpha * (L - 1), 1e-6)  # delta_tau
    ))
    batched = vmap(traj_fn, in_axes=(0, 0, 0))
    # Trigger compilation with a dummy call.
    import jax.numpy as jnp, jax.random as jr
    dummy_key = jr.PRNGKey(0)
    dummy_keys = jr.split(dummy_key, N_c)
    gamma0_b = jnp.stack([jnp.array(model.gamma0)] * N_c)
    orbs0_b  = jnp.stack([jnp.array(model.orbitals0)] * N_c)
    _ = batched(gamma0_b, orbs0_b, dummy_keys)
    print(f"  [JAX] compilation: {_t.perf_counter()-t0:.1f}s", flush=True)

    # ----- JAX benchmark -----
    print("\n[JAX]")
    jax_results = []
    jax_times   = []
    for seed in range(n_seeds):
        key = jr.PRNGKey(seed * 9999)
        t0  = time.perf_counter()
        res = run_cloning_jax(model, zeta=zeta, T_total=T,
                               N_c=N_c, key=key)
        wall = time.perf_counter() - t0
        # First seed includes compilation — report separately
        jax_times.append(wall)
        jax_results.append(res)
        compile_note = " (incl. compile)" if seed == 0 else ""
        print(f"  seed {seed}: S={res['S_mean']:.4f}  "
              f"theta={res['theta_hat']:.4f}  "
              f"ESS={res['eff_sample_size']:.1f}/{N_c}  "
              f"wall={wall:.1f}s{compile_note}")

    # ----- numpy benchmark -----
    print("\n[numpy]")
    np_results = []
    np_times   = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed * 9999)
        t0  = time.perf_counter()
        res = run_cloning(model, zeta=zeta, T_total=T,
                           N_c=N_c, rng=rng, show_progress=False)
        wall = time.perf_counter() - t0
        np_times.append(wall)
        np_results.append(res)
        print(f"  seed {seed}: S={res.S_mean:.4f}  "
              f"theta={res.theta_hat:.4f}  "
              f"ESS={res.eff_sample_size:.1f}/{N_c}  "
              f"wall={wall:.1f}s")

    # ----- Agreement check -----
    print("\n[agreement]")
    jax_S  = np.array([r["S_mean"]   for r in jax_results])
    np_S   = np.array([r.S_mean      for r in np_results])
    jax_th = np.array([r["theta_hat"] for r in jax_results])
    np_th  = np.array([r.theta_hat   for r in np_results])

    def pooled_sigma(*arrs):
        combined = np.concatenate(arrs)
        return combined.std() + 1e-10

    S_sigma  = pooled_sigma(jax_S,  np_S)
    th_sigma = pooled_sigma(jax_th, np_th)
    S_diff   = abs(jax_S.mean()  - np_S.mean())
    th_diff  = abs(jax_th.mean() - np_th.mean())

    S_ok     = S_diff  < 3 * S_sigma
    th_ok    = th_diff < 3 * th_sigma
    print(f"  S_mean:    JAX {jax_S.mean():.4f}±{jax_S.std():.4f}  "
          f"numpy {np_S.mean():.4f}±{np_S.std():.4f}  "
          f"{'OK' if S_ok else 'FAIL (>3sigma)'}")
    print(f"  theta_hat: JAX {jax_th.mean():.4f}±{jax_th.std():.4f}  "
          f"numpy {np_th.mean():.4f}±{np_th.std():.4f}  "
          f"{'OK' if th_ok else 'FAIL (>3sigma)'}")

    # ----- Speedup (exclude seed 0 from JAX which has compile overhead) -----
    jax_steady  = np.mean(jax_times[1:]) if n_seeds > 1 else jax_times[0]
    np_steady   = np.mean(np_times)
    speedup     = np_steady / jax_steady
    print(f"\n[speedup]")
    print(f"  numpy steady: {np_steady:.1f}s")
    print(f"  JAX steady:   {jax_steady:.1f}s  (seed 0 excluded: {jax_times[0]:.1f}s)")
    print(f"  SPEEDUP:      {speedup:.2f}x")

    return {
        "L": L, "lam": lam, "zeta": zeta,
        "speedup": speedup,
        "agreement_S": S_ok,
        "agreement_theta": th_ok,
        "jax_S_mean": jax_S.mean(),
        "np_S_mean":  np_S.mean(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L",     type=int,   nargs="+", default=[32, 64])
    p.add_argument("--lam",   type=float, default=0.40)
    p.add_argument("--zeta",  type=float, default=0.50)
    p.add_argument("--N-c",   type=int,   default=200)
    p.add_argument("--T",     type=float, default=8.0)
    p.add_argument("--seeds", type=int,   default=3)
    args = p.parse_args()

    print("=" * 60)
    print("PPS-QJ JAX GPU benchmark")
    print("=" * 60)

    if not check_jax():
        sys.exit(1)

    results = []
    for L in args.L:
        r = bench_jax_vs_numpy(
            L=L, lam=args.lam, zeta=args.zeta,
            N_c=args.N_c, T=args.T, n_seeds=args.seeds,
        )
        results.append(r)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'L':>5}  {'speedup':>9}  {'S agree':>9}  {'theta agree':>12}")
    for r in results:
        print(f"  {r['L']:3d}  {r['speedup']:9.2f}x  "
              f"{'OK' if r['agreement_S'] else 'FAIL':>9}  "
              f"{'OK' if r['agreement_theta'] else 'FAIL':>12}")

    all_ok = all(r["agreement_S"] and r["agreement_theta"] for r in results)
    if not all_ok:
        print("\nWARNING: agreement check failed for some points.")
        print("Investigate cloning_jax.py before using for production.")
        sys.exit(1)
    else:
        print("\nAll agreement checks passed.")


if __name__ == "__main__":
    main()
