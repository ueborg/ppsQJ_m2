"""Hot-path profiler for the cloning simulation.

Usage (on Habrok interactive node):
    cd ~/pps_qj && source ~/venvs/pps_qj/bin/activate
    python analysis/profile_cloning.py --L 64 --L 96 --L 128

Reports wall-time breakdown between the key kernels so we can confirm
(or refute) Opus-4.8's sandbox cost profile on actual Habrok hardware
before committing to any algorithmic rewrites.

What it measures
----------------
For each L, it runs a SHORT cloning run (N_c=40, T=2, N_REAL=1) to get
relative timings, then profiles the hot functions using cProfile.  The
absolute times are only indicative; the *ratios* are what matter for
deciding where to invest engineering effort.

The key question: is orbitals_from_covariance actually ~37% of wall at
L=128?  If yes, the orbital-space rank-2 QR rewrite is worth implementing.
If the brentq survival-search dominates instead, that's the lever.

Also compares scalar vs batched backend times directly.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np


def _profile_one(L: int, N_c: int, T: float, backend: str, seed: int = 42) -> dict:
    """Run a short cloning simulation and return timing breakdown."""
    from pps_qj.cloning import run_cloning
    from pps_qj.gaussian_backend import build_gaussian_chain_model

    alpha = 0.30  # moderate lambda near criticality
    w = 0.70
    zeta = 0.30

    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    rng = np.random.default_rng(seed)

    # --- cProfile pass ---
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    result = run_cloning(
        model, zeta=zeta, T_total=T, N_c=N_c, rng=rng,
        show_progress=False, record_entropy=True, backend=backend,
    )
    wall = time.perf_counter() - t0
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    profile_text = s.getvalue()

    return {
        "L": L, "N_c": N_c, "T": T, "backend": backend,
        "wall_s": wall,
        "n_steps": result.S_history.shape[0] if hasattr(result.S_history, "shape") else len(result.S_history),
        "profile": profile_text,
    }


def _time_backend_pair(L: int, N_c: int, T: float, reps: int = 3) -> dict:
    """Compare scalar vs batched backend wall-times directly."""
    from pps_qj.cloning import run_cloning
    from pps_qj.gaussian_backend import build_gaussian_chain_model

    alpha, w, zeta = 0.30, 0.70, 0.30
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)

    times = {"scalar": [], "batched": []}
    for rep in range(reps):
        for backend in ("scalar", "batched"):
            rng = np.random.default_rng(rep * 100)
            t0 = time.perf_counter()
            run_cloning(model, zeta=zeta, T_total=T, N_c=N_c, rng=rng,
                        show_progress=False, record_entropy=False, backend=backend)
            times[backend].append(time.perf_counter() - t0)

    return {
        "L": L, "N_c": N_c, "T": T,
        "scalar_mean": float(np.mean(times["scalar"])),
        "batched_mean": float(np.mean(times["batched"])),
        "speedup": float(np.mean(times["scalar"]) / np.mean(times["batched"])),
    }


def main():
    ap = argparse.ArgumentParser(description="Profile cloning hot path")
    ap.add_argument("--L", type=int, nargs="+", default=[32, 64, 96, 128])
    ap.add_argument("--N_c", type=int, default=40,
                    help="Clones per run (keep small; we want ratios, not statistics)")
    ap.add_argument("--T", type=float, default=3.0,
                    help="Time horizon for profiling run (keep short)")
    ap.add_argument("--reps", type=int, default=3,
                    help="Repetitions for scalar/batched comparison")
    ap.add_argument("--outdir", type=str, default="outputs/profiling")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Cloning hot-path profiler  N_c={args.N_c}  T={args.T}  L={args.L}")
    print("=" * 72)

    # --- Scalar/batched backend comparison ---
    print("\n=== Scalar vs Batched backend speedup ===")
    print(f"{'L':>6} {'N_c':>6} {'scalar(s)':>12} {'batched(s)':>12} {'speedup':>10}")
    comp_rows = []
    for L in args.L:
        row = _time_backend_pair(L, args.N_c, args.T, args.reps)
        comp_rows.append(row)
        print(f"{L:>6} {args.N_c:>6} {row['scalar_mean']:>12.3f} "
              f"{row['batched_mean']:>12.3f} {row['speedup']:>10.2f}x")

    # --- cProfile for scalar at each L (the production backend until now) ---
    print("\n=== cProfile breakdown (scalar backend, top functions) ===")
    for L in args.L:
        print(f"\n--- L={L} (scalar) ---")
        res = _profile_one(L, args.N_c, args.T, backend="scalar")
        print(f"Wall: {res['wall_s']:.2f}s  n_steps≈{res['n_steps']}")
        # Filter profile output to the interesting lines
        lines = res["profile"].split("\n")
        keywords = ["gaussian_born", "orbitals_from", "brentq", "apply_proj",
                    "eigvalsh", "eigh", "qr", "matmul", "batched_entropy",
                    "cloning.py", "gaussian_backend"]
        print("  (lines matching hot functions:)")
        for line in lines:
            if any(k in line for k in keywords) and line.strip():
                print("  " + line)
        # Save full profile
        ppath = outdir / f"profile_L{L}_scalar.txt"
        ppath.write_text(res["profile"])
        print(f"  Full profile -> {ppath}")

    # --- Extrapolate to L=128 production costs ---
    print("\n=== Extrapolated cost at production scale ===")
    print("(L=128, N_c=250, T=100, 5 workers — only valid if L^4 scaling holds)")
    T_prod, N_c_prod, n_workers = 100.0, 250, 5
    for row in comp_rows:
        if row["L"] <= max(args.L) - 1:
            scale = (128 / row["L"]) ** 4 * (N_c_prod / args.N_c) * (T_prod / args.T)
            t_scalar_proj = row["scalar_mean"] * scale / n_workers
            t_batched_proj = row["batched_mean"] * scale / n_workers
            print(f"  from L={row['L']:3d}: scalar ~{t_scalar_proj:.1f}h  "
                  f"batched ~{t_batched_proj:.1f}h  (extrapolated, treat as order-of-magnitude)")

    print(f"\nProfile files saved to {outdir}/")


if __name__ == "__main__":
    main()
