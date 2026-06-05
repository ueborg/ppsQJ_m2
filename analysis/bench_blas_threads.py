#!/usr/bin/env python
"""Does threaded BLAS lower the single-realisation wall floor?

Per-realisation cost can't be split across processes, but a single run's
linear algebra (eigh/matmul/qr on 2L x 2L matrices) CAN use multiple BLAS
threads. That trades concurrency for a lower wall floor -- the thing that
blocks "1 day" at L=128. This times one realisation at several thread counts.

BLAS threads must be fixed before numpy imports, so this self-spawns one
subprocess per (L, threads) with the env set, and parses the timing back.

Usage:
    python analysis/bench_blas_threads.py
    python analysis/bench_blas_threads.py --Ls 64,128 --threads 1,2,4,8 --N_c 60 --T 3
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)


def _timer_mode():
    # env (thread counts) already set by the parent before this process started.
    sys.path.insert(0, REPO)
    import numpy as np
    from pps_qj.cloning import run_cloning
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--N_c", type=int, required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--lam", type=float, default=0.31)
    ap.add_argument("--zeta", type=float, default=0.30)
    a, _ = ap.parse_known_args()
    model = build_gaussian_chain_model(L=a.L, w=1.0 - a.lam, alpha=a.lam)
    rng = np.random.default_rng(7)
    t0 = time.perf_counter()
    run_cloning(model, zeta=a.zeta, T_total=a.T, N_c=a.N_c, rng=rng,
                show_progress=False, record_entropy=False, backend="scalar")
    print("WALL_JSON" + json.dumps({"wall": time.perf_counter() - t0,
                                    "threads": os.environ.get("OMP_NUM_THREADS")}))


def _time_one(L, threads, N_c, T):
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = str(threads)
    cmd = [sys.executable, os.path.abspath(__file__), "--_timer",
           "--L", str(L), "--N_c", str(N_c), "--T", str(T)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("WALL_JSON"):
            return json.loads(line[len("WALL_JSON"):])["wall"]
    sys.stderr.write(out.stderr[-500:])
    return float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description="BLAS-thread floor benchmark")
    ap.add_argument("--Ls", type=str, default="64,128")
    ap.add_argument("--threads", type=str, default="1,2,4,8")
    ap.add_argument("--N_c", type=int, default=60)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--outdir", type=str, default="outputs/diagnostics")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    Ls = [int(x) for x in args.Ls.split(",")]
    threads = [int(x) for x in args.threads.split(",")]

    print("=" * 68)
    print(f"BLAS-THREAD FLOOR  N_c={args.N_c} T={args.T}  (timing ratio is "
          f"N_c/T-independent)")
    print("=" * 68)
    summary = {}
    for L in Ls:
        print(f"\n[L={L}]")
        base = None; rows = {}
        for n in threads:
            w = _time_one(L, n, args.N_c, args.T)
            base = base or w
            spd = base / w if w and w == w else float("nan")
            # production floor estimate: scale this T to T=100, N_c=250
            floor_h = w * (100.0 / args.T) * (250.0 / args.N_c) / 3600.0
            rows[n] = dict(wall_s=w, speedup=spd, est_floor_h_at_prod=floor_h)
            print(f"  threads={n}: {w:6.1f}s  speedup={spd:.2f}x  "
                  f"=> est L={L} floor at N_c=250,T=100: {floor_h:.1f} h")
        summary[f"L{L}"] = rows
        best = max(threads, key=lambda n: rows[n]["speedup"])
        print(f"  best: {best} threads ({rows[best]['speedup']:.2f}x); "
              f"floor {rows[1]['est_floor_h_at_prod']:.1f}h -> "
              f"{rows[best]['est_floor_h_at_prod']:.1f}h")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "bench_blas_threads.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n" + "-" * 68)
    print("Note: more threads/realisation = fewer concurrent realisations.")
    print("Use only if WALL (not core-hours) is the binding constraint, and")
    print("only up to the thread count where speedup still beats linear loss.")
    print(f"summary -> {out/'bench_blas_threads.json'}")
    return 0


if __name__ == "__main__":
    if "--_timer" in sys.argv:
        _timer_mode()
    else:
        sys.exit(main())
