#!/usr/bin/env python3
"""Ruche worker: run chi2_mc for one (L,u,c,seed), write poolable JSON components.
  python analysis/chi2_worker.py --L 64 --u 0.75 --c 1.0 --seed 3 --N1 5000 --N2 200000 --outdir DIR
Threads pinned to 1 (before numpy). One realization per core; pool seeds in aggregate."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.chi2_response import chi2_mc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True); ap.add_argument("--u", type=float, default=0.75)
    ap.add_argument("--c", type=float, default=1.0); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--N1", type=int, default=5000); ap.add_argument("--N2", type=int, default=200000)
    ap.add_argument("--outdir", required=True); a = ap.parse_args()
    assert a.L % 4 == 0
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"chi2_L{a.L}_u{a.u}_c{a.c}_s{a.seed}.json")
    if os.path.exists(out):
        print("exists, skip", out); sys.exit(0)
    t0 = time.time(); r = chi2_mc(a.L, a.u, a.c, a.N1, a.N2, seed=a.seed)
    r.update(L=a.L, u=a.u, c=a.c, seed=a.seed, N1=a.N1, N2=a.N2, wall_s=time.time() - t0)
    tmp = out + ".tmp"; json.dump({k: (float(v) if hasattr(v, "__float__") else v) for k, v in r.items()}, open(tmp, "w"))
    os.replace(tmp, out); print(f"wrote {out}  chi2={r['chi2']:.4f}  {r['wall_s']:.0f}s")
