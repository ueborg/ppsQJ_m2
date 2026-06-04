#!/usr/bin/env python
"""D5+ : clones-vs-seeds variance decomposition for the cloning estimator.

The Phase-1 suite found rel-err on B_L scaling far flatter than 1/sqrt(N_c)
(fitted exponent ~ -0.14), the signature of clone genealogical degeneracy.
But n=5 realisations per N_c was too thin to trust.  This script settles it.

Model.  Across independent realisations (seeds) at fixed N_c, the single-run
estimate behaves as
        B_L^(r) = B_L_true + seed_r + clone_noise_r ,
with Var(seed_r) = V_seed (independent of N_c, irreducible) and
Var(clone_noise_r) = V_clone(N_c) ~ A / N_c (reducible by more clones).
Hence the between-realisation variance
        V_between(N_c) = V_seed + A / N_c .
Fitting V_between vs 1/N_c gives the split directly:
    intercept  -> V_seed  (the floor; only MORE SEEDS reduce it)
    slope/N_c  -> V_clone (the part MORE CLONES / MPI would reduce).

If V_seed dominates at production N_c, MPI to raise N_c is wasted effort and
the precision lever is more independent realisations (embarrassingly parallel,
no MPI).  If A/N_c dominates, MPI is justified.

Usage (Habrok interactive node, ~30-45 min):
    cd ~/pps_qj && source ~/venvs/pps_qj/bin/activate
    python analysis/d5_clones_vs_seeds.py
    python analysis/d5_clones_vs_seeds.py --R 24 --T 10 --Ncs 25,50,100,200,400
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from analysis.diagnostic_suite import _one_run, LAM_DEFAULT, ZETA_DEFAULT


def _parse(argv):
    ap = argparse.ArgumentParser(description="clones-vs-seeds variance decomposition")
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--T", type=float, default=10.0,
                    help="trajectory time (shorter than production=100 to fit budget; "
                         "the N_c-dependence of the variance is T-robust)")
    ap.add_argument("--R", type=int, default=20, help="realisations (seeds) per N_c")
    ap.add_argument("--Ncs", type=str, default="25,50,100,200,400")
    ap.add_argument("--lam", type=float, default=LAM_DEFAULT)
    ap.add_argument("--zeta", type=float, default=ZETA_DEFAULT)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--prod-Nc", type=int, default=250,
                    help="N_c at which to report the irreducible fraction")
    ap.add_argument("--target-relerr", type=float, default=0.05)
    ap.add_argument("--outdir", type=str, default="outputs/diagnostics")
    return ap.parse_args(argv if argv is not None else sys.argv[1:])


def _build_tasks(args, Ncs):
    # Independent seed per (N_c, rep); offset keeps them disjoint from the suite.
    tasks, gid = [], 0
    for N_c in Ncs:
        for rep in range(args.R):
            tasks.append(dict(diag="D5x", N_c=N_c, rep=rep,
                              params=dict(L=args.L, lam=args.lam, zeta=args.zeta,
                                          T=args.T, N_c=N_c, seed=900_000 + gid,
                                          backend="scalar", record_entropy=False,
                                          compute_bl=True)))
            gid += 1
    return tasks


def _fit_decomp(Ncs, V):
    """Fit V_between = B + A/x via OLS on x=1/N_c. Returns (B, A, free_exponent)."""
    Ncs = np.asarray(Ncs, float); V = np.asarray(V, float)
    ok = np.isfinite(V) & (V > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    x = 1.0 / Ncs[ok]
    A, B = np.polyfit(x, V[ok], 1)          # V ~ A*x + B
    # free-exponent comparison (like the Phase-1 -0.14 on rel-err = sqrt(V))
    q = float(np.polyfit(np.log(Ncs[ok]), np.log(np.sqrt(V[ok])), 1)[0]) \
        if ok.sum() >= 3 else float("nan")
    return float(B), float(A), q


def main(argv=None):
    args = _parse(argv)
    Ncs = [int(x) for x in args.Ncs.split(",")]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, os.cpu_count() or 1))

    print("=" * 72)
    print(f"D5+ clones-vs-seeds  L={args.L} lam={args.lam} zeta={args.zeta} "
          f"T={args.T}  R={args.R}/N_c  N_cs={Ncs}")
    print("=" * 72, flush=True)

    # crude upfront estimate from one cheap run (smallest N_c)
    t0 = time.perf_counter()
    _ = _one_run(dict(L=args.L, lam=args.lam, zeta=args.zeta, T=args.T,
                      N_c=min(Ncs), seed=1, backend="scalar",
                      record_entropy=False, compute_bl=True))
    c = (time.perf_counter() - t0) / min(Ncs)           # ~sec per clone-unit
    work = sum(c * nc * args.R for nc in Ncs)
    print(f"  est total work {work/60:.0f} core-min  ->  ~{work/workers/60:.0f} "
          f"min wall on {workers} workers (floor {c*max(Ncs)/60:.0f} min)\n", flush=True)

    tasks = _build_tasks(args, Ncs)
    res = {nc: {"B_L": [], "CMI": []} for nc in Ncs}
    n_done, n_total = 0, len(tasks)
    raw = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_run, t["params"]): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            r = fut.result()
            n_done += 1
            if r.get("ok"):
                res[t["N_c"]]["B_L"].append(r.get("B_L_mean"))
                res[t["N_c"]]["CMI"].append(r.get("CMI_mean"))
            raw.append({"N_c": t["N_c"], "rep": t["rep"], "ok": r.get("ok"),
                        "B_L_mean": r.get("B_L_mean"), "CMI_mean": r.get("CMI_mean"),
                        "wall": r.get("wall")})
            if n_done % 10 == 0 or n_done == n_total:
                print(f"  [{n_done:>3}/{n_total}] last: N_c={t['N_c']} "
                      f"wall={r.get('wall', float('nan')):.1f}s", flush=True)
                (outdir / "d5x_raw.json").write_text(json.dumps(raw, indent=1, default=float))
    (outdir / "d5x_raw.json").write_text(json.dumps(raw, indent=1, default=float))

    summary = _analyse(args, Ncs, res)
    (outdir / "d5x_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _print_verdict(args, summary)
    print(f"\nraw -> {outdir/'d5x_raw.json'}   summary -> {outdir/'d5x_summary.json'}")
    return 0


def _one_obs(Ncs, arrs, prod_Nc, target, R, n_boot=500, rng=None):
    """Decompose one observable. arrs: dict N_c -> list of per-realisation values."""
    rng = rng or np.random.default_rng(0)
    Ncs_s = sorted(Ncs)
    rows, V, means = [], [], []
    for nc in Ncs_s:
        a = np.asarray([v for v in arrs[nc] if v is not None and np.isfinite(v)], float)
        m = float(a.mean()) if a.size else float("nan")
        v = float(a.var(ddof=1)) if a.size > 1 else float("nan")
        rows.append(dict(N_c=nc, n=int(a.size), mean=m, var=v,
                         relErr=(math.sqrt(v / a.size) / abs(m))
                                 if a.size > 1 and m else float("nan")))
        V.append(v); means.append(m)
    B, A, q = _fit_decomp(Ncs_s, V)
    mean_ref = means[-1] if np.isfinite(means[-1]) else float(np.nanmean(means))
    # A<=0 means V_between does not fall with N_c: no detectable clone-reducible
    # component, i.e. fully seed-limited. Estimate the floor as the mean variance.
    a_negative = bool(np.isfinite(A) and A <= 0)
    if a_negative:
        B = float(np.nanmean([v for v in V if np.isfinite(v) and v > 0]))
        A = 0.0
    V_prod = B + A / prod_Nc
    frac = B / V_prod if V_prod and np.isfinite(V_prod) and V_prod > 0 else float("nan")
    frac = float(min(1.0, max(0.0, frac))) if np.isfinite(frac) else frac

    # bootstrap over realisations -> CI on B and on irreducible fraction
    Bs, fr = [], []
    for _ in range(n_boot):
        Vb = []
        for nc in Ncs_s:
            a = np.asarray([v for v in arrs[nc] if v is not None and np.isfinite(v)], float)
            if a.size > 1:
                bb = rng.choice(a, size=a.size, replace=True)
                Vb.append(float(bb.var(ddof=1)))
            else:
                Vb.append(float("nan"))
        b, aa, _ = _fit_decomp(Ncs_s, Vb)
        vp = b + aa / prod_Nc
        Bs.append(b)
        if vp and np.isfinite(vp) and vp > 0:
            fr.append(max(0.0, min(1.0, b / vp)))
    ci = lambda z: [float(np.nanpercentile(z, 2.5)), float(np.nanpercentile(z, 97.5))] if z else [float("nan")] * 2

    # how many seeds to hit target rel-err at prod_Nc; and clone-only floor at current R
    R_needed = (V_prod / (target * mean_ref) ** 2) if mean_ref else float("nan")
    floor_relerr_curR = (math.sqrt(max(B, 0) / R) / abs(mean_ref)) if mean_ref else float("nan")
    return dict(rows=rows, B=B, A=A, free_exponent=q, mean_ref=mean_ref,
                a_negative=a_negative,
                V_prod=V_prod, irreducible_fraction=frac,
                B_CI=ci(Bs), frac_CI=ci(fr),
                seeds_needed_at_prodNc=R_needed,
                clone_only_floor_relerr_at_R=floor_relerr_curR)


def _analyse(args, Ncs, res):
    return {"meta": dict(L=args.L, T=args.T, R=args.R, Ncs=Ncs, lam=args.lam,
                         zeta=args.zeta, prod_Nc=args.prod_Nc,
                         target_relerr=args.target_relerr),
            "B_L": _one_obs(Ncs, {nc: res[nc]["B_L"] for nc in Ncs},
                            args.prod_Nc, args.target_relerr, args.R),
            "CMI": _one_obs(Ncs, {nc: res[nc]["CMI"] for nc in Ncs},
                            args.prod_Nc, args.target_relerr, args.R)}


def _print_verdict(args, s):
    print("\n" + "=" * 72)
    print("VERDICT: clones (MPI) vs seeds (more tasks)")
    print("=" * 72)
    for name in ("B_L", "CMI"):
        o = s[name]
        print(f"\n[{name}]  per-N_c (mean, rel-err):")
        for r in o["rows"]:
            print(f"    N_c={r['N_c']:>4}  n={r['n']:>2}  mean={r['mean']:.4g}  "
                  f"relErr={r['relErr']:.1%}" if np.isfinite(r['relErr'])
                  else f"    N_c={r['N_c']:>4}  n={r['n']:>2}  mean={r['mean']:.4g}  relErr=n/a")
        fr, lo_hi = o["irreducible_fraction"], o["frac_CI"]
        print(f"  fit V_between = B + A/N_c :  B(seed floor)={o['B']:.3e}  A={o['A']:.3e}"
              + ("  [A<=0: no clone-reducible component detected]" if o.get("a_negative") else ""))
        print(f"  free exponent on rel-err = {o['free_exponent']:.3f}  "
              f"(independent sampling would be -0.50)")
        print(f"  irreducible (seed) fraction at N_c={args.prod_Nc}: "
              f"{fr:.0%}  (95% CI {lo_hi[0]:.0%}-{lo_hi[1]:.0%})")
        floor = o["clone_only_floor_relerr_at_R"]
        print(f"  clone-only floor at R={args.R} (N_c->inf): rel-err -> {floor:.1%}"
              + ("  -> clones ALONE cannot reach target" if np.isfinite(floor)
                 and floor > args.target_relerr else ""))
        Rn = o["seeds_needed_at_prodNc"]
        print(f"  seeds needed at N_c={args.prod_Nc} for {args.target_relerr:.0%} "
              f"rel-err: R ~= {Rn:.0f}" if np.isfinite(Rn) else "  seeds needed: n/a")

    o = s["B_L"]; fr = o["irreducible_fraction"]
    print("\n" + "-" * 72)
    if np.isfinite(fr) and fr > 0.6:
        print("=> SEED-LIMITED. Most B_L variance is irreducible seed-to-seed")
        print("   fluctuation. MPI to raise N_c would cut only the reducible")
        print("   remainder. Precision lever = MORE REALISATIONS (independent")
        print("   tasks, trivially parallel). Do NOT build MPI for this.")
    elif np.isfinite(fr) and fr < 0.3:
        print("=> CLONE-LIMITED. Variance falls ~1/N_c, so more clones help and")
        print("   MPI population-split is justified to reach the precision target.")
    else:
        print("=> MIXED / inconclusive. Both seeds and clones contribute; rerun")
        print("   with larger --R and a wider --Ncs range before committing to MPI.")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    sys.exit(main())
