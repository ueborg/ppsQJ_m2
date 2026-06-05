#!/usr/bin/env python
"""dtau headroom: can we coarsen the time step (fewer steps = faster) without
moving the observables?  steps = T/dtau, and dtau is the discretisation, not
the physical horizon -- so this is the one remaining per-realisation compute
lever now that T is fixed.

Tested at SMALL L and SMALL zeta on purpose: the Trotter error per step scales
like dt = mult/(2*alpha*(L-1)), which is LARGEST at small alpha (small lambda)
and small L.  So if mult=2 passes here it is safe at your production L/zeta.

For each (zeta) and dtau multiplier it runs R seeds, compares B_L/CMI/theta to
the mult=1 baseline (paired tolerance), and reports the wall speedup.

Usage:
    python analysis/bench_dtau.py
    python analysis/bench_dtau.py --L 32 --zetas 0.15,0.30 --mults 1,1.5,2,3 --R 8
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np


def _run(params):
    """One run at a given delta_tau (None = production default). Returns obs+wall."""
    import numpy as _np
    from pps_qj.cloning import run_cloning
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
    L = params["L"]; alpha = params["lam"]
    model = build_gaussian_chain_model(L=L, w=1.0 - alpha, alpha=alpha)
    rng = _np.random.default_rng(params["seed"])
    t0 = time.perf_counter()
    try:
        res = run_cloning(model, zeta=params["zeta"], T_total=params["T"],
                          N_c=params["N_c"], rng=rng, delta_tau=params["delta_tau"],
                          show_progress=False, record_entropy=False, backend="scalar")
    except Exception as exc:  # noqa: BLE001
        return {**params, "ok": False, "error": repr(exc)}
    wall = time.perf_counter() - t0
    comps = _batched_compute_B_L(res.final_covs, L)
    bl, cmi = comps["B_L"], comps["CMI"]
    fb, fc = _np.isfinite(bl), _np.isfinite(cmi)
    return {**params, "ok": True, "wall": wall, "dtau_used": float(res.delta_tau),
            "B_L": float(_np.mean(bl[fb])) if fb.any() else float("nan"),
            "CMI": float(_np.mean(cmi[fc])) if fc.any() else float("nan"),
            "theta_hat": float(res.theta_hat)}


def _mse(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), (float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1
                             else float("nan"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="dtau headroom check")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--zetas", type=str, default="0.15,0.30")
    ap.add_argument("--lams", type=str, default="",
                    help="optional comma lambda per zeta; default ~0.5*sqrt(zeta)")
    ap.add_argument("--mults", type=str, default="1,1.5,2,3")
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--N_c", type=int, default=200)
    ap.add_argument("--T", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--outdir", type=str, default="outputs/diagnostics")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    workers = max(1, min(args.workers, os.cpu_count() or 1))
    zetas = [float(z) for z in args.zetas.split(",")]
    mults = [float(m) for m in args.mults.split(",")]
    lams = [float(x) for x in args.lams.split(",")] if args.lams else \
        [round(0.5 * z ** 0.5, 3) for z in zetas]

    tasks = []
    for z, lam in zip(zetas, lams):
        base = 1.0 / (2.0 * lam * (args.L - 1))
        for m in mults:
            dt = None if m == 1.0 else m * base
            for r in range(args.R):
                tasks.append(dict(L=args.L, lam=lam, zeta=z, T=args.T, N_c=args.N_c,
                                  seed=400 + r, mult=m, delta_tau=dt))

    print("=" * 76)
    print(f"DTAU HEADROOM  L={args.L} N_c={args.N_c} T={args.T} R={args.R}  "
          f"mults={mults}")
    print(f"  zetas={zetas}  lams={lams}  ({len(tasks)} runs on {workers} workers)")
    print("=" * 76, flush=True)

    res = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            res.append(fut.result()); done += 1
            if done % 12 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {(time.time()-t0)/60:.1f} min", flush=True)

    summary = _analyse(zetas, lams, mults, res)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "bench_dtau.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nsummary -> {out/'bench_dtau.json'}")
    return 0


def _analyse(zetas, lams, mults, res):
    print("\n" + "=" * 76)
    print("VERDICT: largest safe dtau multiplier (= free speedup)")
    print("=" * 76)
    summary = {}
    for z, lam in zip(zetas, lams):
        rows = {}
        for m in mults:
            sub = [r for r in res if r.get("ok") and r["zeta"] == z and r["mult"] == m]
            rows[m] = {obs: _mse([r[obs] for r in sub]) for obs in ("B_L", "CMI", "theta_hat")}
            rows[m]["wall"] = float(np.mean([r["wall"] for r in sub])) if sub else float("nan")
        base = rows[1.0]
        safe_mult = 1.0
        print(f"\n[zeta={z}, lambda={lam}]")
        for m in mults:
            ok_all = True
            deltas = []
            for obs in ("B_L", "CMI", "theta_hat"):
                m0, e0 = base[obs]; mm, em = rows[m][obs]
                tol = 2 * np.sqrt((e0 or 0) ** 2 + (em or 0) ** 2)
                d = abs(mm - m0)
                within = np.isfinite(d) and d <= max(tol, 0.02 * abs(m0) if m0 else 0)
                ok_all = ok_all and within
                deltas.append(f"{obs}:{mm:.3f}({'ok' if within else 'OFF'})")
            spd = base["wall"] / rows[m]["wall"] if rows[m]["wall"] else float("nan")
            tag = "SAFE" if ok_all else "biased"
            if ok_all and m > safe_mult:
                safe_mult = m
            print(f"  mult={m:>3}: {'  '.join(deltas)}   speedup={spd:.2f}x   [{tag}]")
        summary[f"zeta_{z}"] = dict(lam=lam, safe_mult=safe_mult,
                                    rows={str(m): rows[m] for m in mults})
        print(f"  => largest SAFE mult at zeta={z}: {safe_mult}x")

    worst = min((summary[k]["safe_mult"] for k in summary), default=1.0)
    print("\n" + "-" * 76)
    print(f"Grid-safe multiplier (min across tested zeta): {worst}x")
    if worst > 1.0:
        print(f"=> set PPS_DTAU_MULT={worst} in production for a ~{worst:.1f}x free speedup")
        print("   (re-confirm at one production L before the full campaign)")
    else:
        print("=> no headroom: current dtau is already at its accuracy limit")
    print("=" * 76, flush=True)
    summary["grid_safe_mult"] = worst
    return summary


if __name__ == "__main__":
    sys.exit(main())
