#!/usr/bin/env python
"""Is the production time horizon T overkill?  Measure the saturation time.

Cost scales linearly with the number of steps (steps ~ T), so if an observable
plateaus well before the grid's T cap, shortening T is a free speedup.  The
grid sets T = max(30, 5/alpha, 2L) capped at 100/150/200; that was verified
*sufficient* but never checked for being *more than necessary*.

Two independent saturation probes per (L, zeta, lambda) point:
  1. S(t): the per-step entanglement entropy, recorded for free along ONE
     trajectory run to the full T.  Plateau time = state has equilibrated.
  2. B_L(t), CMI(t): a same-seed T-ladder (T*{0.3,0.5,0.7,1.0}).  Because the
     RNG stream up to time t is identical regardless of T_total, the end-of-run
     B_L at the shorter T equals B_L at that time in the longer run -- a true
     saturation curve on the actual crossing observable.

Saturation is a physics timescale (N_c-independent), so this runs at small N_c
and small/mid L and extrapolates the safe T to large L via the 2L rule.

Usage:
    python analysis/t_saturation.py
    python analysis/t_saturation.py --points 32:0.15:0.40,64:0.15:0.40 --R 6
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


def _T_prod(L, alpha):
    """Production time horizon from the grid (time_horizon_v2)."""
    base = max(30.0, 5.0 / max(alpha, 1e-9))
    T = float(max(base, 2.0 * L))
    if L >= 96:
        return min(T, 100.0)
    if L >= 64:
        return min(T, 150.0)
    if L >= 32:
        return min(T, 200.0)
    return T


def _s_curve(params):
    """One trajectory to full T; return downsampled S(t)."""
    import numpy as _np
    from pps_qj.cloning import run_cloning
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    L = params["L"]; alpha = params["lam"]
    model = build_gaussian_chain_model(L=L, w=1.0 - alpha, alpha=alpha)
    rng = _np.random.default_rng(params["seed"])
    try:
        res = run_cloning(model, zeta=params["zeta"], T_total=params["T"],
                          N_c=params["N_c"], rng=rng, show_progress=False,
                          record_entropy=True, entropy_stride=1, backend="scalar")
    except Exception as exc:  # noqa: BLE001
        return {"kind": "scurve", "key": params["key"], "ok": False, "error": repr(exc)}
    S = _np.asarray(res.S_history, float)
    dt = float(res.delta_tau)
    t = _np.arange(S.size) * dt
    # downsample to <=200 points for JSON
    idx = _np.linspace(0, S.size - 1, min(200, S.size)).astype(int)
    return {"kind": "scurve", "key": params["key"], "ok": True,
            "t": t[idx].tolist(), "S": S[idx].tolist(), "T": params["T"], "dt": dt}


def _ladder(params):
    """End-of-run B_L/CMI at T=frac*T_prod (same seed -> point on B_L(t))."""
    import numpy as _np
    from pps_qj.cloning import run_cloning
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
    L = params["L"]; alpha = params["lam"]
    model = build_gaussian_chain_model(L=L, w=1.0 - alpha, alpha=alpha)
    rng = _np.random.default_rng(params["seed"])
    try:
        res = run_cloning(model, zeta=params["zeta"], T_total=params["T"],
                          N_c=params["N_c"], rng=rng, show_progress=False,
                          record_entropy=False, backend="scalar")
        comps = _batched_compute_B_L(res.final_covs, L)
        bl, cmi = comps["B_L"], comps["CMI"]
        fb, fc = _np.isfinite(bl), _np.isfinite(cmi)
        return {"kind": "ladder", "key": params["key"], "frac": params["frac"],
                "ok": True,
                "B_L": float(_np.mean(bl[fb])) if fb.any() else float("nan"),
                "CMI": float(_np.mean(cmi[fc])) if fc.any() else float("nan")}
    except Exception as exc:  # noqa: BLE001
        return {"kind": "ladder", "key": params["key"], "frac": params["frac"],
                "ok": False, "error": repr(exc)}


def _find_tsat(t, S, tol=0.02):
    """Earliest time S(t) enters and stays within tol (relative) of the plateau."""
    t = np.asarray(t); S = np.asarray(S)
    n = S.size
    tail = S[int(0.8 * n):]
    plateau = float(tail.mean())
    band = max(abs(tol * plateau), float(tail.std()))
    if plateau == 0:
        return float("nan"), plateau
    for i in range(n):
        if np.all(np.abs(S[i:] - plateau) <= band):
            return float(t[i]), plateau
    return float(t[-1]), plateau


def main(argv=None):
    ap = argparse.ArgumentParser(description="T saturation probe")
    ap.add_argument("--points", type=str,
                    default="32:0.15:0.40,64:0.15:0.40,32:0.50:0.55,64:0.50:0.55",
                    help="comma list of L:zeta:lambda (near-critical lambda)")
    ap.add_argument("--fracs", type=str, default="0.3,0.5,0.7,1.0")
    ap.add_argument("--R", type=int, default=6, help="seeds per ladder rung")
    ap.add_argument("--N_c", type=int, default=150)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--outdir", type=str, default="outputs/diagnostics")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, os.cpu_count() or 1))
    fracs = [float(x) for x in args.fracs.split(",")]
    pts = []
    for p in args.points.split(","):
        L, z, lam = p.split(":")
        pts.append(dict(L=int(L), zeta=float(z), lam=float(lam),
                        key=f"L{L}_z{z}_l{lam}", T=_T_prod(int(L), float(lam))))

    tasks = []
    for p in pts:
        tasks.append((_s_curve, dict(L=p["L"], zeta=p["zeta"], lam=p["lam"],
                                     N_c=args.N_c, T=p["T"], seed=1, key=p["key"])))
        for frac in fracs:
            for r in range(args.R):
                tasks.append((_ladder, dict(L=p["L"], zeta=p["zeta"], lam=p["lam"],
                                            N_c=args.N_c, T=frac * p["T"],
                                            seed=100 + r, frac=frac, key=p["key"])))

    print("=" * 72)
    print(f"T-SATURATION PROBE  N_c={args.N_c}  R={args.R}  fracs={fracs}")
    for p in pts:
        print(f"  {p['key']}: T_prod={p['T']:.0f}")
    print(f"  {len(tasks)} runs on {workers} workers", flush=True)

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, pr): pr for fn, pr in tasks}
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result()); done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] elapsed {(time.time()-t0)/60:.1f} min",
                      flush=True)
    (outdir / "t_saturation_raw.json").write_text(json.dumps(results, indent=1, default=float))
    _summarize(pts, fracs, results, args.outdir)
    return 0


def _mse(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), (float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1
                             else float("nan"))


def _summarize(pts, fracs, results, outdir):
    fracs = sorted(fracs)
    scurves = {r["key"]: r for r in results if r.get("kind") == "scurve" and r.get("ok")}
    lad = {}
    for r in results:
        if r.get("kind") == "ladder" and r.get("ok"):
            lad.setdefault(r["key"], {}).setdefault(r["frac"], {"B_L": [], "CMI": []})
            lad[r["key"]][r["frac"]]["B_L"].append(r["B_L"])
            lad[r["key"]][r["frac"]]["CMI"].append(r["CMI"])

    print("\n" + "=" * 72)
    print("VERDICT: is T overkill?")
    print("=" * 72)
    summary = {}
    for p in pts:
        k = p["key"]; T = p["T"]
        rec_fracs = []
        line = [f"\n[{k}]  T_prod={T:.0f}"]
        if k in scurves:
            tsat, plateau = _find_tsat(scurves[k]["t"], scurves[k]["S"])
            line.append(f"  S(t) plateau at t={tsat:.0f}  ({tsat/T:.0%} of T)")
            rec_fracs.append(tsat / T)
        for obs in ("B_L", "CMI"):
            if k not in lad or 1.0 not in lad[k]:
                continue
            ref_m, ref_e = _mse(lad[k][1.0][obs])
            rec = 1.0
            for f in fracs:
                if f not in lad[k]:
                    continue
                m, e = _mse(lad[k][f][obs])
                tol = 2 * np.sqrt((ref_e or 0) ** 2 + (e or 0) ** 2)
                if np.isfinite(m) and abs(m - ref_m) <= max(tol, 0.02 * abs(ref_m)):
                    rec = f; break
            rec_fracs.append(rec)
            seq = "  ".join(f"{f:.1f}:{_mse(lad[k][f][obs])[0]:.3f}"
                            for f in fracs if f in lad[k])
            line.append(f"  {obs}(T): {seq}   -> stable from frac {rec:.1f}")
        rec_T = max(rec_fracs) * T if rec_fracs else T
        saved = (1 - rec_T / T) * 100
        line.append(f"  => recommended T ~= {rec_T:.0f}  (saves {saved:.0f}% of steps)")
        print("\n".join(line))
        summary[k] = dict(T_prod=T, recommended_T=rec_T, pct_saved=saved)

    if summary:
        avg_saved = float(np.mean([v["pct_saved"] for v in summary.values()]))
        print("\n" + "-" * 72)
        print(f"Average step saving from trimming T: ~{avg_saved:.0f}%  "
              f"(=> ~{1/(1-avg_saved/100):.2f}x cheaper if it holds across the grid)")
        print("Caveat: measured at small/mid L. Confirm the 2L relaxation rule by")
        print("re-running with one L=128 point before trimming the large-L cap.")
        print("=" * 72, flush=True)
    Path(outdir, "t_saturation_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    sys.exit(main())
