#!/usr/bin/env python
"""B_L(t)/CMI(t) saturation from single trajectories (snapshot probe).

Instead of a multi-run horizon ladder (one run per T), this records B_L and CMI
at many intermediate times WITHIN one trajectory via run_cloning's snapshot hook
-- so R runs give the whole saturation curve at ~1/n_snap the cost of laddering.
Directly answers: does the production observable flatten before T_prod, and by
how much could T be cut?

Saturation here is N_c-robust (a relaxation timescale), so it runs at modest
N_c; the absolute B_L carries finite-N_c bias but the *time it flattens* does not.

Usage (Habrok; ~2-5 h, use tmux or a small job):
    python analysis/bl_saturation.py                 # L=64,128 zeta=0.15
    python analysis/bl_saturation.py --Ls 128 --R 20 --N_c 50
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _T_prod(L, alpha):
    T = float(max(max(30.0, 5.0 / max(alpha, 1e-9)), 2.0 * L))
    return min(T, 100.0) if L >= 96 else min(T, 150.0) if L >= 64 else \
        min(T, 200.0) if L >= 32 else T


def _run(params):
    """One trajectory; snapshot B_L/CMI at intermediate times. snap_fn is built
    here (in the worker process) so nothing unpicklable crosses the pool."""
    import numpy as _np
    from pps_qj.cloning import run_cloning
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
    L = params["L"]

    def snap(covs):
        comps = _batched_compute_B_L(covs, L)
        bl, cmi = comps["B_L"], comps["CMI"]
        fb, fc = _np.isfinite(bl), _np.isfinite(cmi)
        return {"B_L": float(_np.mean(bl[fb])) if fb.any() else float("nan"),
                "CMI": float(_np.mean(cmi[fc])) if fc.any() else float("nan")}

    model = build_gaussian_chain_model(L=L, w=1.0 - params["lam"], alpha=params["lam"])
    rng = _np.random.default_rng(params["seed"])
    try:
        res = run_cloning(model, zeta=params["zeta"], T_total=params["T"],
                          N_c=params["N_c"], rng=rng, record_entropy=False,
                          backend="scalar", snapshot_times=params["snap_times"],
                          snapshot_fn=snap)
    except Exception as exc:  # noqa: BLE001
        return {"L": L, "seed": params["seed"], "ok": False, "error": repr(exc)}
    return {"L": L, "seed": params["seed"], "ok": True,
            "snapshots": [(float(t), v) for t, v in res.snapshots]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="B_L(t)/CMI(t) saturation probe")
    ap.add_argument("--Ls", type=str, default="64,128")
    ap.add_argument("--zeta", type=float, default=0.15)
    ap.add_argument("--lam", type=float, default=0.40)
    ap.add_argument("--N_c", type=int, default=50)
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--n-snap", type=int, default=12)
    ap.add_argument("--tol", type=float, default=0.05, help="rel tol vs final value")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--outdir", type=str, default="outputs/diagnostics")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    workers = max(1, min(args.workers, os.cpu_count() or 1))
    Ls = [int(x) for x in args.Ls.split(",")]

    tasks = []
    for L in Ls:
        T = _T_prod(L, args.lam)
        snap_times = list(np.linspace(T / args.n_snap, T, args.n_snap))
        for r in range(args.R):
            tasks.append(dict(L=L, lam=args.lam, zeta=args.zeta, N_c=args.N_c,
                              T=T, seed=600 + r, snap_times=snap_times))

    print("=" * 72)
    print(f"B_L(t) SATURATION  Ls={Ls} zeta={args.zeta} lam={args.lam} "
          f"N_c={args.N_c} R={args.R}")
    for L in Ls:
        print(f"  L={L}: T_prod={_T_prod(L, args.lam):.0f}, "
              f"{args.n_snap} snapshots")
    print(f"  {len(tasks)} runs on {workers} workers", flush=True)

    by_L_snap = defaultdict(lambda: defaultdict(lambda: {"B_L": [], "CMI": []}))
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result(); done += 1
            if r.get("ok"):
                for t, v in r["snapshots"]:
                    cell = by_L_snap[r["L"]][round(t, 3)]
                    cell["B_L"].append(v["B_L"]); cell["CMI"].append(v["CMI"])
            if done % 5 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {(time.time()-t0)/60:.1f} min", flush=True)

    summary = _analyse_plot(Ls, by_L_snap, args)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "bl_saturation.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nsummary -> {out/'bl_saturation.json'}")
    print(f"plot    -> {out/'bl_saturation.png'}")
    return 0


def _mse(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), (float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1
                             else float("nan"))


def _t_sat(times, means, sems, ref, ref_sem, tol):
    """Earliest time the curve enters and stays within tol(or 2-sem) of final."""
    for i, t in enumerate(times):
        band = max(tol * abs(ref), 2 * np.sqrt((ref_sem or 0) ** 2 + (sems[i] or 0) ** 2))
        if all(np.isfinite(means[j]) and abs(means[j] - ref) <= band
               for j in range(i, len(times))):
            return float(t)
    return float(times[-1])


def _analyse_plot(Ls, by_L_snap, args):
    fig, axes = plt.subplots(len(Ls), 1, figsize=(7, 3.4 * len(Ls)), squeeze=False)
    print("\n" + "=" * 72)
    print("VERDICT: B_L(t)/CMI(t) saturation time vs production horizon")
    print("=" * 72)
    summary = {}
    for ax_i, L in enumerate(Ls):
        snaps = by_L_snap[L]
        times = sorted(snaps)
        T = times[-1] if times else _T_prod(L, args.lam)
        rec = {"T_prod": T}
        ax = axes[ax_i][0]
        print(f"\n[L={L}]  T_prod={T:.0f}")
        for obs, col in (("B_L", "tab:blue"), ("CMI", "tab:orange")):
            ms = [_mse(snaps[t][obs]) for t in times]
            means = [m for m, _ in ms]; sems = [s for _, s in ms]
            ref, ref_sem = means[-1], sems[-1]
            tsat = _t_sat(times, means, sems, ref, ref_sem, args.tol)
            saved = (1 - tsat / T) * 100
            ax.errorbar(times, means, yerr=sems, fmt="o-", color=col, capsize=3,
                        ms=4, label=obs)
            ax.axvline(tsat, color=col, ls=":", lw=1)
            print(f"  {obs}: saturates by t~{tsat:.0f}  ({tsat/T:.0%} of T)  "
                  f"=> T could be cut ~{saved:.0f}%   [final={ref:.4g}]")
            rec[obs] = dict(t_sat=tsat, frac=tsat / T, pct_saveable=saved,
                            final=ref, curve=[(float(t), float(m), float(s))
                                              for t, m, s in zip(times, means, sems)])
        ax.set_title(f"L={L}  zeta={args.zeta}  lam={args.lam}", fontsize=9)
        ax.set_xlabel("t"); ax.set_ylabel("B_L / CMI"); ax.legend(fontsize=8)
        worst = max(rec["B_L"]["frac"], rec["CMI"]["frac"])
        rec["recommended_T"] = worst * T
        rec["pct_saveable"] = (1 - worst) * 100
        summary[f"L{L}"] = rec
        print(f"  => recommended T (max of both) ~ {worst*T:.0f}  "
              f"(save ~{(1-worst)*100:.0f}% of steps)")

    fig.tight_layout()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(args.outdir) / "bl_saturation.png", dpi=120)
    saveables = [summary[k]["pct_saveable"] for k in summary]
    if saveables:
        worst_save = min(saveables)
        print("\n" + "-" * 72)
        print(f"Conservative T saving (min across L): ~{worst_save:.0f}%  "
              f"=> ~{1/(1-worst_save/100):.2f}x on n_steps if it holds")
        print("Pairs multiplicatively with any safe PPS_DTAU_MULT from bench_dtau.")
        print("=" * 72, flush=True)
    return summary


if __name__ == "__main__":
    sys.exit(main())
