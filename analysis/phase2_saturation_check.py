#!/usr/bin/env python
"""Saturation check for the Phase-2 high-L campaign.

Question this answers
---------------------
The Phase-2 grid inherits time_horizon_v2, which caps T at 100 for L>=96.
At L=256, ballistic entanglement spreading may need T ~ L/v ~ O(256) to
reach steady state.  If S and B_L are still drifting at T=100, the crossing
extracted from a T=100 run is a TRANSIENT, not the steady-state crossing,
and the FSS extrapolation is corrupted at the most important size.

This script runs ONE (L, lambda, zeta) point at several T values and reports
whether the observables have plateaued by T=100.  It reuses the exact
production physics path (_run_one_realisation) so the answer is faithful.

Usage
-----
    python analysis/phase2_saturation_check.py \
        --L 256 --zeta 0.20 --lam 0.29 --N_c 60 \
        --T 60 100 200 400 --seeds 3

Defaults target the decisive point (L=256, zeta=0.20, lambda near the
sqrt-zeta crossing).  N_c is kept modest (the trend in T is what matters,
not absolute precision) so the check is affordable: ~1-2h per T value at
L=256, N_c=60 with 5 realisations.

Interpretation
--------------
  - If S_mean and B_L_mean at T=100 agree with T=200 and T=400 within the
    cross-seed error bars, the T=100 cap is SAFE -> keep it, submit Phase 2.
  - If S_mean / B_L_mean are still rising at T=100 and only flatten by
    T=200-400, the cap is TOO LOW for L=256 -> raise time_horizon_phase2
    accordingly (and accept the higher per-task cost).

The script prints a table and writes a JSON + PNG to outputs/phase2_sat/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _compute_BL_CMI(final_covs, L: int) -> dict:
    """Mean B_L and CMI across clones, via the production batched routine."""
    from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L
    comps = _batched_compute_B_L(final_covs, L)
    out = {}
    for k in ("B_L", "CMI", "S_AB"):
        v = comps[k]
        m = np.isfinite(v)
        out[k] = float(np.mean(v[m])) if m.any() else float("nan")
    return out


def run_point(L: int, lam: float, zeta: float, T: float, N_c: int,
              seeds: int, base_seed: int) -> dict:
    """Run `seeds` realisations at fixed (L, lam, zeta, T, N_c); aggregate."""
    from pps_qj.parallel.worker_clone_pps import _run_one_realisation

    alpha = lam
    w = 1.0 - lam

    S_vals, BL_vals, CMI_vals, SAB_vals = [], [], [], []
    n_fail = 0
    for r in range(seeds):
        # Seed offset 5e10 keeps these checks disjoint from every campaign.
        seed = base_seed + r * 999_983 + 50_000_000_000
        args = dict(L=L, w=w, alpha=alpha, zeta=zeta, T=T, N_c=N_c, seed=seed)
        res = _run_one_realisation(args)
        if not res.get("ok", False):
            n_fail += 1
            continue
        S_vals.append(res["S_mean"])
        bc = _compute_BL_CMI(res["final_covs"], L)
        BL_vals.append(bc["B_L"])
        CMI_vals.append(bc["CMI"])
        SAB_vals.append(bc["S_AB"])

    def _ms(x):
        a = np.asarray(x, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0

    S_m, S_e = _ms(S_vals)
    BL_m, BL_e = _ms(BL_vals)
    CMI_m, CMI_e = _ms(CMI_vals)
    SAB_m, SAB_e = _ms(SAB_vals)
    return dict(
        L=L, lam=lam, zeta=zeta, T=T, N_c=N_c, seeds=seeds, n_fail=n_fail,
        S_mean=S_m, S_err=S_e, B_L_mean=BL_m, B_L_err=BL_e,
        CMI_mean=CMI_m, CMI_err=CMI_e, S_AB_mean=SAB_m, S_AB_err=SAB_e,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-2 T-saturation check")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--zeta", type=float, default=0.20)
    ap.add_argument("--lam", type=float, default=0.29,
                    help="lambda near the expected crossing for this zeta")
    ap.add_argument("--N_c", type=int, default=60,
                    help="clones per realisation (modest: trend matters, not abs precision)")
    ap.add_argument("--T", type=float, nargs="+", default=[60, 100, 200, 400])
    ap.add_argument("--seeds", type=int, default=3,
                    help="independent realisations per T (for cross-seed error)")
    ap.add_argument("--base_seed", type=int, default=20260529)
    ap.add_argument("--outdir", type=str, default="outputs/phase2_sat")
    args = ap.parse_args()

    os.environ.setdefault("PPS_RECORD_RENYI", "0")  # not needed for saturation
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Phase-2 saturation check:  L={args.L}  zeta={args.zeta}  lambda={args.lam}")
    print(f"N_c={args.N_c}  seeds={args.seeds}  T values={args.T}")
    print("=" * 72)
    print(f"{'T':>6} {'S_mean':>12} {'B_L_mean':>14} {'CMI_mean':>14} {'fails':>6} {'wall(s)':>9}")
    print("-" * 72)

    rows = []
    for T in args.T:
        t0 = time.time()
        row = run_point(args.L, args.lam, args.zeta, float(T),
                        args.N_c, args.seeds, args.base_seed)
        wall = time.time() - t0
        row["wall_s"] = wall
        rows.append(row)
        print(f"{T:>6.0f} {row['S_mean']:>8.4f}±{row['S_err']:<.4f} "
              f"{row['B_L_mean']:>9.4f}±{row['B_L_err']:<.4f} "
              f"{row['CMI_mean']:>9.4f}±{row['CMI_err']:<.4f} "
              f"{row['n_fail']:>6d} {wall:>9.1f}", flush=True)

    # Verdict: is T=100 within error of the largest T?
    print("-" * 72)
    by_T = {r["T"]: r for r in rows}
    T_ref = max(args.T)
    verdict_lines = []
    if 100.0 in by_T and T_ref > 100.0:
        for key, label in (("S_mean", "S"), ("B_L_mean", "B_L"), ("CMI_mean", "CMI")):
            ek = key.replace("_mean", "_err")
            d = abs(by_T[100.0][key] - by_T[T_ref][key])
            comb_err = (by_T[100.0][ek]**2 + by_T[T_ref][ek]**2) ** 0.5
            nsig = d / comb_err if comb_err > 0 else float("inf")
            status = "OK (plateaued)" if nsig < 2 else "DRIFTING -- raise T"
            verdict_lines.append(
                f"  {label:>4}: |T=100 - T={T_ref:.0f}| = {d:.4f}  "
                f"({nsig:.1f} sigma)  -> {status}")
        print("VERDICT (T=100 vs T={:.0f}):".format(T_ref))
        for ln in verdict_lines:
            print(ln)
    else:
        print("VERDICT: include T=100 and a larger T to get an automatic verdict.")

    out = dict(params=vars(args), rows=rows, verdict=verdict_lines)
    jpath = outdir / f"sat_L{args.L}_zeta{args.zeta}_lam{args.lam}.json"
    jpath.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {jpath}")

    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ts = [r["T"] for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (key, lab) in zip(
            axes, (("S_mean", "<S>"), ("B_L_mean", "<B_L>"), ("CMI_mean", "<CMI>"))):
            ek = key.replace("_mean", "_err")
            ys = [r[key] for r in rows]
            es = [r[ek] for r in rows]
            ax.errorbar(Ts, ys, yerr=es, marker="o", capsize=3)
            ax.axvline(100, ls="--", color="r", alpha=0.6, label="T=100 cap")
            ax.set_xlabel("T")
            ax.set_ylabel(lab)
            ax.set_title(f"{lab} vs T  (L={args.L}, zeta={args.zeta})")
            ax.legend()
        fig.tight_layout()
        ppath = outdir / f"sat_L{args.L}_zeta{args.zeta}_lam{args.lam}.png"
        fig.savefig(ppath, dpi=130)
        print(f"Wrote {ppath}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
