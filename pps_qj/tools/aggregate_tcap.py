"""Aggregate validate_tcap_worker outputs into a saturation-time report.

For each (L, lam, zeta), pools the S_history arrays across seeds, computes a
seed-averaged S(t), and finds the earliest t such that the moving-window mean
of S has stabilised — i.e. it differs from the final-window mean by less than
``tol * S_final``.  That ``t_sat`` is the recommended minimum production T;
the full ``T`` only needs to exceed ``t_sat`` plus an averaging window.

Usage::

    python -m pps_qj.tools.aggregate_tcap /scratch/$USER/pps_qj/tcap_validation_*

Produces, per (L, lam, zeta):
  * t_sat at three tolerances (1%, 2%, 5%)
  * S_late ± seed-spread
  * Suggested production T = max(t_sat at 2%, current cap)
"""
from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict

import numpy as np


def _saturation_time(
    t_grid: np.ndarray,
    S_seed_mean: np.ndarray,
    tol: float,
    avg_window_frac: float = 0.2,
) -> float:
    """Earliest t at which |⟨S⟩_t - S_late| / |S_late| < tol.

    Uses a forward moving-window mean of width ``avg_window_frac * len(t_grid)``
    and compares against the mean of the last ``avg_window_frac`` window
    (``S_late``).  Returns NaN if the trajectory never enters the band.
    """
    n = len(S_seed_mean)
    if n < 4:
        return float("nan")
    win = max(1, int(avg_window_frac * n))
    # late mean = mean over last `win` points
    S_late = float(np.mean(S_seed_mean[-win:]))
    # smoothed forward S(t) using a centered window
    kernel = np.ones(win, dtype=np.float64) / win
    smoothed = np.convolve(S_seed_mean, kernel, mode="same")
    if abs(S_late) < 1e-15:
        return float("nan")
    rel_err = np.abs(smoothed - S_late) / abs(S_late)
    # First index where rel_err < tol AND remains < tol thereafter
    in_band = rel_err < tol
    if not in_band.any():
        return float("nan")
    # Find earliest index from which all subsequent are in-band
    idx_first = n
    for i in range(n - 1, -1, -1):
        if in_band[i]:
            idx_first = i
        else:
            break
    return float(t_grid[idx_first]) if idx_first < n else float("nan")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python -m pps_qj.tools.aggregate_tcap <outdir>"
        )
    outdir = sys.argv[1]
    files  = sorted(glob.glob(os.path.join(outdir, "tcap_*.npz")))
    if not files:
        print(f"No tcap_*.npz files in {outdir}", file=sys.stderr)
        return 1

    # Group by (L, lam, zeta), collect S_history arrays
    grouped: dict = defaultdict(list)
    for fp in files:
        try:
            d = np.load(fp, allow_pickle=False)
        except Exception as e:
            print(f"  [warn] could not read {os.path.basename(fp)}: {e}",
                  file=sys.stderr)
            continue
        if not bool(d["ok"]):
            print(f"  [collapsed] {os.path.basename(fp)}", file=sys.stderr)
            continue
        key = (int(d["L"]), float(d["lam"]), float(d["zeta"]))
        grouped[key].append({
            "seed":      int(d["seed"]),
            "S_history": np.asarray(d["S_history"]),
            "ess_history": np.asarray(d["ess_history"]),
            "delta_tau": float(d["delta_tau"]),
            "T":         float(d["T"]),
            "wall":      float(d["wall_time"]),
            "min_ess":   float(d["min_ess_frac"]),
            "anc":       int(d["n_distinct_ancestors"]),
        })

    if not grouped:
        print("No usable runs.", file=sys.stderr)
        return 1

    print(f"Read {sum(len(v) for v in grouped.values())} successful "
          f"realisations across {len(grouped)} (L, λ, ζ) configurations.\n")

    print("=" * 95)
    print(f"{'L':>3}  {'λ':>5}  {'ζ':>5}  "
          f"{'T_run':>6}  {'⟨S_late⟩':>9}  {'σ_seeds':>8}  "
          f"{'t_sat 1%':>9}  {'t_sat 2%':>9}  {'t_sat 5%':>9}  "
          f"{'recom T':>8}")
    print("=" * 95)

    for key in sorted(grouped.keys()):
        L, lam, zeta = key
        seeds = grouped[key]
        # Trim each S_history to the same length (smallest)
        min_len = min(len(s["S_history"]) for s in seeds)
        if min_len < 4:
            print(f"  L={L} λ={lam:.2f} ζ={zeta:.2f}: too few steps")
            continue
        S_arr   = np.stack([s["S_history"][:min_len] for s in seeds])  # (n_seeds, n_steps)
        dt      = seeds[0]["delta_tau"]
        T_run   = seeds[0]["T"]
        t_grid  = (np.arange(min_len) + 1) * dt

        # Seed-averaged S(t)
        S_mean_t = S_arr.mean(axis=0)
        # Late-window mean and seed spread
        win        = max(1, int(0.2 * min_len))
        S_late_per_seed = S_arr[:, -win:].mean(axis=1)
        S_late_mean     = float(S_late_per_seed.mean())
        S_late_spread   = float(S_late_per_seed.std(ddof=1)) \
                          if len(seeds) > 1 else 0.0

        t_sat_1pct = _saturation_time(t_grid, S_mean_t, 0.01)
        t_sat_2pct = _saturation_time(t_grid, S_mean_t, 0.02)
        t_sat_5pct = _saturation_time(t_grid, S_mean_t, 0.05)

        recommended_T = (
            t_sat_2pct + 0.5 * T_run
            if not np.isnan(t_sat_2pct)
            else float("nan")
        )

        print(f"{L:>3}  {lam:>5.2f}  {zeta:>5.2f}  "
              f"{T_run:>6.0f}  {S_late_mean:>9.4f}  {S_late_spread:>8.4f}  "
              f"{t_sat_1pct:>9.2f}  {t_sat_2pct:>9.2f}  {t_sat_5pct:>9.2f}  "
              f"{recommended_T:>8.1f}")

    print("=" * 95)
    print("\n'recom T' = t_sat at 2% tolerance + half of T_run for averaging.")
    print("If recom T < current production cap, the cap could be tightened.")
    print("If recom T > current production cap, the current results may be ")
    print("under-equilibrated — investigate before drawing scientific conclusions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
