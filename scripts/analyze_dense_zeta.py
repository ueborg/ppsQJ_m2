#!/usr/bin/env python3
"""
Aggregate and analyse fss_*.npz files produced by worker_fss_direct.

Reads from a directory of fss_*.npz files, builds a (L, lam, zeta) -> S_mean
table, fits c_eff = slope of S vs ln L for each (lam, zeta), then:

  1. Locates lam_c(zeta, L) by interpolating where c_eff = 1 for each (zeta, L).
  2. Fits log lam_c = phi * log zeta + const over the dense small-zeta grid.
  3. Checks the 1/L^2 finite-size scaling of lam_c shift vs zeta=1 baseline.
  4. Plots everything.

Usage:
  python scripts/analyze_dense_zeta.py /scratch/.../pps_dense_zeta [--also DIR2 ...]

Optional --also lets you fold in the fss_test results (which have the same
file format) for a combined dataset.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d

# ── helpers ──────────────────────────────────────────────────────────────────

def load_dir(d: Path) -> dict:
    """Load all fss_*.npz files in d, return dict keyed (L,lam,zeta)->row."""
    data = {}
    for p in sorted(d.glob("fss_*.npz")):
        try:
            f = np.load(p, allow_pickle=False)
            key = (int(f["L"]), round(float(f["lam"]), 6),
                   round(float(f["zeta"]), 6))
            data[key] = {
                "S_mean": float(f["S_mean"]),
                "S_err":  float(f["S_err"]),
                "S_var":  float(f["S_var"]),
                "elapsed": float(f["elapsed"]),
            }
        except Exception as e:
            print(f"[warn] {p.name}: {e}", file=sys.stderr)
    return data


def fit_c_eff(Ls, S_means, Ls_fit=None):
    """Fit S = a + c*ln(L) over Ls_fit. Returns (c, a, r^2)."""
    if Ls_fit is None:
        Ls_fit = [L for L in Ls if L >= 32]
    pairs = [(L, S) for L, S in zip(Ls, S_means) if L in Ls_fit]
    if len(pairs) < 3:
        return np.nan, np.nan, np.nan
    xs = np.log([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    slope, intercept, r, *_ = linregress(xs, ys)
    return slope, intercept, r**2


def find_lam_c(lams, c_effs, target=1.0):
    """Interpolate lambda where c_eff crosses target. Returns nan if no crossing."""
    lams = np.array(lams)
    c_effs = np.array(c_effs)
    mask = np.isfinite(c_effs)
    lams, c_effs = lams[mask], c_effs[mask]
    if len(lams) < 2:
        return np.nan
    # find where c_eff crosses target (descending in lam)
    diffs = c_effs - target
    for i in range(len(diffs) - 1):
        if diffs[i] * diffs[i+1] <= 0:
            # linear interp
            t = diffs[i] / (diffs[i] - diffs[i+1])
            return lams[i] + t * (lams[i+1] - lams[i])
    return np.nan


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--also", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or args.input_dir.parent / "dense_zeta_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load all files ────────────────────────────────────────────────────────
    data = load_dir(args.input_dir)
    for d in (args.also or []):
        data.update(load_dir(d))
    print(f"Loaded {len(data)} data points from "
          f"{args.input_dir}" +
          (f" + {[str(d) for d in args.also]}" if args.also else ""))

    Ls_all    = sorted(set(k[0] for k in data))
    lams_all  = sorted(set(k[1] for k in data))
    zetas_all = sorted(set(k[2] for k in data))
    print(f"L values:    {Ls_all}")
    print(f"zeta values: {zetas_all}")
    print(f"lam values:  {[round(l,4) for l in lams_all]}")

    # ── fit c_eff(lam, zeta) at each L subset ─────────────────────────────────
    # For each (lam, zeta), fit c_eff over L in {32,48,64,96,128}
    Ls_fit = [L for L in Ls_all if L <= 128]
    c_eff_table = defaultdict(dict)  # (lam, zeta) -> {L_max: c_eff}

    for zeta in zetas_all:
        for lam in lams_all:
            Ls_avail = sorted(L for L in Ls_all if (L, lam, zeta) in data)
            S_vals   = [data[(L, lam, zeta)]["S_mean"] for L in Ls_avail]
            if len(Ls_avail) >= 3:
                c, a, r2 = fit_c_eff(Ls_avail, S_vals,
                                      Ls_fit=[L for L in Ls_avail if L <= 128])
                c_eff_table[(lam, zeta)]["L128"] = c

    # ── find lam_c(zeta) at L<=128 ────────────────────────────────────────────
    lam_c_by_zeta = {}
    for zeta in zetas_all:
        lams_z = sorted(lam for lam in lams_all
                        if (lam, zeta) in c_eff_table
                        and "L128" in c_eff_table[(lam, zeta)])
        c_effs = [c_eff_table[(lam, zeta)]["L128"] for lam in lams_z]
        lc = find_lam_c(lams_z, c_effs)
        lam_c_by_zeta[zeta] = lc
        print(f"  zeta={zeta:.4f}: lam_c = {lc:.4f}  "
              f"(from {len(lams_z)} lambda points, "
              f"c_eff = {[round(c,3) for c in c_effs]})")

    # ── power-law fit: log lam_c = phi * log zeta + const ────────────────────
    zetas_fit = [z for z in zetas_all if not np.isnan(lam_c_by_zeta.get(z, np.nan))]
    lam_cs    = [lam_c_by_zeta[z] for z in zetas_fit]

    if len(zetas_fit) >= 3:
        log_z  = np.log(zetas_fit)
        log_lc = np.log(lam_cs)
        phi, log_A, r, *_ = linregress(log_z, log_lc)
        A = np.exp(log_A)
        print(f"\nPower-law fit: lam_c = {A:.4f} * zeta^{phi:.4f}  (R²={r**2:.4f})")
        print(f"  Prediction phi=0.5 (chiral vertex): {'CONSISTENT' if abs(phi-0.5)<0.05 else 'INCONSISTENT'}")
        print(f"  Prediction A~0.497 (Born-rule):     A={A:.4f}")
    else:
        phi, A, r = np.nan, np.nan, np.nan
        print("\nNot enough zeta points for power-law fit.")

    # ── plot 1: lam_c vs zeta with power-law fit ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    zs_plot  = [z for z in zetas_fit]
    lcs_plot = [lam_c_by_zeta[z] for z in zs_plot]
    ax.scatter(zs_plot, lcs_plot, color="steelblue", zorder=5, label="measured $\\lambda_c$")
    if not np.isnan(phi):
        zz = np.linspace(min(zs_plot)*0.8, max(zs_plot)*1.2, 200)
        ax.plot(zz, A * zz**phi, "k--",
                label=f"fit: $A \\zeta^\\phi$, $\\phi={phi:.3f}$, $A={A:.3f}$")
        ax.plot(zz, 0.497 * zz**0.5, "r:", alpha=0.7,
                label="prediction: $0.497\\,\\zeta^{0.5}$")
    ax.set_xlabel("$\\zeta$"); ax.set_ylabel("$\\lambda_c$")
    ax.set_title("Phase boundary vs $\\zeta$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]   # log-log version
    if len(zs_plot) >= 2:
        ax.scatter(np.log(zs_plot), np.log(lcs_plot), color="steelblue", zorder=5)
        if not np.isnan(phi):
            lz = np.linspace(np.log(min(zs_plot))*1.1, np.log(max(zs_plot))*0.9, 200)
            ax.plot(lz, log_A + phi*lz, "k--",
                    label=f"slope $\\phi={phi:.3f}$")
            ax.plot(lz, np.log(0.497) + 0.5*lz, "r:",
                    label="slope $0.5$ (prediction)", alpha=0.7)
        ax.set_xlabel("$\\ln\\zeta$"); ax.set_ylabel("$\\ln\\lambda_c$")
        ax.set_title("Log-log: power-law fit")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Dense-$\\zeta$ analysis  —  $\\phi={phi:.3f}\\pm?$  "
                 f"(prediction: 0.5)", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "lam_c_vs_zeta.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")
    plt.close(fig)

    # ── plot 2: c_eff(lam) for each zeta ─────────────────────────────────────
    n_z = len(zetas_all)
    ncols = min(5, n_z)
    nrows = (n_z + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows),
                             squeeze=False)
    for idx, zeta in enumerate(sorted(zetas_all)):
        ax = axes[idx // ncols][idx % ncols]
        lams_z = sorted(lam for lam in lams_all
                        if (lam, zeta) in c_eff_table
                        and "L128" in c_eff_table[(lam, zeta)])
        c_effs = [c_eff_table[(lam, zeta)]["L128"] for lam in lams_z]
        ax.plot(lams_z, c_effs, "o-", color="steelblue")
        ax.axhline(1.0, color="k", lw=0.8, ls="--", label="$c=1$")
        lc = lam_c_by_zeta.get(zeta, np.nan)
        if not np.isnan(lc):
            ax.axvline(lc, color="tomato", lw=1, ls=":", label=f"$\\lambda_c={lc:.3f}$")
        ax.set_title(f"$\\zeta={zeta}$", fontsize=10)
        ax.set_xlabel("$\\lambda$"); ax.set_ylabel("$c_{{\\rm eff}}$")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    for idx in range(n_z, nrows*ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("$c_{{\\rm eff}}(\\lambda)$ per $\\zeta$ — $L\\leq 128$ fit", fontsize=12)
    fig.tight_layout()
    out_path2 = out_dir / "c_eff_per_zeta.png"
    fig.savefig(out_path2, dpi=150)
    print(f"Saved {out_path2}")
    plt.close(fig)

    # ── CSV summary ───────────────────────────────────────────────────────────
    csv_path = out_dir / "lam_c_table.csv"
    with open(csv_path, "w") as f:
        f.write("zeta,lam_c\n")
        for z in sorted(zetas_all):
            f.write(f"{z:.6f},{lam_c_by_zeta.get(z, float('nan')):.6f}\n")
    print(f"Saved {csv_path}")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
