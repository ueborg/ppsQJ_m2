#!/usr/bin/env python3
"""
Aggregate and analyse fss_*.npz files produced by worker_fss_direct.

Uses Binder cumulant B_L crossings to locate lambda_c(zeta), then fits
the power-law log(lam_c) = phi * log(zeta) + const.

NOTE: c_eff=1 threshold is NOT used — it was identified as unreliable
(cf. CONTEXT.md: "c_eff threshold method gives misleading results — abandoned").
The correct observable is the Binder crossing B_L(lambda) = B_{L'}(lambda).

Usage:
    python scripts/analyze_dense_zeta.py /scratch/.../pps_dense_zeta [--also DIR ...]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_dir(d: Path) -> dict:
    data = {}
    for p in sorted(d.glob("fss_*.npz")):
        try:
            f = np.load(p, allow_pickle=False)
            key = (int(f["L"]), round(float(f["lam"]), 6),
                   round(float(f["zeta"]), 6))
            row = {k: float(f[k]) for k in
                   ["S_mean", "S_err", "S_var"] if k in f}
            # B_L present only if worker was patched (new runs)
            row["B_L_mean"] = float(f["B_L_mean"]) if "B_L_mean" in f else np.nan
            row["B_L_err"]  = float(f["B_L_err"])  if "B_L_err"  in f else np.nan
            data[key] = row
        except Exception as e:
            print(f"[warn] {p.name}: {e}", file=sys.stderr)
    return data


# ── Binder crossing ───────────────────────────────────────────────────────────

def binder_crossing(lams_sorted, BL_sorted, BLp_sorted):
    """
    Find crossing of B_L(lambda) and B_{L'}(lambda) by linear interpolation.
    Returns (lam_cross, B_cross) or (nan, nan) if no crossing found.
    """
    lams = np.array(lams_sorted)
    diff = np.array(BL_sorted) - np.array(BLp_sorted)
    for i in range(len(diff) - 1):
        if np.isnan(diff[i]) or np.isnan(diff[i+1]):
            continue
        if diff[i] * diff[i+1] <= 0:
            t = diff[i] / (diff[i] - diff[i+1])
            lc = lams[i] + t * (lams[i+1] - lams[i])
            bc = BL_sorted[i] + t * (BLp_sorted[i] - BL_sorted[i+1])
            return float(lc), float(bc)
    return np.nan, np.nan


# ── c_eff fit (for reference only, not used for lam_c) ───────────────────────

def fit_c_eff(Ls, S_means):
    pairs = [(L, S) for L, S in zip(Ls, S_means)
             if L >= 32 and np.isfinite(S)]
    if len(pairs) < 3:
        return np.nan
    xs = np.log([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    slope, *_ = linregress(xs, ys)
    return slope


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--also", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or args.input_dir.parent / "dense_zeta_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dir(args.input_dir)
    for d in (args.also or []):
        data.update(load_dir(d))
    print(f"Loaded {len(data)} points.")

    Ls_all    = sorted(set(k[0] for k in data))
    lams_all  = sorted(set(k[1] for k in data))
    zetas_all = sorted(set(k[2] for k in data))
    print(f"L:    {Ls_all}")
    print(f"zeta: {zetas_all}")

    has_B_L = any(np.isfinite(v["B_L_mean"]) for v in data.values())
    if not has_B_L:
        print("\nWARNING: No B_L data found in these files.")
        print("These were run with the OLD worker_fss_direct before the B_L patch.")
        print("Re-run the jobs after pulling the patched worker to get B_L.")
        print("\nFalling back to c_eff diagnostic (NOT for lam_c extraction).")
        print("The c_eff=1 threshold is NOT a reliable lam_c estimator.\n")
    else:
        print(f"\nB_L data present. Using Binder crossings for lam_c.\n")

    # ── Binder crossings (preferred, only if B_L present) ────────────────────
    lam_c_binder = {}   # (zeta, L_pair) -> lam_c

    if has_B_L:
        L_pairs = [(Ls_all[i], Ls_all[i+1]) for i in range(len(Ls_all)-1)
                   if Ls_all[i] >= 32]

        print("Binder crossings B_L / B_{L'} (lam_c per zeta per L-pair):")
        for zeta in zetas_all:
            print(f"\n  zeta={zeta:.4f}:")
            for (L, Lp) in L_pairs:
                lams_shared = sorted(
                    lam for lam in lams_all
                    if (L, lam, zeta) in data and (Lp, lam, zeta) in data
                    and np.isfinite(data[(L,  lam, zeta)]["B_L_mean"])
                    and np.isfinite(data[(Lp, lam, zeta)]["B_L_mean"])
                )
                if len(lams_shared) < 2:
                    print(f"    L={L}/L'={Lp}: insufficient points")
                    continue
                BL  = [data[(L,  l, zeta)]["B_L_mean"] for l in lams_shared]
                BLp = [data[(Lp, l, zeta)]["B_L_mean"] for l in lams_shared]
                lc, bc = binder_crossing(lams_shared, BL, BLp)
                lam_c_binder[(zeta, L, Lp)] = lc
                status = f"{lc:.4f}" if np.isfinite(lc) else "no crossing in window"
                print(f"    L={L}/L'={Lp}: lam_c = {status}  "
                      f"(B_L range [{min(BL):.3f}, {max(BL):.3f}])")

        # Best estimate: largest-L pair available
        lam_c_best = {}
        for zeta in zetas_all:
            candidates = [(L, Lp, lam_c_binder[(zeta, L, Lp)])
                          for (z, L, Lp) in lam_c_binder if z == zeta
                          and np.isfinite(lam_c_binder[(z, L, Lp)])]
            if candidates:
                # Use the pair with largest Lp
                best = max(candidates, key=lambda x: x[1])
                lam_c_best[zeta] = best[2]
                print(f"\n  zeta={zeta:.4f}: best lam_c = {best[2]:.4f} "
                      f"(from L={best[0]}/L'={best[1]})")
            else:
                lam_c_best[zeta] = np.nan
                print(f"\n  zeta={zeta:.4f}: no reliable Binder crossing")

    # ── c_eff for diagnostic / fallback ──────────────────────────────────────
    c_eff_table = {}
    for zeta in zetas_all:
        for lam in lams_all:
            Ls_avail = [L for L in Ls_all if (L, lam, zeta) in data]
            S_vals   = [data[(L, lam, zeta)]["S_mean"] for L in Ls_avail]
            if len(Ls_avail) >= 3:
                c_eff_table[(lam, zeta)] = fit_c_eff(Ls_avail, S_vals)

    # ── power-law fit ─────────────────────────────────────────────────────────
    lam_c_for_fit = lam_c_best if has_B_L else {}
    phi = A = np.nan

    if len(lam_c_for_fit) >= 3:
        zf  = [z for z in zetas_all if np.isfinite(lam_c_for_fit.get(z, np.nan))]
        lcf = [lam_c_for_fit[z] for z in zf]
        phi, log_A, r, *_ = linregress(np.log(zf), np.log(lcf))
        A = np.exp(log_A)
        print(f"\nPower-law fit (Binder): lam_c = {A:.4f} * zeta^{phi:.4f}  "
              f"(R²={r**2:.4f})")
        print(f"  Prediction phi=0.5 (chiral vertex): "
              f"{'CONSISTENT' if abs(phi-0.5)<0.1 else 'TENSION'}")
    else:
        print("\nNot enough Binder crossings for power-law fit.")
        if not has_B_L:
            print("  -> Re-run jobs with patched worker to get B_L data.")

    # ── plots ─────────────────────────────────────────────────────────────────

    # Plot 1: lam_c vs zeta
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    if has_B_L and lam_c_for_fit:
        zs  = [z for z in zetas_all if np.isfinite(lam_c_for_fit.get(z, np.nan))]
        lcs = [lam_c_for_fit[z] for z in zs]
        ax.scatter(zs, lcs, color="steelblue", zorder=5,
                   label="Binder crossing $\\lambda_c$")
        if not np.isnan(phi):
            zz = np.linspace(min(zs)*0.8, max(zs)*1.2, 300)
            ax.plot(zz, A*zz**phi, "k--",
                    label=f"fit $A\\zeta^\\phi$, $\\phi={phi:.3f}$, $A={A:.3f}$")
            ax.plot(zz, 0.364*zz**0.5, "r:", alpha=0.7,
                    label="$0.364\\,\\zeta^{0.5}$ (our Born)")
            ax.plot(zz, 0.497*zz**0.5, "g:", alpha=0.7,
                    label="$0.497\\,\\zeta^{0.5}$ (Carollo)")
    else:
        ax.text(0.5, 0.5, "No B_L data\nRe-run with patched worker",
                ha="center", va="center", transform=ax.transAxes,
                color="tomato", fontsize=12)
    ax.set_xlabel("$\\zeta$"); ax.set_ylabel("$\\lambda_c$")
    ax.set_title("Phase boundary from Binder crossings")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if has_B_L and len(lam_c_for_fit) >= 2:
        ax.scatter(np.log(zs), np.log(lcs), color="steelblue", zorder=5)
        if not np.isnan(phi):
            lz = np.linspace(min(np.log(zs))*1.1, max(np.log(zs))*0.9, 200)
            ax.plot(lz, np.log(A) + phi*lz, "k--", label=f"slope $\\phi={phi:.3f}$")
            ax.plot(lz, np.log(0.364) + 0.5*lz, "r:", label="slope 0.5 (Born)", alpha=0.7)
            ax.plot(lz, np.log(0.497) + 0.5*lz, "g:", label="slope 0.5 (Carollo)", alpha=0.7)
        ax.set_xlabel("$\\ln\\zeta$"); ax.set_ylabel("$\\ln\\lambda_c$")
        ax.set_title("Log-log: power-law fit")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient Binder\ncrossings for fit",
                ha="center", va="center", transform=ax.transAxes,
                color="grey", fontsize=12)

    fig.suptitle(f"Dense-$\\zeta$ phase boundary  —  $\\phi={phi:.3f}$  "
                 f"(prediction: 0.5)", fontsize=12)
    fig.tight_layout()
    p1 = out_dir / "lam_c_vs_zeta.png"
    fig.savefig(p1, dpi=150); plt.close(fig)
    print(f"\nSaved {p1}")

    # Plot 2: B_L(lambda) curves per zeta (only if data present)
    if has_B_L:
        n_z = len(zetas_all)
        ncols = min(4, n_z)
        nrows = (n_z + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows),
                                 squeeze=False)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Ls_all)))
        for idx, zeta in enumerate(sorted(zetas_all)):
            ax = axes[idx // ncols][idx % ncols]
            for Li, L in enumerate(Ls_all):
                lams_z = sorted(lam for lam in lams_all
                                if np.isfinite(data.get((L, lam, zeta),
                                               {}).get("B_L_mean", np.nan)))
                if not lams_z:
                    continue
                BLs = [data[(L, lam, zeta)]["B_L_mean"] for lam in lams_z]
                errs = [data[(L, lam, zeta)]["B_L_err"]  for lam in lams_z]
                ax.errorbar(lams_z, BLs, yerr=errs, fmt="o-",
                            color=colors[Li], label=f"L={L}", ms=4, lw=1)
            lc = lam_c_best.get(zeta, np.nan)
            if np.isfinite(lc):
                ax.axvline(lc, color="tomato", ls=":", lw=1.2,
                           label=f"$\\lambda_c={lc:.3f}$")
            ax.set_title(f"$\\zeta={zeta}$", fontsize=10)
            ax.set_xlabel("$\\lambda$"); ax.set_ylabel("$B_L$")
            ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)
        for idx in range(n_z, nrows*ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.suptitle("Binder cumulant $B_L(\\lambda)$ per $\\zeta$", fontsize=12)
        fig.tight_layout()
        p2 = out_dir / "binder_per_zeta.png"
        fig.savefig(p2, dpi=150); plt.close(fig)
        print(f"Saved {p2}")

    # Plot 3: c_eff for reference
    n_z = len(zetas_all)
    ncols = min(4, n_z)
    nrows = (n_z + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.5*nrows),
                             squeeze=False)
    for idx, zeta in enumerate(sorted(zetas_all)):
        ax = axes[idx // ncols][idx % ncols]
        lams_z = sorted(lam for lam in lams_all
                        if (lam, zeta) in c_eff_table
                        and np.isfinite(c_eff_table[(lam, zeta)]))
        c_effs = [c_eff_table[(lam, zeta)] for lam in lams_z]
        ax.plot(lams_z, c_effs, "o-", color="steelblue")
        ax.axhline(1.0, color="k", lw=0.8, ls="--", label="$c=1$")
        ax.set_title(f"$\\zeta={zeta}$", fontsize=10)
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("$c_{{\\rm eff}}$ (diagnostic only)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    for idx in range(n_z, nrows*ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("$c_{{\\rm eff}}$ — diagnostic only, NOT used for $\\lambda_c$",
                 fontsize=12, color="tomato")
    fig.tight_layout()
    p3 = out_dir / "c_eff_diagnostic.png"
    fig.savefig(p3, dpi=150); plt.close(fig)
    print(f"Saved {p3}")

    # CSV
    csv_path = out_dir / "lam_c_table.csv"
    with open(csv_path, "w") as f:
        f.write("zeta,lam_c_binder,method\n")
        for z in sorted(zetas_all):
            lc = lam_c_for_fit.get(z, np.nan)
            method = "binder" if has_B_L and np.isfinite(lc) else "missing_B_L"
            f.write(f"{z:.6f},{lc:.6f},{method}\n")
    print(f"Saved {csv_path}")
    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
