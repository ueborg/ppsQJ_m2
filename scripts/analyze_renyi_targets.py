"""Analyze Renyi entropy targeted reruns and test free-Dirac CFT prediction.

Reads .npz files from a directory containing the output of
submit_renyi_targets.sh, then:
  1) For each (lam, zeta) test point, fits S_n(L) = a_n + c_n * ln L over the
     available L values (32, 48, 64, 96, 128).
  2) Reports c_n and the ratio c_n / c_1 for n in {2, 3}.
  3) Compares against the free-Dirac CFT prediction c_n / c_1 = (1 + 1/n) / 2:
        n=2: 3/4 = 0.750
        n=3: 4/6 = 0.667
  4) Plots S_n vs ln L per (lam, zeta) with fitted slopes overlaid.
  5) Translation-averaged correlation function decay |C(r)| vs r per point,
     fitted to a power law C(r) ~ r^{-eta}. Free Dirac predicts eta = 1.

Usage:
  python scripts/analyze_renyi_targets.py /scratch/$USER/pps_qj/pps_clone_renyi/
"""
from __future__ import annotations
import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_npz_dir(d):
    """Load all .npz files in directory d, keyed by (L, lam, zeta)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "clone_*.npz"))):
        z = np.load(f, allow_pickle=True)
        L = int(z["L"])
        lam = float(z["lam"])
        zeta = float(z["zeta"])
        key = (L, lam, zeta)
        out[key] = {k: z[k] for k in z.files}
    return out


def fit_log_law(L_arr, S_arr, S_err=None):
    """Weighted fit S = a + c * ln L. Returns (a, c, sigma_c)."""
    L_arr = np.asarray(L_arr, dtype=float)
    S_arr = np.asarray(S_arr, dtype=float)
    ln_L = np.log(L_arr)
    if S_err is None or np.any(~np.isfinite(S_err)) or np.any(S_err <= 0):
        w = np.ones_like(L_arr)
    else:
        w = 1.0 / np.asarray(S_err) ** 2
    A = np.column_stack([np.ones_like(ln_L), ln_L])
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    coef = cov @ A.T @ W @ S_arr
    sigma_c = float(np.sqrt(cov[1, 1]))
    return float(coef[0]), float(coef[1]), sigma_c


def fit_power_law(r, c_r, r_min=2, r_max=None):
    """Fit |C(r)| = A * r^{-eta} over r in [r_min, r_max] using log-log linear regression.
    Returns (A, eta, sigma_eta)."""
    r = np.asarray(r, dtype=float)
    c_r = np.asarray(c_r, dtype=float)
    if r_max is None:
        r_max = int(r.max() // 2)  # avoid edge effects beyond L/2
    mask = (r >= r_min) & (r <= r_max) & (c_r > 1e-10)
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    log_r = np.log(r[mask])
    log_c = np.log(c_r[mask])
    A = np.column_stack([np.ones_like(log_r), log_r])
    cov = np.linalg.inv(A.T @ A)
    coef = cov @ A.T @ log_c
    sigma_eta = float(np.sqrt(cov[1, 1]))
    return float(np.exp(coef[0])), float(-coef[1]), sigma_eta


def main(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir.rstrip("/")), "renyi_analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading from: {input_dir}")
    print(f"Writing to:   {output_dir}")

    data = load_npz_dir(input_dir)
    if not data:
        print(f"ERROR: no .npz files found in {input_dir}")
        sys.exit(1)
    print(f"Loaded {len(data)} npz files.")

    # Group by (lam, zeta), check that we have full L range
    by_lz = defaultdict(dict)
    for (L, lam, zeta), v in data.items():
        by_lz[(lam, zeta)][L] = v

    print(f"\nFound {len(by_lz)} (lam, zeta) test points:")
    for (lam, zeta), by_L in sorted(by_lz.items()):
        Ls = sorted(by_L.keys())
        print(f"  (lam={lam:.3f}, zeta={zeta:.3f}): L = {Ls}")

    # =========================================================================
    # CFT TEST: c_n = (c/6) * (1 + 1/n)
    # Predicted ratios: c_2 / c_1 = 3/4, c_3 / c_1 = 2/3
    # =========================================================================
    print("\n" + "=" * 70)
    print("CFT test: c_n / c_1 ratios (free-Dirac prediction in parens)")
    print("=" * 70)
    print(f"{'lam':>6} {'zeta':>6} {'c_1':>8} {'c_2':>8} {'c_3':>8} "
          f"{'c_2/c_1':>10} {'c_3/c_1':>10}")
    print(f"{'':>6} {'':>6} {'':>8} {'':>8} {'':>8} "
          f"{'(=0.750)':>10} {'(=0.667)':>10}")
    print("-" * 70)

    results = {}
    for (lam, zeta), by_L in sorted(by_lz.items()):
        Ls = np.array(sorted(by_L.keys()))
        if len(Ls) < 3:
            continue
        S1 = np.array([float(by_L[L].get("S_mean", np.nan)) for L in Ls])
        S2 = np.array([float(by_L[L].get("S_renyi_2_mean", np.nan)) for L in Ls])
        S3 = np.array([float(by_L[L].get("S_renyi_3_mean", np.nan)) for L in Ls])
        if np.all(np.isnan(S2)) or np.all(np.isnan(S3)):
            print(f"{lam:>6.3f} {zeta:>6.3f}   (Renyi not recorded; was PPS_RECORD_RENYI=1 set?)")
            continue
        try:
            a1, c1, sc1 = fit_log_law(Ls, S1)
            a2, c2, sc2 = fit_log_law(Ls, S2)
            a3, c3, sc3 = fit_log_law(Ls, S3)
            r21 = c2 / c1 if abs(c1) > 1e-6 else float("nan")
            r31 = c3 / c1 if abs(c1) > 1e-6 else float("nan")
            results[(lam, zeta)] = dict(
                Ls=Ls, S1=S1, S2=S2, S3=S3, c1=c1, c2=c2, c3=c3, a1=a1, a2=a2, a3=a3,
                r21=r21, r31=r31, sc1=sc1, sc2=sc2, sc3=sc3,
            )
            print(f"{lam:>6.3f} {zeta:>6.3f} {c1:>8.4f} {c2:>8.4f} {c3:>8.4f} "
                  f"{r21:>10.4f} {r31:>10.4f}")
        except Exception as e:
            print(f"{lam:>6.3f} {zeta:>6.3f}   FIT FAILED: {e}")

    # =========================================================================
    # PLOT 1: S_n vs ln L for each (lam, zeta) point with fitted slopes
    # =========================================================================
    n_pts = len(results)
    if n_pts == 0:
        print("\nNo points with Renyi data. Did you set PPS_RECORD_RENYI=1?")
        return
    ncols = 3
    nrows = int(np.ceil(n_pts / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax, ((lam, zeta), r) in zip(axes.flat, sorted(results.items())):
        ln_L = np.log(r['Ls'])
        for S, c, a, n, color in [
            (r['S1'], r['c1'], r['a1'], 1, 'C0'),
            (r['S2'], r['c2'], r['a2'], 2, 'C1'),
            (r['S3'], r['c3'], r['a3'], 3, 'C2'),
        ]:
            ax.plot(ln_L, S, 'o', color=color, ms=7, label=f'$S_{n}$ (c={c:.3f})')
            ln_extend = np.linspace(ln_L.min() - 0.2, ln_L.max() + 0.2, 50)
            ax.plot(ln_extend, a + c * ln_extend, '--', color=color, alpha=0.5)
        ax.set_xlabel(r'$\ln L$')
        ax.set_ylabel(r'$S_n$')
        ax.set_title(f'lam={lam:.3f}, zeta={zeta:.3f}\n'
                     f'$c_2/c_1$={r["r21"]:.3f}  $c_3/c_1$={r["r31"]:.3f}')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.flat[n_pts:]:
        ax.set_visible(False)
    fig.suptitle('Renyi entropies vs $\\ln L$ — free-Dirac CFT predicts $c_n/c_1$ = $(1+1/n)/2$\n'
                 'i.e. $c_2/c_1=0.75$ and $c_3/c_1=0.667$', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "renyi_log_fits.png"), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {output_dir}/renyi_log_fits.png")

    # =========================================================================
    # PLOT 2: c_n / c_1 ratios as a function of (lam, zeta), vs predictions
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    pts = sorted(results.keys())
    x = np.arange(len(pts))
    r21 = [results[p]['r21'] for p in pts]
    r31 = [results[p]['r31'] for p in pts]
    ax.axhline(0.75, ls='--', color='C1', alpha=0.5, label=r'$c_2/c_1$ Dirac (0.75)')
    ax.axhline(2.0 / 3.0, ls='--', color='C2', alpha=0.5, label=r'$c_3/c_1$ Dirac (0.667)')
    ax.plot(x, r21, 'o-', color='C1', ms=8, label=r'measured $c_2/c_1$')
    ax.plot(x, r31, 's-', color='C2', ms=8, label=r'measured $c_3/c_1$')
    ax.set_xticks(x)
    ax.set_xticklabels([f'lam={l:.2f}\nzeta={z:.2f}' for (l, z) in pts], rotation=0, fontsize=8)
    ax.set_ylabel('Renyi ratio')
    ax.set_ylim(0.4, 1.0)
    ax.set_title('CFT consistency: $c_n/c_1$ measured vs free-Dirac prediction')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "renyi_ratios.png"), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_dir}/renyi_ratios.png")

    # =========================================================================
    # PLOT 3: Correlation function decay |C(r)| vs r per point at L=128
    # Power-law fit: |C(r)| ~ A * r^-eta. Free Dirac: eta = 1.
    # =========================================================================
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    eta_results = {}
    for ax, ((lam, zeta), r) in zip(axes.flat, sorted(results.items())):
        # Use the L=128 corr decay if available; else largest L
        Ls_have = sorted(by_lz[(lam, zeta)].keys())
        L_use = 128 if 128 in Ls_have else max(Ls_have)
        d = by_lz[(lam, zeta)][L_use]
        r_arr = np.asarray(d.get("corr_decay_r", []), dtype=float)
        c_arr = np.asarray(d.get("corr_decay_mean", []), dtype=float)
        if len(r_arr) == 0:
            ax.text(0.5, 0.5, "no corr data", transform=ax.transAxes, ha='center')
            ax.set_title(f'lam={lam:.3f}, zeta={zeta:.3f}')
            continue
        ax.loglog(r_arr, c_arr, 'o', ms=6, label=f'data L={L_use}')
        A, eta, seta = fit_power_law(r_arr, c_arr, r_min=2, r_max=L_use // 2)
        eta_results[(lam, zeta)] = (A, eta, seta)
        if np.isfinite(eta):
            r_fit = np.linspace(2, L_use // 2, 50)
            ax.loglog(r_fit, A * r_fit ** (-eta), '--', alpha=0.6,
                      label=f'fit $\\eta$={eta:.3f}$\\pm${seta:.3f}')
        # Reference line eta=1
        r_ref = np.linspace(2, L_use // 2, 50)
        c_ref0 = c_arr[r_arr == 2][0] if 2 in r_arr else c_arr[0]
        ax.loglog(r_ref, c_ref0 * (2.0 / r_ref), ':', color='red', alpha=0.5, label=r'$\eta=1$ (Dirac)')
        ax.set_xlabel('r'); ax.set_ylabel(r'$|C(r)|$')
        ax.set_title(f'lam={lam:.3f}, zeta={zeta:.3f}')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8)
    for ax in axes.flat[n_pts:]:
        ax.set_visible(False)
    fig.suptitle(r'Correlation function decay $|C(r)| \sim r^{-\eta}$ — free-Dirac CFT predicts $\eta = 1$',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "corr_decay.png"), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_dir}/corr_decay.png")

    # Print eta summary
    print("\n" + "=" * 70)
    print("Correlation function power-law exponent eta (Dirac predicts 1.0):")
    print("=" * 70)
    print(f"{'lam':>6} {'zeta':>6} {'eta':>10} {'sigma_eta':>12}")
    for (lam, zeta), (A, eta, seta) in sorted(eta_results.items()):
        print(f"{lam:>6.3f} {zeta:>6.3f} {eta:>10.4f} {seta:>12.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_renyi_targets.py <input_dir> [<output_dir>]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
