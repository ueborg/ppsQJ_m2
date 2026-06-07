"""
fit_opdim.py -- extract Delta_B (hence y_zeta, phi) and the Jian/Foster
multifractal exponents from worker_opdim_pps output.

Reads all opdim_*.npz in a directory, and for each (L, lam):
  * C_sc(r) = bulk-windowed trajectory covariance of the bond expectation
        C_sc(r) = mean_{bulk x} [ <b_x b_{x+r}>_traj - <b_x>_traj <b_{x+r}>_traj ],
    fitted as |C_sc(r)| ~ r^{-2 Delta_B}  ->  Delta_B, y_zeta = 2 - Delta_B,
    phi_Born = y_lambda / y_zeta = 1/(2 y_zeta)  (y_lambda = 1/2, Foster/Jian).
    NB: this is the LOCAL BORN-CORNER exponent. It governs
        lambda_c(1) - lambda_c(zeta) ~ (1-zeta)^{phi_Born}  near zeta=1,
    NOT the global lambda_c(zeta) ~ zeta^phi (that is the small-zeta endpoint,
    a different fixed point).  See theory/Y_ZETA_DERIVATION.md Sec. 7/11.
  * X1     from <g(r)>_traj         ~ r^{-2 X1}
  * X_typ  from exp<log g(r)>_traj  ~ r^{-2 X_typ}
  * x2     from Var(log g(r))_traj  ~ -2 x2 * log r
  * c_ent  from S_half(L) ~ (c_ent/...) log L across the L grid (at fixed lam).

Power-law fits use a bulk r-window [r_lo, r_hi] (default r_lo=3, r_hi=0.55*r_max)
to avoid short-distance lattice effects and long-distance finite-size/noise.
lambda_c(zeta=1) is identified as the lam with the cleanest C_sc power law
(largest fit R^2), and Delta_B(lambda_c) is the reported IR dimension.

CAVEAT: the raw squared-Majorana g(r) carries sublattice (even/odd-r) structure
in this Kitaev/SSH model; X_typ/x2 here use the magnitude and should be sanity-
checked against the per-(L,lam) plots.  The bond-bond C_sc(r) correlates the SAME
operator translated, so it is free of that alternation and is the clean channel
for Delta_B / y_zeta (the primary deliverable).

Usage:
  python analysis/fit_opdim.py <dir_with_opdim_npz> [--rlo 3] [--rhi-frac 0.55]
"""
from __future__ import annotations
import sys, glob, os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def _loglog_slope(r, y, r_lo, r_hi):
    """Fit log|y| = a + slope*log r over r in [r_lo, r_hi]. Returns slope, R^2, n."""
    r = np.asarray(r, float); y = np.asarray(y, float)
    m = np.isfinite(y) & (np.abs(y) > 0) & (r >= r_lo) & (r <= r_hi)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())
    lx, ly = np.log(r[m]), np.log(np.abs(y[m]))
    A = np.polyfit(lx, ly, 1)
    slope = float(A[0])
    pred = np.polyval(A, lx)
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return slope, r2, int(m.sum())


def _lin_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan
    A = np.polyfit(x[m], y[m], 1)
    pred = np.polyval(A, x[m])
    ss_res = np.sum((y[m] - pred) ** 2)
    ss_tot = np.sum((y[m] - y[m].mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return float(A[0]), r2


def _csc(b, L, bulk_frac):
    """C_sc(r) = bulk-windowed trajectory covariance of bond expectation b (N, L-1)."""
    w0 = int(round(bulk_frac * L))
    lo, hi = w0, (L - 1) - w0                 # bulk bonds [lo, hi)
    mean_b = b.mean(0)
    rmax = max(hi - lo - 1, 0)
    rs, cs = [], []
    for r in range(1, rmax + 1):
        x = np.arange(lo, hi - r)
        if x.size == 0:
            continue
        cov = (b[:, x] * b[:, x + r]).mean(0) - mean_b[x] * mean_b[x + r]
        rs.append(r); cs.append(float(np.mean(cov)))
    return np.asarray(rs, float), np.asarray(cs, float)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python analysis/fit_opdim.py <dir> [--rlo R] [--rhi-frac F]")
    d = sys.argv[1]
    r_lo = 3.0
    rhi_frac = 0.55
    if "--rlo" in sys.argv:
        r_lo = float(sys.argv[sys.argv.index("--rlo") + 1])
    if "--rhi-frac" in sys.argv:
        rhi_frac = float(sys.argv[sys.argv.index("--rhi-frac") + 1])

    files = sorted(glob.glob(os.path.join(d, "opdim_*.npz")))
    if not files:
        raise SystemExit(f"no opdim_*.npz in {d}")

    rows = []   # (L, lam, Delta_B, y_zeta, phi, R2_csc, X1, X_typ, x2, S_half, N)
    per = {}    # (L,lam) -> dict for plotting
    for f in files:
        dat = np.load(f)
        L = int(dat["L"]); lam = float(dat["lam"]); bf = float(dat["bulk_frac"])
        N = int(dat["N_traj"])
        b, g = dat["b"], dat["g"]

        # --- C_sc -> Delta_B ---
        r_csc, csc = _csc(b, L, bf)
        rhi = max(r_lo + 2, rhi_frac * (r_csc.max() if r_csc.size else r_lo))
        slope_csc, r2_csc, _ = _loglog_slope(r_csc, csc, r_lo, rhi)
        Delta_B = -slope_csc / 2.0
        y_zeta = 2.0 - Delta_B
        phi = (1.0 / (2.0 * y_zeta)) if (np.isfinite(y_zeta) and y_zeta != 0) else np.nan

        # --- multifractal from g ---
        rg = np.arange(1, g.shape[1] + 1)
        gbar = g.mean(0)
        logg = np.log(np.clip(g, 1e-300, None))
        logg_mean = logg.mean(0)
        logg_var = logg.var(0)
        rhi_g = max(r_lo + 2, rhi_frac * rg.max())
        sX1, _, _ = _loglog_slope(rg, gbar, r_lo, rhi_g)
        X1 = -sX1 / 2.0
        # X_typ: exp(logg_mean) ~ r^{-2 X_typ}  =>  logg_mean ~ -2 X_typ log r
        mt = np.isfinite(logg_mean) & (rg >= r_lo) & (rg <= rhi_g)
        sXt = np.polyfit(np.log(rg[mt]), logg_mean[mt], 1)[0] if mt.sum() >= 3 else np.nan
        X_typ = -sXt / 2.0
        mv = np.isfinite(logg_var) & (rg >= r_lo) & (rg <= rhi_g)
        sx2 = np.polyfit(np.log(rg[mv]), logg_var[mv], 1)[0] if mv.sum() >= 3 else np.nan
        x2 = -sx2 / 2.0

        S_half = float(np.mean(dat["S_half"]))
        rows.append((L, lam, Delta_B, y_zeta, phi, r2_csc, X1, X_typ, x2, S_half, N))
        per[(L, lam)] = dict(r_csc=r_csc, csc=csc, rg=rg, gbar=gbar,
                             logg_mean=logg_mean, logg_var=logg_var)

    rows.sort()
    print("\n=== per-(L, lam) fits  (y_lambda=1/2 assumed; phi_Born=1/(2 y_zeta)) ===")
    hdr = f"{'L':>4} {'lam':>5} | {'Delta_B':>8} {'y_zeta':>7} {'phi':>6} {'R2':>5} |" \
          f" {'X1':>6} {'X_typ':>6} {'x2':>6} | {'S(L/2)':>7} {'N':>5}"
    print(hdr); print("-" * len(hdr))
    for (L, lam, Db, yz, ph, r2, X1, Xt, x2, Sh, N) in rows:
        print(f"{L:>4} {lam:>5.3f} | {Db:>8.3f} {yz:>7.3f} {ph:>6.3f} {r2:>5.2f} |"
              f" {X1:>6.2f} {Xt:>6.2f} {x2:>6.2f} | {Sh:>7.3f} {N:>5}")

    # --- identify lambda_c per L = lam with cleanest C_sc power law (max R^2) ---
    print("\n=== lambda_c(zeta=1) per L (max-R^2 C_sc power law) ===")
    Ls = sorted(set(r[0] for r in rows))
    chosen = {}
    for L in Ls:
        sub = [r for r in rows if r[0] == L and np.isfinite(r[5])]
        if not sub:
            continue
        best = max(sub, key=lambda r: r[5])
        chosen[L] = best
        print(f"  L={L}: lambda_c~{best[1]:.3f}  Delta_B={best[2]:.3f}  "
              f"y_zeta={best[3]:.3f}  phi={best[4]:.3f}  (R2={best[5]:.2f})")

    # --- combined estimate at the largest L (least finite-size bias) ---
    if chosen:
        Lbig = max(chosen)
        best = chosen[Lbig]
        print("\n=== PRIMARY RESULT (largest L) ===")
        print(f"  L={Lbig}, lambda_c~{best[1]:.3f}")
        print(f"  Delta_B(lambda_c) = {best[2]:.3f}   "
              f"[no-click/UV value ~1.0; this IS the Born-corner IR dimension]")
        print(f"  y_zeta = 2 - Delta_B = {best[3]:.3f}")
        print(f"  phi    = 1/(2 y_zeta) = {best[4]:.3f}")
        # CORRECT cross-check (Y_ZETA_DERIVATION Sec. 7/11): predict the LOCAL boundary
        # slope and test it against boundary data NEAR zeta=1, NOT the extract_yzeta
        # zeta->0 collapse (that is the small-zeta endpoint, a different fixed point).
        print(f"  => PREDICT  lambda_c(1)-lambda_c(zeta) ~ (1-zeta)^{best[4]:.3f}  near zeta=1")
        print( "     TEST: fit boundary at zeta>~0.7 (the (1-zeta) regime).")
        print( "     (do NOT cross-check against extract_yzeta.py zeta->0 collapse)")

    # --- c_ent: S_half vs log L at the chosen lambda_c (use the modal lam) ---
    if len(chosen) >= 2:
        lam_star = np.median([chosen[L][1] for L in chosen])
        # nearest available lam to lam_star per L
        pts = []
        for L in Ls:
            sub = [r for r in rows if r[0] == L]
            if sub:
                rr = min(sub, key=lambda r: abs(r[1] - lam_star))
                pts.append((L, rr[9]))
        if len(pts) >= 2:
            Lv = np.array([p[0] for p in pts], float)
            Sv = np.array([p[1] for p in pts], float)
            slope, r2 = _lin_slope(np.log(Lv), Sv)
            print(f"\n=== c_ent (S(L/2) vs log L at lam~{lam_star:.3f}) ===")
            print(f"  slope d S(L/2)/d log L = {slope:.3f}  (R2={r2:.2f})   "
                  f"[Jian/Foster c_ent = 0.39 for vN]")

    # --- plots ---
    if HAVE_MPL and per:
        keys = sorted(per)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for (L, lam) in keys:
            P = per[(L, lam)]
            m = np.abs(P["csc"]) > 0
            ax[0].loglog(P["r_csc"][m], np.abs(P["csc"][m]), "o-", ms=3,
                         label=f"L={L},lam={lam:.2f}")
            mg = np.isfinite(P["gbar"]) & (P["gbar"] > 0)
            ax[1].loglog(P["rg"][mg], P["gbar"][mg], "o-", ms=3,
                         label=f"L={L},lam={lam:.2f}")
        ax[0].set_xlabel("r (bond separation)"); ax[0].set_ylabel("|C_sc(r)|")
        ax[0].set_title(r"single-copy-mass corr.  $|C_{sc}|\sim r^{-2\Delta_B}$")
        ax[1].set_xlabel("r (Majorana separation)"); ax[1].set_ylabel(r"$\overline{g}(r)$")
        ax[1].set_title(r"squared Majorana corr.  $\overline{g}\sim r^{-2X_1}$")
        for a in ax:
            a.grid(alpha=0.3, which="both"); a.legend(fontsize=6, ncol=2)
        out_png = os.path.join(d, "opdim_fits.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
