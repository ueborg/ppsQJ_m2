#!/usr/bin/env python3
"""
fit_areaphase.py -- GATE 2 analysis: discriminate phi=1/2 vs phi=1 from the
area-phase correlation length xi(zeta, lambda).

  python analysis/fit_areaphase.py /scratch/$USER/pps_qj/pps_areaphase

Loads areaphase_*.npz, then for each L:
  (1) lambda-flatness: at fixed (L,zeta), is xi independent of lambda?  (window-law
      signature; coherent-channel xi would track lambda).
  (2) zeta-scaling:    fit xi(zeta) ~ A * zeta^{-p} at fixed (L, lambda-offset).
      p ~ 1/2  -> phi=1/2 (window law);  p ~ 1 -> phi=1 (coherent).
  (3) odd-r null: odd_null should be ~0 if the two-chain decoupling holds.
"""
import sys, glob, os
import numpy as np


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "areaphase_*.npz"))):
        z = np.load(f, allow_pickle=True)
        rows.append(dict(L=int(z["L"]), lam=float(z["lam"]), zeta=float(z["zeta"]),
                         lam_c=float(z["lam_c_center"]),
                         xi=float(z["xi_mean"]), xi_err=float(z["xi_err"]),
                         R2=float(z["R2_mean"]), odd=float(z["odd_null_mean"]),
                         ess=float(z["ess_mean"])))
    return rows


def fit_power(zetas, xis, werr=None):
    """log xi = log A - p log zeta  -> slope -p."""
    m = np.isfinite(zetas) & np.isfinite(xis) & (xis > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    x = np.log(zetas[m]); y = np.log(xis[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    p = -coef[0]
    yhat = A @ coef
    ss = float(np.sum((y - yhat) ** 2)); st = float(np.sum((y - y.mean()) ** 2))
    R2 = 1 - ss / st if st > 0 else np.nan
    return float(p), float(R2)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = load(d)
    if not rows:
        print(f"no areaphase_*.npz in {d}"); return
    Ls = sorted(set(r["L"] for r in rows))
    offs = sorted(set(round(r["lam"] - r["lam_c"], 3) for r in rows))

    print("=" * 70)
    print("GATE 2: area-phase xi(zeta, lambda)  ->  phi=1/2 (p=0.5) vs phi=1 (p=1)")
    print("=" * 70)

    # odd-r null health
    odds = np.array([r["odd"] for r in rows if np.isfinite(r["odd"])])
    print(f"\nodd-r null (should be ~0 if decoupling holds): "
          f"median={np.nanmedian(odds):.3f} max={np.nanmax(odds):.3f}")
    R2s = np.array([r["R2"] for r in rows if np.isfinite(r["R2"])])
    print(f"exponential-fit R2: median={np.nanmedian(R2s):.3f} "
          f"(low R2 => not clean exponential; deepen area phase)")

    for L in Ls:
        print(f"\n--- L = {L} ---")
        # (1) lambda-flatness at fixed zeta
        print(" lambda-flatness  xi(lambda) at fixed zeta (flat => window law):")
        for z in sorted(set(r["zeta"] for r in rows if r["L"] == L)):
            sub = sorted([r for r in rows if r["L"] == L and r["zeta"] == z],
                         key=lambda r: r["lam"])
            xs = [f"{r['xi']:.2f}" for r in sub]
            print(f"   zeta={z:.2f}: xi over lambda = [{', '.join(xs)}]  "
                  f"(lam={['%.3f'%r['lam'] for r in sub]})")
        # (2) zeta-scaling at fixed offset
        print(" zeta-scaling  xi ~ zeta^{-p}:")
        for off in offs:
            sub = [r for r in rows if r["L"] == L
                   and abs((r["lam"] - r["lam_c"]) - off) < 1e-6]
            if len(sub) < 3:
                continue
            zz = np.array([r["zeta"] for r in sub])
            xx = np.array([r["xi"] for r in sub])
            p, R2 = fit_power(zz, xx)
            verdict = ("phi=1/2 (window law)" if abs(p - 0.5) < abs(p - 1.0)
                       else "phi=1 (coherent)")
            print(f"   offset={off:+.3f}: p={p:.3f} (R2={R2:.3f})  -> {verdict}")


if __name__ == "__main__":
    main()
