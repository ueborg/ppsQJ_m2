#!/usr/bin/env python
"""
fit_nu_zeta1.py -- extract nu (and lambda_c, omega) at the Born corner zeta=1
from the worker_opdim_pps npz outputs, via a Slevin-Ohtsuki cost-function FSS of
the B_L crossing observable with one irrelevant correction L^{-omega}.
Cross-checks: pairwise B_L crossings, and the entanglement central charge c_ent
from S_half(L) at lambda_c.

WHY zeta=1 only: it is the single clean Born fixed point (no cloning, no ESS
collapse, trajectories independent so large L is reachable). nu is a single
well-defined number there, unlike along the PPS line. This measures it directly,
validating/replacing the imported Jian nu~2.1 that y_lambda=1/2 rests on.

Usage:
    python analysis/fit_nu_zeta1.py /scratch/$USER/pps_qj/pps_nu_zeta1 [--out OUT]
    python analysis/fit_nu_zeta1.py --selftest
"""
import argparse, glob, os, sys
import numpy as np
from scipy.optimize import least_squares


def load(d):
    recs = {}
    for f in sorted(glob.glob(os.path.join(d, "opdim_*.npz"))):
        z = np.load(f, allow_pickle=True)
        L = int(z["L"]); lam = float(z["lam"])
        bm = float(z["B_L_mean"]); be = float(z["B_L_err"])
        S = float(z["S_mean"]) if "S_mean" in z.files else np.nan
        recs.setdefault(L, []).append((lam, bm, be, S))
    for L in recs:
        recs[L] = sorted(recs[L])
    return recs


def _flat(recs):
    lam, L, y, e = [], [], [], []
    for LL, pts in recs.items():
        for (lm, bm, be, _S) in pts:
            if np.isfinite(bm):
                lam.append(lm); L.append(float(LL)); y.append(bm)
                e.append(max(be, 1e-3))
    return (np.array(lam), np.array(L), np.array(y), np.array(e))


def _model(p, lam, L, nord, mord):
    lc, nu, om = p[0], p[1], p[2]
    a = p[3:3 + nord + 1]
    b = p[3 + nord + 1:]
    x = (lam - lc) * L ** (1.0 / nu)
    F = sum(a[n] * x ** n for n in range(nord + 1))
    G = sum(b[m] * x ** m for m in range(mord + 1))
    return F + L ** (-om) * G


def fit_so(lam, L, y, e, nord=3, mord=1, nu0=2.0, lc0=0.5, om0=1.0):
    p0 = [lc0, nu0, om0] + [float(np.mean(y))] + [0.0] * nord + [0.0] * (mord + 1)
    lo = [0.30, 0.30, 0.1] + [-np.inf] * (nord + 1 + mord + 1)
    hi = [0.70, 6.0, 4.0] + [np.inf] * (nord + 1 + mord + 1)
    res = least_squares(lambda p: (_model(p, lam, L, nord, mord) - y) / e,
                        p0, bounds=(lo, hi), max_nfev=40000)
    chi2 = float(np.sum(res.fun ** 2)); dof = max(len(y) - len(res.x), 1)
    return res.x, chi2, dof


def bootstrap(lam, L, y, e, nord, mord, nboot=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nboot):
        yb = y + rng.normal(0, e)
        try:
            p, _, _ = fit_so(lam, L, yb, e, nord, mord)
            out.append([p[0], p[1], p[2]])
        except Exception:
            pass
    out = np.array(out)
    return out  # columns lc, nu, omega


def pairwise_crossings(recs):
    Ls = sorted(recs)
    rows = []
    for i in range(len(Ls)):
        for j in range(i + 1, len(Ls)):
            La, Lb = Ls[i], Ls[j]
            a = np.array([(p[0], p[1]) for p in recs[La]])
            b = np.array([(p[0], p[1]) for p in recs[Lb]])
            lo, hi = max(a[:, 0].min(), b[:, 0].min()), min(a[:, 0].max(), b[:, 0].max())
            if hi <= lo:
                continue
            g = np.linspace(lo, hi, 600)
            d = np.interp(g, a[:, 0], a[:, 1]) - np.interp(g, b[:, 0], b[:, 1])
            s = np.where(np.diff(np.sign(d)) != 0)[0]
            if len(s):
                k = s[0]
                x0, x1, y0, y1 = g[k], g[k + 1], d[k], d[k + 1]
                rows.append((La, Lb, float(x0 - y0 * (x1 - x0) / (y1 - y0))))
    return rows


def c_ent(recs, lc):
    """S_half(L) ~ (c/6) ln L at the lambda closest to lc."""
    Ls, S = [], []
    for LL, pts in recs.items():
        arr = np.array([(p[0], p[3]) for p in pts])
        k = int(np.argmin(np.abs(arr[:, 0] - lc)))
        if np.isfinite(arr[k, 1]):
            Ls.append(LL); S.append(arr[k, 1])
    if len(Ls) < 3:
        return None
    Ls = np.array(Ls, float); S = np.array(S)
    A = np.vstack([np.log(Ls), np.ones_like(Ls)]).T
    slope, _ = np.linalg.lstsq(A, S, rcond=None)[0]
    return 6.0 * slope


def report(recs, nord=3, mord=1, nboot=200):
    lam, L, y, e = _flat(recs)
    Ls = sorted(recs)
    print(f"loaded L={Ls}, {len(y)} (L,lam) points, lam in "
          f"[{lam.min():.3f},{lam.max():.3f}]")
    if len(Ls) < 3:
        print("need >=3 L for FSS"); return
    p, chi2, dof = fit_so(lam, L, y, e, nord, mord)
    bo = bootstrap(lam, L, y, e, nord, mord, nboot=nboot)
    lc_e, nu_e, om_e = (np.std(bo[:, 0]), np.std(bo[:, 1]), np.std(bo[:, 2])) \
        if len(bo) > 10 else (np.nan, np.nan, np.nan)
    print("\n=== Slevin-Ohtsuki cost-function FSS (B_L, zeta=1) ===")
    print(f"  poly order F={nord}, correction G order={mord}, L^-omega term")
    print(f"  lambda_c = {p[0]:.4f} +- {lc_e:.4f}")
    print(f"  nu       = {p[1]:.3f} +- {nu_e:.3f}   (Jian Born nu ~ 2.1; y_lambda=1/2 => nu=2)")
    print(f"  omega    = {p[2]:.2f} +- {om_e:.2f}")
    print(f"  chi2/dof = {chi2/dof:.2f}  ({dof} dof)")
    xc = pairwise_crossings(recs)
    if xc:
        print("\n  pairwise B_L crossings (lambda_c drift check):")
        for (a, b, lcx) in xc:
            print(f"    ({a:>3},{b:>3}): {lcx:.4f}")
    c = c_ent(recs, p[0])
    if c is not None:
        print(f"\n  c_ent from S_half(L) ~ (c/6)lnL at lambda_c: c = {c:.3f}")
    return p, (lc_e, nu_e, om_e), chi2 / dof


def selftest():
    rng = np.random.default_rng(1)
    nu_true, lc_true, om_true = 2.0, 0.50, 1.2
    Ls = [64, 96, 128, 160, 192]
    lams = np.array([0.44, 0.46, 0.48, 0.49, 0.50, 0.51, 0.52, 0.54, 0.56])
    recs = {}
    for L in Ls:
        pts = []
        for lm in lams:
            x = (lm - lc_true) * L ** (1.0 / nu_true)
            B = 1.2 - 0.8 * x + 0.15 * x ** 2 + (0.5) * L ** (-om_true)  # F + L^-om G
            be = 0.02
            pts.append((lm, B + rng.normal(0, be), be,
                        (1.0 / 6) * 6 * np.log(L) + rng.normal(0, 0.02)))  # c=1 synthetic
        recs[L] = sorted(pts)
    print("[selftest] truth: nu=2.0, lambda_c=0.50, omega=1.2, c_ent=1.0")
    report(recs, nboot=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default=None)
    ap.add_argument("--nord", type=int, default=3)
    ap.add_argument("--mord", type=int, default=1)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.data_dir:
        ap.error("data_dir required (or --selftest)")
    recs = load(a.data_dir)
    report(recs, a.nord, a.mord, a.nboot)


if __name__ == "__main__":
    main()
