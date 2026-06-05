#!/usr/bin/env python
"""Form-agnostic determination of the critical line lambda_c(zeta).

Pipeline (no critical-line form assumed anywhere):
  1. per-L bias correction (from debias_collapse) applied to full data
  2. for each zeta independently: FSS collapse with FREE (lambda_c, nu),
     minimizing B_L collapse scatter over the chosen L -- gives lambda_c(zeta)
     and nu(zeta) with a collapse-quality flag. Bad/unphysical zeta are dropped.
  3. fit the surviving lambda_c(zeta) to competing forms and let chi2 decide:
       - power law   lambda_c = A * zeta^phi   (phi FREE)
       - linear      lambda_c = m * zeta + b
       - small-zeta-only power (isolates the asymptotic exponent)
     plus a log-log plot whose SLOPE is phi. Straight => power law; curved =>
     crossover/saturation.

This answers "what scaling do I have", rather than confirming a chosen one.

Usage:
    python analysis/scaling_form.py \
        --a /scratch/$USER/pps_qj/pps_clone_dense /scratch/$USER/pps_qj/pps_clone_rescue \
        --b /scratch/$USER/pps_qj/clone_aggregate2.pkl \
        --Ls 16,24,32,64 --small-zeta-cut 0.3 --out outputs/diagnostics/scaling_form
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).resolve().parent))
from debias_collapse import per_L_correction, apply_correction, _by_zL  # noqa: E402


def _collapse_residual(params, curves, log_err=0.3):
    """Scatter of B_L curves under x=(lam-lc)*L^(1/nu); free (lc, nu)."""
    lc, nu = params
    if nu < 0.3 or nu > 6 or lc < 0.0 or lc > 0.98:
        return 1e9
    scaled = []
    for L, (lams, bm, be) in curves:
        x = (lams - lc) * L ** (1.0 / nu)
        o = np.argsort(x)
        scaled.append((x[o], bm[o]))
    if len(scaled) < 3:
        return 1e9
    tot, n = 0.0, 0
    for i in range(len(scaled)):
        for j in range(i + 1, len(scaled)):
            x1, y1 = scaled[i]; x2, y2 = scaled[j]
            lo, hi = max(x1.min(), x2.min()), min(x1.max(), x2.max())
            if hi <= lo:
                continue
            m = (x1 >= lo) & (x1 <= hi)
            if m.sum() < 3:
                continue
            f2 = interp1d(x2, y2, kind="linear", bounds_error=False,
                          fill_value="extrapolate")
            ly1 = np.log(np.maximum(y1[m], 1e-8))
            ly2 = np.log(np.maximum(f2(x1[m]), 1e-8))
            tot += np.sum(((ly1 - ly2) / log_err) ** 2); n += int(m.sum())
    return tot / max(n, 1)


def best_collapse_z(by_zL, z, Ls, min_lams=4):
    """Free-(lc,nu) collapse for one zeta over available Ls. Returns
    (lc, nu, quality, n_L) or None if too little data."""
    cz = by_zL.get(z, {})
    curves = []
    for L in Ls:
        pts = cz.get(L, [])
        if len(pts) >= min_lams:
            lams = np.array([p[0] for p in pts])
            bm = np.array([p[1] for p in pts])
            be = np.array([p[2] for p in pts])
            curves.append((L, (lams, bm, be)))
    if len(curves) < 3:
        return None
    best = (np.inf, None)
    for lc0 in np.linspace(0.02, 0.85, 30):
        for nu0 in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            q = _collapse_residual((lc0, nu0), curves)
            if q < best[0]:
                best = (q, (lc0, nu0))
    res = minimize(_collapse_residual, best[1], args=(curves,),
                   method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 800})
    lc, nu = res.x
    return float(lc), float(nu), float(res.fun), len(curves)


def fit_forms(zs, lcs, errs, small_cut):
    """Fit competing critical-line forms; return dict of results."""
    zs = np.asarray(zs); lcs = np.asarray(lcs); errs = np.asarray(errs)
    out = {}
    pos = lcs > 1e-4
    # power law (free phi) via log-log regression on positive points
    if pos.sum() >= 3:
        lr = linregress(np.log(zs[pos]), np.log(lcs[pos]))
        out["power"] = dict(phi=float(lr.slope), phi_err=float(lr.stderr),
                            A=float(np.exp(lr.intercept)), r2=float(lr.rvalue ** 2))
    # linear lambda_c = m zeta + b
    if len(zs) >= 3:
        w = 1.0 / np.maximum(errs, 1e-3) ** 2
        p, cov = np.polyfit(zs, lcs, 1, w=w, cov=True)
        pred = np.polyval(p, zs)
        chi2 = float(np.sum(w * (lcs - pred) ** 2) / max(len(zs) - 2, 1))
        out["linear"] = dict(m=float(p[0]), b=float(p[1]),
                             m_err=float(np.sqrt(cov[0, 0])), chi2=chi2)
    # small-zeta-only power (asymptotic exponent)
    sm = pos & (zs <= small_cut)
    if sm.sum() >= 3:
        lr = linregress(np.log(zs[sm]), np.log(lcs[sm]))
        out["power_smallz"] = dict(phi=float(lr.slope), phi_err=float(lr.stderr),
                                   A=float(np.exp(lr.intercept)),
                                   r2=float(lr.rvalue ** 2), n=int(sm.sum()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="form-agnostic lambda_c(zeta) scaling")
    ap.add_argument("--a", required=True, nargs="+")
    ap.add_argument("--b", required=True, nargs="+")
    ap.add_argument("--Ls", type=str, default="16,24,32,64")
    ap.add_argument("--qmax", type=float, default=5.0, help="max collapse residual to keep a zeta")
    ap.add_argument("--lc-max", type=float, default=0.95)
    ap.add_argument("--small-zeta-cut", type=float, default=0.3)
    ap.add_argument("--out", type=str, default="outputs/diagnostics/scaling_form")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    Ls = [int(x) for x in args.Ls.split(",")]

    full_raw, corr = per_L_correction(args.a, args.b)
    corrected = apply_correction(full_raw, corr)
    by_zL = _by_zL(corrected)
    zetas = sorted(by_zL)

    print("=" * 74)
    print(f"FORM-AGNOSTIC lambda_c(zeta)  (free nu per zeta), Ls={Ls}")
    print("=" * 74)
    print(f"{'zeta':>6} {'lambda_c':>9} {'nu':>6} {'quality':>9} {'n_L':>4}  status")
    zs, lcs, errs, nus = [], [], [], []
    for z in zetas:
        r = best_collapse_z(by_zL, z, Ls)
        if r is None:
            print(f"{z:>6.3f}      ---      ---       ---   --  too few L")
            continue
        lc, nu, q, nL = r
        ok = (q <= args.qmax) and (0.0 < lc < args.lc_max) and (0.3 < nu < 6)
        status = "keep" if ok else "DROP"
        print(f"{z:>6.3f} {lc:>9.4f} {nu:>6.2f} {q:>9.3f} {nL:>4}  {status}")
        if ok:
            zs.append(z); lcs.append(lc); nus.append(nu)
            errs.append(max(0.01, 0.15 * lc))  # crude; collapse gives no direct err

    if len(zs) < 3:
        print("\nToo few trustworthy zeta to fit a form. The collapse fails where")
        print("data is bias/Friedel/statistics-limited -- exactly the small-zeta")
        print("regime that distinguishes the forms. Need the higher-N_c runs.")
        return 0

    forms = fit_forms(zs, lcs, errs, args.small_zeta_cut)
    print("\n" + "-" * 74)
    print("CANDIDATE FORMS (let chi2/R^2 decide; phi=0.5 is sqrt, phi=1 is linear):")
    if "power" in forms:
        f = forms["power"]
        print(f"  power  lambda_c = {f['A']:.3f} * zeta^{f['phi']:.3f} "
              f"(+-{f['phi_err']:.3f})   R^2={f['r2']:.3f}")
    if "power_smallz" in forms:
        f = forms["power_smallz"]
        print(f"  power (zeta<={args.small_zeta_cut}, n={f['n']}): "
              f"phi = {f['phi']:.3f} +- {f['phi_err']:.3f}   R^2={f['r2']:.3f}  "
              f"<- the asymptotic exponent")
    if "linear" in forms:
        f = forms["linear"]
        print(f"  linear lambda_c = {f['m']:.3f}*zeta + {f['b']:.3f}  chi2/dof={f['chi2']:.2f}")

    print("\nINTERPRETATION:")
    nu_med = np.median(nus)
    print(f"  median nu = {nu_med:.2f} (free per zeta; expect ~2 if class-DIII multicritical)")
    if "power_smallz" in forms:
        phi = forms["power_smallz"]["phi"]; pe = forms["power_smallz"]["phi_err"]
        verdict = ("consistent with sqrt(zeta) (phi=1/2)" if abs(phi - 0.5) < 2 * pe + 0.1
                   else "consistent with linear (phi=1)" if abs(phi - 1.0) < 2 * pe + 0.1
                   else "NOT a clean 1/2 or 1 -- ambiguous from current data")
        print(f"  small-zeta exponent phi = {phi:.2f} +- {pe:.2f}  =>  {verdict}")
    print("  (phi pinned by the SMALL-zeta points; if those were DROPped above,")
    print("   the scaling type is undetermined and needs the higher-N_c small-zeta runs.)")

    # plots
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    zs_a, lcs_a = np.array(zs), np.array(lcs)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].errorbar(zs_a, lcs_a, yerr=errs, fmt="o", capsize=3, color="tab:blue", label="data")
    zz = np.linspace(min(zs_a) * 0.5, max(zs_a), 100)
    if "power" in forms:
        ax[0].plot(zz, forms["power"]["A"] * zz ** forms["power"]["phi"], "-",
                   color="tab:red", label=f"power phi={forms['power']['phi']:.2f}")
    if "linear" in forms:
        ax[0].plot(zz, forms["linear"]["m"] * zz + forms["linear"]["b"], "--",
                   color="tab:green", label="linear")
    ax[0].set_xlabel(r"$\zeta$"); ax[0].set_ylabel(r"$\lambda_c$")
    ax[0].set_title("critical line (free-nu collapse)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    # log-log
    pos = lcs_a > 0
    ax[1].loglog(zs_a[pos], lcs_a[pos], "o", color="tab:blue", label="data")
    if pos.sum() >= 2:
        for slope, c, lab in ((0.5, "tab:red", "slope 1/2"), (1.0, "tab:green", "slope 1")):
            anchor = lcs_a[pos][-1] / zs_a[pos][-1] ** slope
            ax[1].loglog(zz, anchor * zz ** slope, "--", color=c, alpha=0.6, label=lab)
    ax[1].set_xlabel(r"$\zeta$"); ax[1].set_ylabel(r"$\lambda_c$")
    ax[1].set_title("log-log: slope = exponent"); ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out / "scaling_form.png", dpi=130)
    print(f"\nplot -> {out/'scaling_form.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
