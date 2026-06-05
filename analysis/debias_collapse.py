#!/usr/bin/env python
"""Bias-corrected FSS: build a debiased B_L aggregate by 1/N_c-extrapolating
matched points across two datasets, then run YOUR Binder-crossing analysis on
both raw and debiased data and compare lambda_c(zeta).

Matched (L,lam,zeta) points that share T but differ in N_c give a 2-point
1/N_c ladder -> extrapolated B_inf (with propagated error). We then run the
same pairwise-crossing + 1/sqrt(L) extrapolation + lambda_c=A*sqrt(zeta) fit
on (a) the higher-N_c raw values and (b) the debiased values, side by side.

Reuses the loaders/extrapolation from nc_bias_pairs.

Usage:
    python analysis/debias_collapse.py \
        --a /scratch/$USER/pps_qj/pps_clone_dense /scratch/$USER/pps_qj/pps_clone_rescue \
        --b /scratch/$USER/pps_qj/clone_aggregate2.pkl \
        --Ls 32,48,64,96 --out outputs/diagnostics/debias_collapse
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nc_bias_pairs import _load_many, _index  # noqa: E402


def build_raw_debiased(a_paths, b_paths):
    """Return raw{(L,l,z):(B_hi,err_hi)} and deb{(L,l,z):(B_inf,err_inf)} for
    points present in both datasets at equal T, different N_c."""
    A, B = _index(_load_many(a_paths)), _index(_load_many(b_paths))
    raw, deb = {}, {}
    for k in set(A) & set(B):
        ra, rb = A[k], B[k]
        na, nb = int(ra["N_c"]), int(rb["N_c"])
        if na == nb:
            continue
        ba, bb = ra.get("B_L_mean"), rb.get("B_L_mean")
        ea, eb = ra.get("B_L_err", np.nan), rb.get("B_L_err", np.nan)
        if ba is None or bb is None or not (np.isfinite(ba) and np.isfinite(bb)):
            continue
        if na > nb:
            n_hi, b_hi, e_hi, n_lo, b_lo, e_lo = na, ba, ea, nb, bb, eb
        else:
            n_hi, b_hi, e_hi, n_lo, b_lo, e_lo = nb, bb, eb, na, ba, ea
        # b_inf = (1+a) b_hi - a b_lo,  a = n_lo/(n_hi-n_lo)
        a = n_lo / (n_hi - n_lo)
        b_inf = (1 + a) * b_hi - a * b_lo
        e_inf = float(np.sqrt(((1 + a) * (e_hi or 0)) ** 2 + (a * (e_lo or 0)) ** 2))
        L, lam, z, T = k
        raw[(L, lam, z)] = (float(b_hi), float(e_hi) if np.isfinite(e_hi) else 0.05 * abs(b_hi))
        deb[(L, lam, z)] = (float(b_inf), e_inf if np.isfinite(e_inf) else 0.1 * abs(b_inf))
    return raw, deb


def _by_zL(data):
    """{(L,l,z):(B,e)} -> {z: {L: (lams, bm, be)}} sorted, B>0."""
    out = defaultdict(lambda: defaultdict(list))
    for (L, lam, z), (b, e) in data.items():
        if b > 0:
            out[round(z, 3)][int(L)].append((lam, b, e))
    for z in out:
        for L in out[z]:
            out[z][L].sort(key=lambda t: t[0])
    return out


def _crossing(cz, L1, L2):
    """Pairwise Binder crossing of B_{L1}, B_{L2} at one zeta (same logic as
    plot_binder_proper.pairwise_crossing)."""
    p1, p2 = cz.get(L1), cz.get(L2)
    if not p1 or not p2 or len(p1) < 4 or len(p2) < 4:
        return None, None
    d1 = {l: (b, e) for l, b, e in p1}
    d2 = {l: (b, e) for l, b, e in p2}
    common = sorted(set(d1) & set(d2))
    if len(common) < 4:
        return None, None
    lams = np.array(common)
    diff = np.array([d2[l][0] - d1[l][0] for l in common])
    zcs = np.where(np.diff(np.sign(diff)))[0]
    if len(zcs) == 0:
        return None, None
    i = zcs[0]
    if abs(diff[i + 1] - diff[i]) < 1e-12:
        return None, None
    t = -diff[i] / (diff[i + 1] - diff[i])
    lc = float(lams[i] + t * (lams[i + 1] - lams[i]))
    e = [d1[common[i]][1], d2[common[i]][1], d1[common[i + 1]][1], d2[common[i + 1]][1]]
    err_diff = float(np.sqrt(sum(x ** 2 for x in e)))
    dlc = float(err_diff * (lams[i + 1] - lams[i]) / max(abs(diff[i + 1] - diff[i]), 1e-6))
    return lc, min(dlc, 0.05)


def lambda_c_inf(by_zL, z, Ls):
    """Pairwise crossings over consecutive Ls, extrapolate lc vs 1/sqrt(Lmin)."""
    cz = by_zL.get(z, {})
    pairs = [(Ls[i], Ls[i + 1]) for i in range(len(Ls) - 1)]
    xs, ys, es = [], [], []
    for L1, L2 in pairs:
        lc, e = _crossing(cz, L1, L2)
        if lc is not None:
            xs.append(1.0 / np.sqrt(L1)); ys.append(lc); es.append(e or 0.02)
    if len(ys) == 0:
        return None, None, []
    if len(ys) == 1:
        return ys[0], es[0], list(zip([p[0] for p in pairs], ys, es))
    w = 1.0 / np.array(es) ** 2
    p = np.polyfit(np.array(xs), np.array(ys), 1, w=w)
    lc_inf = float(np.polyval(p, 0.0))
    # crude error: spread of pair crossings
    lc_err = float(np.std(ys) / np.sqrt(len(ys)))
    return lc_inf, lc_err, list(zip([pp[0] for pp in pairs[:len(ys)]], ys, es))


def fit_A(zs, lcs, errs):
    """Weighted fit lambda_c = A*sqrt(zeta) through origin."""
    zs, lcs = np.asarray(zs), np.asarray(lcs)
    w = 1.0 / np.maximum(np.asarray(errs), 1e-3) ** 2
    sq = np.sqrt(zs)
    A = float(np.sum(w * sq * lcs) / np.sum(w * sq * sq))
    resid = lcs - A * sq
    chi2 = float(np.sum(w * resid ** 2) / max(len(zs) - 1, 1))
    return A, chi2


def main(argv=None):
    ap = argparse.ArgumentParser(description="bias-corrected vs raw Binder FSS")
    ap.add_argument("--a", required=True, nargs="+")
    ap.add_argument("--b", required=True, nargs="+")
    ap.add_argument("--Ls", type=str, default="32,48,64,96")
    ap.add_argument("--out", type=str, default="outputs/diagnostics/debias_collapse")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    Ls = [int(x) for x in args.Ls.split(",")]

    raw, deb = build_raw_debiased(args.a, args.b)
    print(f"matched points: {len(raw)}  (debiased the same set)")
    bz_raw, bz_deb = _by_zL(raw), _by_zL(deb)
    zetas = sorted(set(bz_raw) & set(bz_deb))

    print("\n" + "=" * 70)
    print(f"lambda_c(zeta) from Binder crossings, Ls={Ls}  [1/sqrt(L) extrap]")
    print("=" * 70)
    print(f"{'zeta':>6} {'lc_RAW':>16} {'lc_DEBIASED':>18} {'shift':>9}")
    z_ok, lc_raw_ok, e_raw_ok, lc_deb_ok, e_deb_ok = [], [], [], [], []
    for z in zetas:
        lcr, er, _ = lambda_c_inf(bz_raw, z, Ls)
        lcd, ed, _ = lambda_c_inf(bz_deb, z, Ls)
        if lcr is None or lcd is None:
            print(f"{z:>6.3f}   (insufficient crossings)")
            continue
        shift = lcd - lcr
        print(f"{z:>6.3f} {lcr:>8.4f}+-{er:<6.4f} {lcd:>9.4f}+-{ed:<6.4f} {shift:>+9.4f}")
        z_ok.append(z)
        lc_raw_ok.append(lcr); e_raw_ok.append(er)
        lc_deb_ok.append(lcd); e_deb_ok.append(ed)

    if len(z_ok) >= 2:
        A_raw, chi_raw = fit_A(z_ok, lc_raw_ok, e_raw_ok)
        A_deb, chi_deb = fit_A(z_ok, lc_deb_ok, e_deb_ok)
        print("\n" + "-" * 70)
        print(f"lambda_c = A*sqrt(zeta) fit:")
        print(f"  RAW      : A = {A_raw:.3f}   (chi2/dof = {chi_raw:.2f})")
        print(f"  DEBIASED : A = {A_deb:.3f}   (chi2/dof = {chi_deb:.2f})")
        print(f"  => debiasing shifts A by {A_deb - A_raw:+.3f} "
              f"({100*(A_deb-A_raw)/A_raw:+.0f}%)")
        print("\nNote: L=48,96 carry OBC Friedel oscillations; the clean triple is")
        print("32/64/128. A trustworthy large-L anchor still needs the higher-N_c")
        print("L=128 run -- this debiased fit is the best obtainable from current data.")
    else:
        A_raw = A_deb = float("nan")

    # plot
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if z_ok:
        sq = np.sqrt(np.array(z_ok))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(sq, lc_raw_ok, yerr=e_raw_ok, fmt="o", color="tab:gray",
                    capsize=3, label="raw (biased)")
        ax.errorbar(sq, lc_deb_ok, yerr=e_deb_ok, fmt="s", color="tab:blue",
                    capsize=3, label="debiased (1/N_c extrap)")
        xx = np.linspace(0, max(sq) * 1.05, 50)
        if np.isfinite(A_raw):
            ax.plot(xx, A_raw * xx, "--", color="tab:gray", lw=1)
            ax.plot(xx, A_deb * xx, "-", color="tab:blue", lw=1.5)
        ax.set_xlabel(r"$\sqrt{\zeta}$"); ax.set_ylabel(r"$\lambda_c(\infty)$")
        ax.set_title("Raw vs bias-corrected critical line"); ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "debias_collapse.png", dpi=130)
        print(f"\nplot -> {out/'debias_collapse.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
