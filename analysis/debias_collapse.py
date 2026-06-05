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


def per_L_correction(a_paths, b_paths, bl_lo=0.3, bl_hi=12.0):
    """Learn a per-L bias factor from matched points, return it plus the FULL
    dataset-A points (all lambda) so crossings keep their resolution.

    factor(L) = median over crossing-relevant matched points of B_inf / B_hi,
    where B_hi is A's (production-N_c) value. Multiply A's B_L by factor(L) to
    debias. Scatter (MAD) is reported so you can judge if the per-L (lambda-
    independent) model is even valid.
    """
    A, B = _index(_load_many(a_paths)), _index(_load_many(b_paths))
    # full raw from ALL of A (every lambda), keyed without T
    full = {}
    for (L, lam, z, T), r in A.items():
        b = r.get("B_L_mean"); e = r.get("B_L_err", np.nan)
        if b is not None and np.isfinite(b) and b > 0:
            full[(int(L), round(lam, 4), round(z, 3))] = (
                float(b), float(e) if np.isfinite(e) else 0.05 * b)
    # per-L correction factor from matched crossing-relevant points
    ratios = defaultdict(list)
    for k in set(A) & set(B):
        ra, rb = A[k], B[k]
        na, nb = int(ra["N_c"]), int(rb["N_c"])
        if na == nb:
            continue
        ba, bb = ra.get("B_L_mean"), rb.get("B_L_mean")
        if ba is None or bb is None or not (np.isfinite(ba) and np.isfinite(bb)):
            continue
        n_hi, b_hi, n_lo, b_lo = (na, ba, nb, bb) if na > nb else (nb, bb, na, ba)
        if not (bl_lo <= abs(b_hi) <= bl_hi) or b_hi == 0:
            continue
        a = n_lo / (n_hi - n_lo)
        b_inf = (1 + a) * b_hi - a * b_lo
        if np.isfinite(b_inf):
            ratios[int(k[0])].append(b_inf / b_hi)
    corr = {}
    for L, rs in ratios.items():
        rs = np.array(rs)
        med = float(np.median(rs))
        corr[L] = (med, float(np.median(np.abs(rs - med))), len(rs))
    return full, corr


def apply_correction(full, corr):
    out = {}
    for (L, lam, z), (b, e) in full.items():
        f = corr.get(L, (1.0, 0.0, 0))[0]
        out[(L, lam, z)] = (b * f, e * abs(f))
    return out


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

    full_raw, corr = per_L_correction(args.a, args.b)
    print(f"full dataset-A points (all lambda): {len(full_raw)}")
    print("\nPer-L bias correction factor (B_inf/B_hi), from matched points:")
    print(f"{'L':>4} {'factor':>8} {'scatter(MAD)':>13} {'n_pairs':>8}  note")
    for L in sorted(corr):
        f, mad, n = corr[L]
        note = "ok" if (n >= 6 and mad < 0.10) else ("few pairs" if n < 6 else "noisy")
        print(f"{L:>4} {f:>8.3f} {mad:>13.3f} {n:>8}  {note}")
    print("  (factor<1 => raw overestimates B_L; large MAD => per-L model unreliable)")

    corrected = apply_correction(full_raw, corr)
    bz_raw, bz_cor = _by_zL(full_raw), _by_zL(corrected)
    zetas = sorted(bz_raw)

    print("\n" + "=" * 70)
    print(f"lambda_c(zeta) from Binder crossings on FULL data, Ls={Ls}")
    print("=" * 70)
    print(f"{'zeta':>6} {'lc_RAW':>16} {'lc_CORRECTED':>18} {'shift':>9}")
    z_ok, lc_raw_ok, e_raw_ok, lc_cor_ok, e_cor_ok = [], [], [], [], []
    for z in zetas:
        lcr, er, _ = lambda_c_inf(bz_raw, z, Ls)
        lcc, ec, _ = lambda_c_inf(bz_cor, z, Ls)
        if lcr is None or lcc is None:
            continue
        z_ok.append(z)
        lc_raw_ok.append(lcr); e_raw_ok.append(er or 0.02)
        lc_cor_ok.append(lcc); e_cor_ok.append(ec or 0.02)
        print(f"{z:>6.3f} {lcr:>8.4f}+-{er or 0:<6.4f} {lcc:>9.4f}+-{ec or 0:<6.4f} "
              f"{lcc-lcr:>+9.4f}")

    A_raw = A_cor = float("nan")
    if len(z_ok) >= 2:
        A_raw, chi_raw = fit_A(z_ok, lc_raw_ok, e_raw_ok)
        A_cor, chi_cor = fit_A(z_ok, lc_cor_ok, e_cor_ok)
        print("\n" + "-" * 70)
        print("lambda_c = A*sqrt(zeta) fit:")
        print(f"  RAW       : A = {A_raw:.3f}   (chi2/dof = {chi_raw:.2f})")
        print(f"  CORRECTED : A = {A_cor:.3f}   (chi2/dof = {chi_cor:.2f})")
        print(f"  => correction shifts A by {A_cor - A_raw:+.3f} "
              f"({100*(A_cor-A_raw)/A_raw:+.0f}%)")
        print("\nCaveats: L=48,96 carry OBC Friedel; L=128 factor rests on 2 pairs")
        print("(excluded unless in --Ls). Trust the fit only where chi2/dof ~ O(1)")
        print("and the per-L factors above are 'ok'. Small-zeta L=128 still needs the")
        print("higher-N_c run -- this corrects mid/large zeta from existing data.")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if z_ok:
        sq = np.sqrt(np.array(z_ok))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(sq, lc_raw_ok, yerr=e_raw_ok, fmt="o", color="tab:gray",
                    capsize=3, label="raw")
        ax.errorbar(sq, lc_cor_ok, yerr=e_cor_ok, fmt="s", color="tab:blue",
                    capsize=3, label="per-L bias-corrected")
        xx = np.linspace(0, max(sq) * 1.05, 50)
        if np.isfinite(A_raw):
            ax.plot(xx, A_raw * xx, "--", color="tab:gray", lw=1, label=f"A={A_raw:.2f}")
            ax.plot(xx, A_cor * xx, "-", color="tab:blue", lw=1.5, label=f"A={A_cor:.2f}")
        ax.set_xlabel(r"$\sqrt{\zeta}$"); ax.set_ylabel(r"$\lambda_c(\infty)$")
        ax.set_title("Raw vs per-L bias-corrected critical line"); ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "debias_collapse.png", dpi=130)
        print(f"\nplot -> {out/'debias_collapse.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
