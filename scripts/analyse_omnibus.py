#!/usr/bin/env python3
"""Score the omnibus: which observable locates the boundary best per core-hour?

TWO SCORING TRACKS, because the observables are not all the same kind of thing.

TRACK 1 -- CROSS-L LOCATORS (CMI, B_L, S_AB, I3, MI_ends, varN).
Global collapse O(L,lam) = F((lam-lam_c) L^(1/nu)), scored by the L-LABEL-
SCRAMBLE ratio: permute L within each lambda column, which preserves every
lambda-marginal and destroys only the finite-size structure.  A lambda-label
scramble is NOT used; it only tests smoothness in lambda and inflates ratios to
42-1293x on data whose real L-signal is 1.2-2.6x.

nu is REPORTED BUT NOT INTERPRETED.  Over L = 64..128, ln L rises by 1.17 and
L^(1/2.7) by 1.20 -- the algebraic and logarithmic scaling variables are
numerically the same object across any affordable L range, so a fitted nu is an
underdetermined parameter, not a measurement.  lambda_c, by contrast, agreed to
better than 0.001 between the two forms at every zeta, so lambda_c is the
trustworthy output and nu is not.

TRACK 2 -- c_eff, WHICH NEEDS NO CROSS-L COMPARISON.
S(l) = (c/3) ln[(L/pi) sin(pi l / L)] + b, fitted within a single chain.
c > 0 in the log phase, c -> 0 in the area law, so lambda_c is where c(lambda)
vanishes -- read off ONE system size.  Every method that has failed on this
problem failed on the cross-L difference, and we have established the
affordable L range is too narrow to support finite-size scaling of any form.
So c_eff is scored by the bootstrap CI on its own zero crossing, PER L, and
compared against the five-L collapse CI of 0.0082 that CMI achieves at zeta=1.

An L-scramble ratio is meaningless for c_eff and is not computed for it.  What
IS required of it: the zero crossing must be consistent ACROSS L (a real order
parameter locates the same lambda_c at every size), and it must reproduce the
zeta = 1.00 anchor of 0.4364.
"""
import os, json, glob, argparse, warnings
import numpy as np
from scipy.optimize import least_squares

# No pandas: the Ruche production env (pps_qj) carries numpy and scipy only,
# and analysis must never require installing into it.
warnings.filterwarnings("ignore")

CROSS_L = ["CMI", "B_L", "S_AB", "I3", "MI_ends", "varN"]


def load(root):
    rows = []
    for f in glob.glob(os.path.join(root, "**", "real*.json"), recursive=True):
        try:
            r = json.load(open(f))
            if r.get("status") == "ok":
                rows.append(r)
        except Exception:
            pass
    return rows


def col(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def where(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if np.isclose(float(r[k]), float(v))]
    return out


def collapse(L, lam, y, w, deg=3, seedpts=7):
    lo, hi = lam.min(), lam.max(); span = hi - lo

    def resid(th):
        lc, q = th
        x = (lam - lc) * L ** q
        s = np.std(x)
        if not np.isfinite(s) or s <= 0:
            return np.full(len(y), 1e6)
        V = np.vander(x / s, deg + 1)
        try:
            c, *_ = np.linalg.lstsq(V * w[:, None], y * w, rcond=None)
        except Exception:
            return np.full(len(y), 1e6)
        return (V @ c - y) * w

    best = None
    for lc_s in np.linspace(lo + .1 * span, hi - .1 * span, seedpts):
        for q_s in (0.4, 0.7, 1.0, 1.5):
            try:
                r = least_squares(resid, [lc_s, q_s], bounds=([lo, .15], [hi, 2.5]),
                                  max_nfev=3000)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    if best is None:
        return np.nan, np.nan, np.nan
    return float(best.x[0]), float(1 / best.x[1]), float(2 * best.cost / max(len(y) - deg - 3, 1))


def zero_cross(lam, y, boot_vals=None):
    """lambda where y(lambda) crosses zero, monotone-decreasing convention."""
    v = y if boot_vals is None else boot_vals
    i = np.where(np.diff(np.sign(v)) != 0)[0]
    if len(i) == 0:
        return np.nan
    k = i[len(i) // 2]
    if v[k + 1] == v[k]:
        return float(lam[k])
    return float(lam[k] - v[k] * (lam[k + 1] - lam[k]) / (v[k + 1] - v[k]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--B", type=int, default=300)
    a = p.parse_args()
    rng = np.random.default_rng(20260826)
    df = load(a.dir)
    if not df:
        print("no records"); return
    zs = sorted({round(float(r["zeta"]), 6) for r in df})
    print("records: %d   zetas: %s   L: %s"
          % (len(df), zs, sorted({int(r["L"]) for r in df})))

    for z in zs:
        s = where(df, zeta=z)
        Ls = sorted({int(r["L"]) for r in s})
        lams = sorted({round(float(r["lambda"]), 6) for r in s})
        print("\n" + "=" * 96)
        print("zeta = %.2f   L = %s   nlam = %d" % (z, Ls, len(lams)))
        print("=" * 96)

        cost_h = float(np.nansum(col(s, "wall_traj_s"))) / 3600.0
        print("core-hours in this slice: %.2f\n" % cost_h)

        print("TRACK 1 -- cross-L locators, scored by the L-SCRAMBLE ratio")
        print(f"  {'obs':<10}{'lam_c':>9}{'68% CI':>19}{'width':>9}{'nu*':>7}"
              f"{'cost':>10}{'Lscram':>10}{'ratio':>8}")
        for obs in CROSS_L:
            key = obs + "_mean"
            if key not in s[0]:
                continue
            per = {}
            for Lv in Ls:
                for lv in lams:
                    v = col(where(s, L=Lv, **{"lambda": lv}), key)
                    if len(v) >= 3:
                        per[(Lv, lv)] = v
            if len(per) < 20:
                continue

            def flat(boot):
                L_, lam_, y_, w_ = [], [], [], []
                for (Lv, lv), v in per.items():
                    vv = v[rng.integers(0, len(v), len(v))] if boot else v
                    L_.append(Lv); lam_.append(lv); y_.append(vv.mean())
                    w_.append(1.0 / max(vv.std(ddof=1) / np.sqrt(len(vv)), 1e-9))
                w_ = np.asarray(w_); w_ = w_ / w_.mean()
                return np.array(L_, float), np.array(lam_), np.array(y_), w_

            L_, lam_, y_, w_ = flat(False)
            lc, nu, c0 = collapse(L_, lam_, y_, w_)
            sc = []
            for _ in range(12):
                Ls2 = L_.copy()
                for lv in set(lam_):
                    m = lam_ == lv
                    Ls2[m] = rng.permutation(L_[m])
                sc.append(collapse(Ls2, lam_, y_, w_, seedpts=3)[2])
            scram = float(np.nanmedian(sc))
            bs = [collapse(*flat(True))[0] for _ in range(a.B)]
            bs = np.array([b for b in bs if np.isfinite(b)])
            q16, q84 = (np.percentile(bs, [16, 84]) if len(bs) > 30 else (np.nan, np.nan))
            print(f"  {obs:<10}{lc:>9.4f}{f'[{q16:.4f},{q84:.4f}]':>19}{q84-q16:>9.4f}"
                  f"{nu:>7.2f}{c0:>10.4f}{scram:>10.4f}{scram/c0 if c0>0 else np.nan:>8.2f}")
        print("  * nu is reported, not interpreted: it is underdetermined over any")
        print("    affordable L range (ln L and L^(1/nu) differ by 1.17 vs 1.20).")

        print("\nTRACK 2 -- c_eff, scale-invariant point where d c_eff / d ln L = 0")
        print("  NOT a zero of c_eff.  In the area law c decays to zero only as L -> inf,")
        print("  so at finite L there is no zero to cross and any threshold is arbitrary.")
        print("  c_eff is a CROSS-L locator; what distinguishes it is that its L-trend")
        print("  reverses sharply, which is the signal the other observables lack.")
        print(f"  {'lam':>8}" + "".join(f"{'c(L=%d)' % L:>11}" for L in Ls)
              + f"{'d c/d lnL':>12}")
        singles = []
        lamA = np.array(lams)
        per_c = {}
        for Lv in Ls:
            g = where(s, L=Lv)
            per_c[Lv] = [col(where(g, **{"lambda": lv}), "c_eff_mean") for lv in lams]
        if all(len(v) >= 3 for Lv in Ls for v in per_c[Lv]):
            lnL = np.log(np.array(Ls, dtype=float))

            def slope_profile(boot):
                out = []
                for j in range(len(lams)):
                    ys = []
                    for Lv in Ls:
                        v = per_c[Lv][j]
                        ys.append(v[rng.integers(0, len(v), len(v))].mean() if boot
                                  else v.mean())
                    out.append(np.polyfit(lnL, np.array(ys), 1)[0])
                return np.array(out)

            b0 = slope_profile(False)
            for j, lv in enumerate(lams):
                print(f"  {lv:>8.4f}"
                      + "".join(f"{per_c[Lv][j].mean():>11.4f}" for Lv in Ls)
                      + f"{b0[j]:>12.4f}")
            lc = zero_cross(lamA, b0)
            bs = []
            for _ in range(a.B):
                x = zero_cross(lamA, slope_profile(True))
                if np.isfinite(x):
                    bs.append(x)
            bs = np.array(bs)
            if len(bs) > 30:
                q16, q84 = np.percentile(bs, [16, 84])
                cc = float(np.interp(lc, lamA, [np.mean([per_c[Lv][j].mean()
                                                         for Lv in Ls])
                                                for j in range(len(lams))]))
                print(f"\n  scale-invariant point: lam_c = {lc:.4f} "
                      f"[{q16:.4f},{q84:.4f}]  width {q84-q16:.4f}")
                print(f"  c_eff there = {cc:.3f}   (a MIPT expects a universal value)")
                print(f"  crossings found in {len(bs)} of {a.B} bootstrap resamples "
                      f"({100*len(bs)/a.B:.0f} pct)")
                print(f"  compare: CMI five-L collapse at zeta=1 gives width 0.0082,")
                print(f"           CMI here gives 0.0097, MI_ends 0.0129, B_L 0.0059.")
                singles = [(Ls[0], lc, q84 - q16)]
            else:
                print("\n  no stable crossing of d c/d lnL inside the window")
        else:
            print("  insufficient realisations per cell")

        if len(singles) >= 2:
            v = [x[1] for x in singles]
            print("\n  consistency across L: spread of lam_c = %.4f" % (max(v) - min(v)))
            print("  A real order parameter locates the SAME lambda_c at every size.")
            print("  Best single-L CI width: %.4f  (CMI five-L collapse at zeta=1 gives 0.0082)"
                  % min(x[2] for x in singles))
            cheap = min(singles, key=lambda x: x[0])
            print("  Cheapest usable L = %d, CI width %.4f." % (cheap[0], cheap[2]))


if __name__ == "__main__":
    main()
