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
import numpy as np, pandas as pd
from scipy.optimize import least_squares
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
    return pd.DataFrame(rows)


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


def zero_cross(lam, y, rng, boot_vals=None):
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
    if df.empty:
        print("no records"); return
    print("records: %d   zetas: %s   L: %s"
          % (len(df), sorted(df.zeta.unique()), sorted(df.L.unique())))

    for z in sorted(df.zeta.unique()):
        s = df[np.isclose(df.zeta, z)]
        Ls = sorted(s.L.unique()); lams = sorted(np.round(s["lambda"].unique(), 6))
        print("\n" + "=" * 96)
        print("zeta = %.2f   L = %s   nlam = %d" % (z, Ls, len(lams)))
        print("=" * 96)

        cost_h = s.wall_traj_s.sum() / 3600.0
        print("core-hours in this slice: %.2f\n" % cost_h)

        print("TRACK 1 -- cross-L locators, scored by the L-SCRAMBLE ratio")
        print(f"  {'obs':<10}{'lam_c':>9}{'68% CI':>19}{'width':>9}{'nu*':>7}"
              f"{'cost':>10}{'Lscram':>10}{'ratio':>8}")
        for obs in CROSS_L:
            col = obs + "_mean"
            if col not in s.columns:
                continue
            per = {}
            for Lv in Ls:
                for lv in lams:
                    v = s[(s.L == Lv) & (np.isclose(s["lambda"], lv))][col].to_numpy()
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

        print("\nTRACK 2 -- c_eff, zero crossing from a SINGLE L (no cross-L difference)")
        print(f"  {'L':>6}{'lam_c':>9}{'68% CI':>19}{'width':>9}{'c at lam_lo':>13}"
              f"{'c at lam_hi':>13}")
        singles = []
        for Lv in Ls:
            g = s[s.L == Lv]
            vals = [g[np.isclose(g["lambda"], lv)]["c_eff_mean"].to_numpy() for lv in lams]
            if any(len(v) < 3 for v in vals):
                continue
            lamA = np.array(lams)
            m = np.array([v.mean() for v in vals])
            lc = zero_cross(lamA, m)
            bs = []
            for _ in range(a.B):
                mb = np.array([v[rng.integers(0, len(v), len(v))].mean() for v in vals])
                x = zero_cross(lamA, mb)
                if np.isfinite(x):
                    bs.append(x)
            bs = np.array(bs)
            q16, q84 = (np.percentile(bs, [16, 84]) if len(bs) > 30 else (np.nan, np.nan))
            print(f"  {Lv:>6}{lc:>9.4f}{f'[{q16:.4f},{q84:.4f}]':>19}{q84-q16:>9.4f}"
                  f"{m[0]:>13.4f}{m[-1]:>13.4f}")
            singles.append((Lv, lc, q84 - q16))
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
