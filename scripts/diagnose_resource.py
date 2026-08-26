#!/usr/bin/env python3
"""Which resource actually buys information?  Zero new simulation.

Two diagnostics on data already on disk.

(A) n_real SCALING -- "would 48 or 100 trajectories fix this?"
Subsample the existing 12 realisations per cell down to n = 2,4,6,8,10,12 and
measure P(unique sign change), the CI width on lambda_c, and the mean number of
sign changes, as functions of n.  If P_unique climbs toward 1 and the width
follows n^-1/2, the estimator is statistics-limited and buying trajectories
works.  If it plateaus, more trajectories on the same estimator are wasted
compute.  This answers the question without running the trajectories.

The width is also fitted to n^-p.  p ~ 0.5 is ordinary Monte Carlo behaviour.
p ~ 0 means the error is dominated by something that averaging does not remove,
which points at finite-population bias rather than variance.

(B) GENEALOGY vs OBSERVABLE RESIDUAL -- is the sampler failure causing the
weaving?  Within each (L, lambda) cell, remove the cell mean and correlate the
residual magnitude |O_r - Obar| against that realisation's n_ancestors and
genealogical ESS.  Within cells, so the physical dependence on L and lambda is
removed.  If the most genealogically collapsed realisations produce the largest
excursions, the sampler failure and the curve weaving are linked.  If not,
genealogy may look terrible without driving the locator uncertainty.

DIAGNOSTIC ONLY.  Low-GESS realisations are NOT to be discarded on this basis;
that would change the estimator and bias the result.
"""
import os, sys, json, glob, argparse, itertools
import numpy as np


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


def cross_stats(A, B, lams, idx, rng, nsub, B_boot):
    """P(unique), mean #sign changes, and spread of the interpolated crossing."""
    uniq = 0
    nsc = []
    xs = []
    for _ in range(B_boot):
        d = []
        for k in range(len(idx)):
            sa = rng.choice(len(A[k]), nsub, replace=False)
            sb = rng.choice(len(B[k]), nsub, replace=False)
            d.append(B[k][sb].mean() - A[k][sa].mean())
        d = np.array(d)
        sc = np.where(np.diff(np.sign(d)) != 0)[0]
        nsc.append(len(sc))
        if len(sc) == 1:
            uniq += 1
        if len(sc):
            j = sc[len(sc) // 2]
            L0, L1 = lams[idx[j]], lams[idx[j + 1]]
            if d[j + 1] != d[j]:
                xs.append(L0 - d[j] * (L1 - L0) / (d[j + 1] - d[j]))
    w = (np.percentile(xs, 84) - np.percentile(xs, 16)) if len(xs) > 30 else np.nan
    return uniq / B_boot, float(np.mean(nsc)), w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--obs", default="CMI,B_L,c_eff")
    p.add_argument("--B", type=int, default=400)
    a = p.parse_args()
    rng = np.random.default_rng(20260826)
    rows = load(a.dir)
    if not rows:
        print("no records"); return
    Ls = sorted({int(r["L"]) for r in rows})
    lams = sorted({round(float(r["lambda"]), 6) for r in rows})
    zs = sorted({round(float(r["zeta"]), 4) for r in rows})
    nre = min(len({r["real"] for r in rows if int(r["L"]) == Lv
                   and np.isclose(float(r["lambda"]), lv)})
              for Lv in Ls for lv in lams)
    print("records %d   zeta %s   L %s   nlam %d   n_real per cell %d"
          % (len(rows), zs, Ls, len(lams), nre))

    print("\n" + "=" * 90)
    print("(A) n_real SCALING -- does buying trajectories fix the multiplicity?")
    print("=" * 90)
    ns = [n for n in (2, 4, 6, 8, 10, 12) if n <= nre]
    for obs in a.obs.split(","):
        key = obs + "_mean"
        if key not in rows[0]:
            continue
        V = {}
        for Lv in Ls:
            for lv in lams:
                v = np.array([r[key] for r in rows if int(r["L"]) == Lv
                              and np.isclose(float(r["lambda"]), lv)], float)
                if len(v) >= 3:
                    V[(Lv, lv)] = v
        print("\n--- %s ---" % obs)
        print(f"  {'pair':>10}" + "".join(f"{'n=%d' % n:>22}" for n in ns))
        print(f"  {'':>10}" + "".join(f"{'P_uniq  width':>22}" for n in ns))
        for L1, L2 in itertools.combinations(Ls, 2):
            idx = [i for i, lv in enumerate(lams) if (L1, lv) in V and (L2, lv) in V]
            if len(idx) < 5:
                continue
            A = [V[(L1, lams[i])] for i in idx]
            Bv = [V[(L2, lams[i])] for i in idx]
            line = f"  {f'({L1},{L2})':>10}"
            widths = []
            for n in ns:
                pu, mn, w = cross_stats(A, Bv, lams, idx, rng, n, a.B)
                widths.append(w)
                line += f"{pu:>10.3f}{w:>12.4f}"
            print(line)
            ok = [(n, w) for n, w in zip(ns, widths) if np.isfinite(w)]
            if len(ok) >= 3:
                x = np.log([o[0] for o in ok]); y = np.log([o[1] for o in ok])
                pexp = -np.polyfit(x, y, 1)[0]
                print(f"  {'':>10}  width ~ n^-{pexp:.2f}   "
                      f"[0.5 = ordinary Monte Carlo; ~0 = not averagable]")

    print("\n" + "=" * 90)
    print("(B) GENEALOGY vs OBSERVABLE RESIDUAL, within cells")
    print("=" * 90)
    print("Spearman rank correlation of |O_r - Obar| against the realisation's")
    print("genealogy, pooled over cells after removing each cell mean.\n")

    def spearman(x, y):
        if len(x) < 10:
            return np.nan
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        d = np.sqrt((rx @ rx) * (ry @ ry))
        return float(rx @ ry / d) if d > 0 else np.nan

    print(f"  {'obs':<10}{'vs n_anc':>12}{'vs gen_ESS':>13}{'n':>8}")
    for obs in a.obs.split(","):
        key = obs + "_mean"
        if key not in rows[0]:
            continue
        res, anc, ges = [], [], []
        for Lv in Ls:
            for lv in lams:
                cell = [r for r in rows if int(r["L"]) == Lv
                        and np.isclose(float(r["lambda"]), lv)]
                if len(cell) < 3:
                    continue
                v = np.array([r[key] for r in cell], float)
                m = v.mean()
                for r, vv in zip(cell, v):
                    res.append(abs(vv - m))
                    anc.append(float(r.get("n_distinct_ancestors", np.nan)))
                    ges.append(float(r.get("genealogical_ess", np.nan)))
        res, anc, ges = map(np.asarray, (res, anc, ges))
        ok = np.isfinite(res) & np.isfinite(anc)
        ok2 = np.isfinite(res) & np.isfinite(ges)
        print(f"  {obs:<10}{spearman(res[ok], anc[ok]):>12.3f}"
              f"{spearman(res[ok2], ges[ok2]):>13.3f}{int(ok.sum()):>8}")
    print("\n  Strong NEGATIVE correlation = the collapsed realisations throw the")
    print("  biggest excursions, linking sampler failure to curve weaving.")
    print("  Near zero = genealogy looks terrible without driving the uncertainty.")
    print("  DIAGNOSTIC ONLY -- do not filter realisations on this.")


if __name__ == "__main__":
    main()
