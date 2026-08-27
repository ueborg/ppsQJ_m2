#!/usr/bin/env python3
"""Score the adaptive-resampling pilot.

Reads all adaptive_L*_T*_task*.json files from $WORKDIR/pps/adaptive.
Every JSON is a list of per-realisation records, each with fields:
  zeta, lam, mode (always/ess0.9/ess0.5/never), real, status
  CMI, B_L, c_eff               weighted final-snapshot observables
  CMI_tavg50, CMI_tavg75        weighted tail averages (last 50%/25% of record)
  n_events                      resampling events fired
  gess_root, gess_recent        genealogical ESS at t=0 and t=T/2 lookback
  ess_final                     final population ESS fraction
  wall                          wall seconds

THREE QUESTIONS, in order of importance:

Q1. HORIZON. Does the tavg crossing location DRIFT between TMULT=1 and TMULT=2?
    If yes, T=L is not equilibrated and every existing number is biased.
    Test: compare CMI_tavg50 lambda_c at matched (zeta, L, mode=always) across Tmult.

Q2. BIAS-FREE. Is ess/never mean compatible with always at matched (zeta, lam)?
    If the means differ by more than ~2 sigma, the implementation is wrong or
    the weight variance is too large. Checked against always as reference.

Q3. VARIANCE. Does adaptive resampling reduce Var[CMI] at matched cost?
    Reported as variance ratio and an F-test (p < 0.05 is required, not just
    directional).  Wall-time ratio is reported separately per DEC-MASTER-METRIC-001:
    the accepted metric is t_wall * sigma^2(lambda_c), not variance alone.

Q4. NEVER-MODE ESS. Does essF(never) stay workable at L=48?
    If > ~0.15 at zeta <= 0.20, the low-zeta campaign can run interaction-free.

ACCEPTANCE for promoting the algorithm is NOT GESS (diagnostic_only per
DEC-MASTER-METRIC-001): it is bias-free (Q2 passes) AND variance reduction at
matched cost (Q3 F-test passes).  Any production claim then still needs
t_wall * sigma^2(lambda_c) at full L.
"""
import os, sys, json, glob, argparse
import numpy as np


def load(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "*.json"))):
        try:
            data = json.load(open(f))
            if isinstance(data, list):
                rows.extend(data)
            elif isinstance(data, dict):
                rows.append(data)
        except Exception as e:
            print("skip %s: %s" % (f, e))
    return [r for r in rows
            if r.get("status") == "ok"
            and "mode" in r and "L" in r]


def get(rows, **kw):
    out = rows
    for k, v in kw.items():
        if isinstance(v, float):
            out = [r for r in out if np.isclose(float(r.get(k, np.nan)), v, atol=1e-4)]
        else:
            out = [r for r in out if str(r.get(k)) == str(v)]
    return out


def col(rows, key):
    return np.array([r[key] for r in rows if key in r and np.isfinite(r[key])], float)


def crossing(vals_by_lam, lams, B, rng):
    """Bootstrap P(unique sign change). vals_by_lam: dict lam -> ndarray."""
    lams = sorted(lams)
    per = [np.asarray(vals_by_lam.get(l, np.array([])), float) for l in lams]
    ok = [i for i, v in enumerate(per) if len(v) >= 3]
    if len(ok) < 4:
        return np.nan, np.nan, np.nan
    cnt = {0: 0, 1: 0}
    xs = []
    for _ in range(B):
        d = np.array([per[i][rng.integers(0, len(per[i]), len(per[i]))].mean()
                      for i in ok])
        sc = np.where(np.diff(np.sign(d)) != 0)[0]
        n = len(sc)
        cnt[0 if n == 0 else (1 if n == 1 else 2)] = cnt.get(n, 0) + 1
        if n == 1:
            j = sc[0]
            il, ir = ok[j], ok[j + 1]
            if d[j + 1] != d[j]:
                xs.append(lams[il] - d[j] * (lams[ir] - lams[il]) / (d[j + 1] - d[j]))
    pu = cnt.get(1, 0) / B
    return pu, float(np.median(xs)) if xs else np.nan, float(np.std(xs)) if xs else np.nan


def f_test(a, b):
    """One-sided F-test: is Var[b] < Var[a]?  Returns (ratio, p-value)."""
    from scipy.stats import f as fdist
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    if va == 0:
        return np.nan, np.nan
    ratio = vb / va
    na, nb = len(a) - 1, len(b) - 1
    p = float(fdist.cdf(ratio, nb, na))  # P(F < ratio); small = b is better
    return ratio, p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--obs", default="CMI")
    p.add_argument("--B", type=int, default=600)
    p.add_argument("--ref", default="always", help="reference arm name")
    a = p.parse_args()
    rng = np.random.default_rng(20260827)
    rows = load(a.dir)
    if not rows:
        print("no records in", a.dir); return
    modes = sorted({r["mode"] for r in rows})
    zetas = sorted({round(float(r["zeta"]), 4) for r in rows})
    Ls = sorted({int(r["L"]) for r in rows})
    tmults = sorted({round(float(r.get("Tmult", r.get("tmult", 1.0))), 4) for r in rows})
    print("records: %d   modes: %s" % (len(rows), modes))
    print("zetas: %s   L: %s   Tmults: %s\n" % (zetas, Ls, tmults))

    # ------------------------------------------------------------------ Q1
    print("=" * 90)
    print("Q1: HORIZON -- does the tavg crossing drift between Tmult=1 and Tmult=2?")
    print("    If yes, T=L numbers are finite-time biased.")
    print("=" * 90)
    obs_t = a.obs + "_tavg50"
    for z in zetas:
        for L in Ls:
            ref_rows = get(rows, zeta=z, L=L, mode=a.ref)
            tmrs = sorted({round(float(r.get("Tmult", 1.0)), 2) for r in ref_rows})
            if len(tmrs) < 2:
                continue
            lams = sorted({round(float(r["lam"]), 6) for r in ref_rows})
            print("  zeta=%.2f  L=%d   Tmults found: %s" % (z, L, tmrs))
            for tm in tmrs:
                sub = get(ref_rows, **{"Tmult": float(tm)})
                if not sub:
                    continue
                vbl = {}
                for lam in lams:
                    sub_l = get(sub, lam=lam) or get(sub, **{"lambda": lam})
                    v = col(sub_l, obs_t)
                    if len(v) >= 3:
                        vbl[lam] = v
                pu, lc, std = crossing(vbl, lams, a.B, rng)
                print("    Tmult=%.0f  lam*=%.4f (std %.4f)  P(unique)=%.3f"
                      % (tm, lc, std, pu))
            print()

    # ------------------------------------------------------------------ Q2
    print("=" * 90)
    print("Q2: BIAS-FREE -- is each arm's mean compatible with '%s' arm?" % a.ref)
    print("    Difference per (zeta, lam): mean ± 2*pooled-SEM.")
    print("    A CI straddling zero means COMPATIBLE (no bias detected).")
    print("=" * 90)
    for z in zetas:
        for L in Ls:
            ref = get(rows, zeta=z, L=L, mode=a.ref)
            lams = sorted({round(float(r.get("lam", r.get("lambda", 0.0))), 6) for r in ref})
            if not ref:
                continue
            print("  zeta=%.2f  L=%d" % (z, L))
            for mode in [m for m in modes if m != a.ref]:
                arm = get(rows, zeta=z, L=L, mode=mode)
                diffs = []
                for lam in lams:
                    rv = col(get(ref, lam=lam), a.obs)
                    av = col(get(arm, lam=lam), a.obs)
                    if len(rv) < 3 or len(av) < 3:
                        continue
                    d = av.mean() - rv.mean()
                    se = np.sqrt(rv.var(ddof=1)/len(rv) + av.var(ddof=1)/len(av))
                    diffs.append((lam, d, se))
                if not diffs:
                    continue
                bad = [(l, d, s) for l, d, s in diffs if abs(d) > 2 * s]
                ev = np.mean([r["n_events"] for r in arm]) if arm else np.nan
                print("    %-10s  n_events=%.1f  %d of %d lam CIs straddle 0%s"
                      % (mode, ev, len(diffs) - len(bad), len(diffs),
                         ("  *** BIAS FLAG: %d lambda outside 2sigma" % len(bad))
                         if bad else ""))
            print()

    # ------------------------------------------------------------------ Q3
    print("=" * 90)
    print("Q3: VARIANCE -- does adaptive resampling reduce Var[%s] at matched cost?" % a.obs)
    print("    F-test p < 0.05 required; p < 0.5 = arm has lower variance than ref.")
    print("    Variance ratio < 1 = better.  Wall ratio shown separately.")
    print("=" * 90)
    print(f"  {'mode':<12}{'zeta':>7}{'L':>5}{'n':>5}{'var ratio':>12}"
          f"{'F-test p':>10}{'wall ratio':>12}{'n_events':>10}")
    for z in zetas:
        for L in Ls:
            ref = get(rows, zeta=z, L=L, mode=a.ref)
            rv_all = col(ref, a.obs)
            if len(rv_all) < 5:
                continue
            rw = np.median(col(ref, "wall")) if ref else np.nan
            for mode in [m for m in modes if m != a.ref]:
                arm = get(rows, zeta=z, L=L, mode=mode)
                av_all = col(arm, a.obs)
                if len(av_all) < 5:
                    continue
                ratio, pval = f_test(rv_all, av_all)
                aw = np.median(col(arm, "wall"))
                ev = np.mean([r["n_events"] for r in arm])
                flag = " *" if (pval < 0.05 and ratio < 1) else ""
                print(f"  {mode:<12}{z:>7.2f}{L:>5}{len(av_all):>5}"
                      f"{ratio:>12.3f}{pval:>10.3f}{aw/rw:>12.2f}{ev:>10.1f}{flag}")
    print("  * = statistically significant variance reduction vs reference")

    # ------------------------------------------------------------------ Q4
    print("\n" + "=" * 90)
    print("Q4: NEVER-MODE ESS -- does interaction-free sampling stay workable at L=48?")
    print("    essFrac > 0.15 at zeta <= 0.20 means the low-zeta campaign can run")
    print("    without any cloning.  Decays with T, so also broken out by Tmult.")
    print("=" * 90)
    never = get(rows, mode="never")
    print(f"  {'zeta':>7}{'L':>5}{'Tmult':>7}{'n':>5}{'essF med':>10}"
          f"{'gess_root':>11}{'gess_recent':>13}{'CMI':>9}")
    for z in zetas:
        for L in Ls:
            for tm in tmults:
                sub = get(never, zeta=z, L=L, **{"Tmult": tm})
                if not sub:
                    continue
                ef = col(sub, "ess_final")
                gr = col(sub, "gess_root")
                ge = col(sub, "gess_recent")
                cv = col(sub, a.obs)
                print(f"  {z:>7.2f}{L:>5}{tm:>7.1f}{len(sub):>5}"
                      f"{np.nanmedian(ef):>10.3f}{np.nanmedian(gr):>11.1f}"
                      f"{np.nanmedian(ge):>13.1f}{np.nanmean(cv):>9.4f}")

    # ------------------------------------------------------------------ Q5 bonus
    print("\n" + "=" * 90)
    print("BONUS: c_eff* comparison across arms (benchmark, not a gate)")
    print("  Is c_eff at its scale-invariant point consistent across resampling modes?")
    print("=" * 90)
    for z in zetas:
        for L in Ls:
            for mode in modes:
                sub = get(rows, zeta=z, L=L, mode=mode)
                cv = col(sub, "c_eff")
                if len(cv) < 3:
                    continue
                print("  zeta=%.2f L=%d %-10s  c_eff = %.3f +- %.3f"
                      % (z, L, mode, cv.mean(), cv.std(ddof=1)/np.sqrt(len(cv))))


if __name__ == "__main__":
    main()
