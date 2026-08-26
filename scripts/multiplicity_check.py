#!/usr/bin/env python3
"""LAMC's multiplicity gate, applied to the small-L omnibus data.

THE QUESTION.  The mid-zeta failure was never about precision.  It was about
CROSSING MULTIPLICITY: LAMC's frozen gate rejects 8-10 of 10 L-pairs at
zeta = 0.40..0.70 on sign_change_multiplicity, and at zeta = 0.55 specifically
CMI attains a unique crossing in 4.3 percent of bootstrap resamples with
n_valid = 0 of 10 pairs.  Adding lambda points and realisations did not move it.

I built the c_eff locator on the premise that the affordable L range could not
support finite-size scaling at all.  The omnibus then measured CMI's L-scramble
ratio at zeta = 0.55 as 11.96 on L in {32,48,64}, against 2.4 on refine's
L in {64..128} at identical lever arm ln 2 -- a fivefold gain in genuine
L-signal purely from dropping to smaller L.

That raises the possibility that the premise was wrong: not that the L range is
too narrow, but that the WRONG PART of it was being used.  The L-scramble ratio
does not settle this, because it is not the statistic that failed.  This script
computes the one that did.

IF CMI's multiplicity is also repaired at small L, then c_eff is unnecessary and
the campaign is simply "run the incumbent observable on a lower ladder" -- far
cheaper and less exotic than anything else proposed.  If it is not repaired,
c_eff's 100 percent crossing rate is doing real work and is worth its width.

Same statistic, same zeta, different ladder.  Read-only.  No new simulation.
"""
import os, sys, json, glob, argparse
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--obs", default="CMI,B_L,MI_ends,varN,I3,S_AB,c_eff")
    a = p.parse_args()
    rng = np.random.default_rng(20260826)

    rows = load(a.dir)
    if not rows:
        print("no records"); return
    zs = sorted({round(float(r["zeta"]), 6) for r in rows})
    Ls = sorted({int(r["L"]) for r in rows})
    lams = sorted({round(float(r["lambda"]), 6) for r in rows})
    print("records %d   zeta %s   L %s   nlam %d" % (len(rows), zs, Ls, len(lams)))
    print("\nReference, LAMC frozen gate on refine L in {64..128} at zeta = 0.55:")
    print("  CMI unique-crossing probability 4.3 pct, n_valid 0 of 10 pairs.\n")

    for z in zs:
        s = [r for r in rows if np.isclose(float(r["zeta"]), z)]
        print("=" * 88)
        print("zeta = %.2f" % z)
        print("=" * 88)
        print(f"{'obs':<10}{'pair':>12}{'P(none)':>10}{'P(unique)':>11}"
              f"{'P(multi)':>10}{'med #sc':>9}{'sep both sides':>16}")
        for obs in a.obs.split(","):
            key = obs + "_mean"
            if key not in s[0]:
                continue
            V = {}
            for Lv in Ls:
                for lv in lams:
                    v = np.array([r[key] for r in s
                                  if int(r["L"]) == Lv
                                  and np.isclose(float(r["lambda"]), lv)], float)
                    if len(v) >= 3:
                        V[(Lv, lv)] = v
            uniq = []
            for i in range(len(Ls)):
                for j in range(i + 1, len(Ls)):
                    L1, L2 = Ls[i], Ls[j]
                    ok = [lv for lv in lams if (L1, lv) in V and (L2, lv) in V]
                    if len(ok) < 5:
                        continue
                    A = [V[(L1, lv)] for lv in ok]
                    Bv = [V[(L2, lv)] for lv in ok]
                    cnt = {0: 0, 1: 0, 2: 0}
                    nsc = []
                    for _ in range(a.B):
                        d = np.array([Bv[k][rng.integers(0, len(Bv[k]), len(Bv[k]))].mean()
                                      - A[k][rng.integers(0, len(A[k]), len(A[k]))].mean()
                                      for k in range(len(ok))])
                        n = int((np.diff(np.sign(d)) != 0).sum())
                        nsc.append(n)
                        cnt[0 if n == 0 else (1 if n == 1 else 2)] += 1
                    # LAMC curve_separation: |D| > 3 se(D) on BOTH sides of the crossing
                    dm = np.array([Bv[k].mean() - A[k].mean() for k in range(len(ok))])
                    se = np.array([np.sqrt(Bv[k].var(ddof=1)/len(Bv[k])
                                           + A[k].var(ddof=1)/len(A[k]))
                                   for k in range(len(ok))])
                    t = np.abs(dm) / np.maximum(se, 1e-12)
                    sgn = np.sign(dm)
                    sep = (np.any((t > 3) & (sgn > 0)) and np.any((t > 3) & (sgn < 0)))
                    pu = cnt[1] / a.B
                    uniq.append(pu)
                    print(f"{obs:<10}{f'({L1},{L2})':>12}{cnt[0]/a.B:>10.3f}{pu:>11.3f}"
                          f"{cnt[2]/a.B:>10.3f}{np.median(nsc):>9.0f}"
                          f"{'yes' if sep else 'no':>16}")
            if uniq:
                print(f"{'':<10}{'MEDIAN':>12}{'':>10}{np.median(uniq):>11.3f}\n")

    print("=" * 88)
    print("READING THIS")
    print("=" * 88)
    print("P(unique) is the statistic LAMC's gate rejects on and the one that did")
    print("NOT move when lambda points went 7 -> 15 and realisations 2 -> 12.")
    print("If CMI's P(unique) here is near 1 where it was 0.043 on L in {64..128},")
    print("the mid-zeta pathology is a property of the LADDER, not the observable,")
    print("and c_eff is not needed.  If CMI is still low while c_eff is high, the")
    print("observable is doing the work and c_eff's wider interval is worth paying.")


if __name__ == "__main__":
    main()
