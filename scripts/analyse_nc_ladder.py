#!/usr/bin/env python3
"""Score the N_c ladder: does lambda_c MOVE with population size?

The pass condition is stated before the data: lambda_c must be flat in N_c to
within 0.003, which is well below the 0.0138 statistical width CMI achieves at
zeta = 0.55 and far below the 0.08 observable spread.  Genealogy healing while
lambda_c sits still is a PASS, not a concern -- DEC-MASTER-METRIC-001 makes ESS
and GESS diagnostic_only, and the whole point is that the locator is what
matters.

Reports, per observable:
  lambda_c(N_c) from the joint all-L slope-zero, with bootstrap CI
  the trend in lambda_c across the ladder and whether it exceeds 0.003
  P(unique) at each rung, i.e. does more population fix the MULTIPLICITY
  genealogical ESS as a FRACTION of N_c, to see whether the sampler heals at all
"""
import os, json, glob, argparse
import numpy as np

OBS = ["CMI", "B_L", "c_eff", "MI_ends"]


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


def slope_zero(lams, per, Ls, rng, boot):
    lnL = np.log(np.array(Ls, float))
    s = []
    for lv in lams:
        ys = []
        for Lv in Ls:
            v = per.get((Lv, lv))
            if v is None or len(v) < 3:
                ys = None; break
            ys.append(v[rng.integers(0, len(v), len(v))].mean() if boot else v.mean())
        s.append(np.polyfit(lnL, np.array(ys), 1)[0] if ys else np.nan)
    s = np.array(s); lam = np.array(lams)
    ok = np.isfinite(s)
    if ok.sum() < 4:
        return np.nan, 0
    s, lam = s[ok], lam[ok]
    i = np.where(np.diff(np.sign(s)) != 0)[0]
    if len(i) == 0:
        return np.nan, 0
    k = i[len(i) // 2]
    if s[k + 1] == s[k]:
        return float(lam[k]), len(i)
    return float(lam[k] - s[k] * (lam[k + 1] - lam[k]) / (s[k + 1] - s[k])), len(i)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--B", type=int, default=400)
    a = p.parse_args()
    rng = np.random.default_rng(20260826)
    rows = load(a.dir)
    if not rows:
        print("no records"); return
    Ncs = sorted({int(r["N_c"]) for r in rows})
    Ls = sorted({int(r["L"]) for r in rows})
    lams = sorted({round(float(r["lambda"]), 6) for r in rows})
    print("records %d   N_c %s   L %s   nlam %d   zeta %.3f"
          % (len(rows), Ncs, Ls, len(lams), rows[0]["zeta"]))

    print("\n=== GENEALOGY: does the sampler heal at all? ===")
    print(f"  {'N_c':>7}{'n_anc med':>11}{'gen_ESS med':>13}{'gen_ESS/N_c':>13}"
          f"{'ESS med':>10}{'wall med':>11}")
    for Nc in Ncs:
        s = [r for r in rows if int(r["N_c"]) == Nc]
        g = [r.get("genealogical_ess", np.nan) for r in s]
        gf = [r.get("gen_ess_frac", np.nan) for r in s]
        print(f"  {Nc:>7}{np.median([r['n_distinct_ancestors'] for r in s]):>11.1f}"
              f"{np.nanmedian(g):>13.2f}{np.nanmedian(gf):>13.4f}"
              f"{np.median([r['eff_sample_size'] for r in s]):>10.1f}"
              f"{np.median([r['wall_traj_s'] for r in s]):>11.1f}")

    print("\n=== THE TEST: does lambda_c move with N_c? ===")
    print("PASS = flat to within 0.003 across the ladder.\n")
    for obs in OBS:
        key = obs + "_mean"
        if key not in rows[0]:
            continue
        print("--- %s ---" % obs)
        print(f"  {'N_c':>7}{'lam_c':>9}{'68% CI':>19}{'width':>9}{'#zeros':>8}")
        vals = []
        for Nc in Ncs:
            s = [r for r in rows if int(r["N_c"]) == Nc]
            per = {}
            for Lv in Ls:
                for lv in lams:
                    v = np.array([r[key] for r in s if int(r["L"]) == Lv
                                  and np.isclose(float(r["lambda"]), lv)], float)
                    if len(v) >= 3:
                        per[(Lv, lv)] = v
            lc, nz = slope_zero(lams, per, Ls, rng, False)
            bs = [slope_zero(lams, per, Ls, rng, True)[0] for _ in range(a.B)]
            bs = np.array([b for b in bs if np.isfinite(b)])
            q16, q84 = (np.percentile(bs, [16, 84]) if len(bs) > 30 else (np.nan, np.nan))
            print(f"  {Nc:>7}{lc:>9.4f}{f'[{q16:.4f},{q84:.4f}]':>19}"
                  f"{q84-q16:>9.4f}{nz:>8}")
            if np.isfinite(lc):
                vals.append((Nc, lc))
        if len(vals) >= 2:
            v = [x[1] for x in vals]
            drift = max(v) - min(v)
            mono = all(v[i] <= v[i+1] for i in range(len(v)-1)) or \
                   all(v[i] >= v[i+1] for i in range(len(v)-1))
            verdict = "PASS, flat" if drift < 0.003 else (
                "FAIL, monotone drift" if mono else "drifts, non-monotone")
            print(f"  spread across ladder = {drift:.4f}   {verdict}")
            if drift >= 0.003 and mono:
                print("  -> finite-N_c BIAS is the bottleneck. More realisations will")
                print("     not help; they shrink eps and leave b(N_c) untouched.")
            elif drift < 0.003:
                print("  -> population size is not biasing this locator. If the")
                print("     genealogy healed above while this stayed flat, the")
                print("     genealogy is ugly and harmless, and the budget should")
                print("     go to realisations rather than population.")
        print()


if __name__ == "__main__":
    main()
