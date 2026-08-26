#!/usr/bin/env python3
"""Score the Phase 2 pilot against the PRE-REGISTERED criterion.

Criterion, fixed before the run:
  the coupled arm attains a UNIQUE sign change of D(lambda) in >= 80 percent of
  realisation bootstrap resamples, for >= 2 of the 3 L-pairs, at a zeta where
  the independent arm attains 0.

Multiplicity is the statistic because it is what LAMC's gate rejects on: at
zeta = 0.40..0.70 it rejects 8-10 of 10 pairs on sign_change_multiplicity and
only 0-2 on curve_collapse.
"""
import os, json, glob, argparse
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
    p.add_argument("--obs", default="CMI_mean")
    p.add_argument("--B", type=int, default=1000)
    a = p.parse_args()
    rng = np.random.default_rng(20260826)

    rows = load(a.dir)
    if not rows:
        print("no records"); return
    modes = sorted({r["mode"] for r in rows})
    Ls = sorted({r["L"] for r in rows})
    lams = sorted({round(r["lambda"], 6) for r in rows})
    zeta = rows[0]["zeta"]
    print("zeta=%.3f  L=%s  nlam=%d  modes=%s  obs=%s  records=%d"
          % (zeta, Ls, len(lams), modes, a.obs, len(rows)))

    V = {}
    for r in rows:
        V.setdefault((r["mode"], r["L"], round(r["lambda"], 6)), []).append(r[a.obs])

    print("\n=== P(unique sign change of D) under the realisation bootstrap ===")
    print("(the number the pre-registered criterion is stated in)\n")
    hdr = f"{'pair':<12}" + "".join(f"{m:>18}" for m in modes)
    print(hdr)
    summary = {m: [] for m in modes}
    for i in range(len(Ls)):
        for j in range(i + 1, len(Ls)):
            L1, L2 = Ls[i], Ls[j]
            line = f"{f'({L1},{L2})':<12}"
            for m in modes:
                A = [np.asarray(V.get((m, L1, l), [])) for l in lams]
                Bv = [np.asarray(V.get((m, L2, l), [])) for l in lams]
                ok = [k for k in range(len(lams)) if len(A[k]) >= 3 and len(Bv[k]) >= 3]
                if len(ok) < 5:
                    line += f"{'--':>18}"; continue
                cnt = {0: 0, 1: 0, 2: 0}
                for _ in range(a.B):
                    d = np.array([Bv[k][rng.integers(0, len(Bv[k]), len(Bv[k]))].mean()
                                  - A[k][rng.integers(0, len(A[k]), len(A[k]))].mean()
                                  for k in ok])
                    n = int((np.diff(np.sign(d)) != 0).sum())
                    cnt[0 if n == 0 else (1 if n == 1 else 2)] += 1
                pu = cnt[1] / a.B
                summary[m].append(pu)
                line += f"{pu:>10.3f} ({cnt[2]/a.B:.2f}m)"[:18].rjust(18)
            print(line)
    print("\n  format: P(unique)  (P(multiple) in parentheses)")

    print("\n=== VERDICT AGAINST THE PRE-REGISTERED CRITERION ===")
    for m in modes:
        s = summary[m]
        npass = sum(1 for x in s if x >= 0.80)
        print("  %-14s pairs with P(unique) >= 0.80 : %d of %d   [median %.3f]"
              % (m, npass, len(s), np.median(s) if s else float("nan")))
    if "coupled" in summary and "independent" in summary:
        c = sum(1 for x in summary["coupled"] if x >= 0.80)
        i = sum(1 for x in summary["independent"] if x >= 0.80)
        print()
        if c >= 2 and i == 0:
            print("  PASS: coupling collapses the multiplicity where the production")
            print("        seeding cannot. Adopt coupled lambda for Phase 3.")
        elif c > i:
            print("  PARTIAL: coupling helps but misses the bar. Consider the exact")
            print("        version (uniformization, shared dominating rate) -- but that")
            print("        implementation is unvalidated (ARCH sec 7.2) and must be")
            print("        certified first.")
        else:
            print("  FAIL: cheap coupling does not fix the multiplicity. Do NOT spend")
            print("        on uniformization for this reason. The binding constraint is")
            print("        elsewhere -- most likely per-point variance, which means the")
            print("        levers are the wide-L ladder and more realisations.")

    print("\n=== HOW MUCH CORRELATION SURVIVED TO t = T (the stated caveat) ===")
    print("lag-1 correlation of the per-realisation residual across adjacent lambda")
    for m in modes:
        cs = []
        for L in Ls:
            M = []
            for l in lams:
                v = V.get((m, L, l), [])
                if len(v) >= 3:
                    M.append(np.asarray(v))
            if len(M) < 3:
                continue
            n = min(len(x) for x in M)
            A = np.array([x[:n] for x in M])
            R = A - A.mean(axis=1, keepdims=True)
            for k in range(len(M) - 1):
                sd = R[k].std() * R[k + 1].std()
                if sd > 0:
                    cs.append(float((R[k] * R[k + 1]).mean() / sd))
        print("  %-14s median rho = %s" % (m, f"{np.median(cs):+.3f}" if cs else "--"))
    print("  independent should sit near 0. coupled near 1 means the streams stayed")
    print("  in step; near 0 means they desynchronised and the exact version is needed.")


if __name__ == "__main__":
    main()
