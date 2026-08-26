#!/usr/bin/env python3
"""Analysis for Phase 0: the paired configuration validation and the cost model.

  python scripts/analyse_phase0.py validation --dir $WORKDIR/pps/validate
  python scripts/analyse_phase0.py benchmark  --glob '$WORKDIR/pps/bench/*.json'

VALIDATION IS PAIRED.  Arms share a seed at fixed (L, lambda, zeta, real), so
the estimator is the mean of the PER-SEED difference and its bootstrap CI, not
a comparison of arm means.  Differencing per seed cancels the realisation-level
fluctuation that both arms share; comparing means throws that away and inflates
the error by roughly sqrt(2) for nothing.

DEC-MASTER-METRIC-001: wall time is diagnostic_only.  The two factors of
t_wall * sigma^2 are reported SEPARATELY and never multiplied into a score here.
"""
import os, sys, json, glob, argparse
import numpy as np


def load_validation(root):
    rows = []
    for f in glob.glob(os.path.join(root, "**", "real*.json"), recursive=True):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    return [r for r in rows if r.get("status") == "ok"], len(rows)


def boot_paired(d, B=5000, seed=7):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return np.nan, np.nan, np.nan
    m = float(d.mean())
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cmd_validation(a):
    rows, ntot = load_validation(a.dir)
    print("records: %d ok of %d" % (len(rows), ntot))
    if not rows:
        return
    arms = sorted({r["arm"] for r in rows})
    cells = sorted({(r["L"], r["zeta"]) for r in rows})
    print("arms  :", ", ".join(arms))
    print("cells :", cells)
    base = a.baseline
    if base not in arms:
        print("baseline %s absent" % base); return

    idx = {}
    for r in rows:
        idx[(r["arm"], r["L"], r["zeta"], r["real"])] = r

    OBS = ["B_L_mean", "CMI_mean", "S_AB_mean", "theta_hat"]
    print("\n=== PAIRED DIFFERENCES vs %s  (mean of per-seed diff, 95%% bootstrap CI) ===" % base)
    print("A CI straddling zero means the knob does not move the observable.\n")
    for (L, z) in cells:
        print("--- L=%d  zeta=%.2f ---" % (L, z))
        hdr = f"{'arm':<15}" + "".join(f"{o:>26}" for o in OBS) + f"{'n':>5}{'wall ratio':>12}"
        print(hdr)
        b_wall = [idx[k]["wall_s"] for k in idx if k[0] == base and k[1] == L and k[2] == z]
        for arm in arms:
            if arm == base:
                continue
            diffs = {o: [] for o in OBS}
            wr = []
            n = 0
            for k in list(idx):
                if k[0] != arm or k[1] != L or k[2] != z:
                    continue
                kb = (base, k[1], k[2], k[3])
                if kb not in idx:
                    continue
                n += 1
                for o in OBS:
                    diffs[o].append(idx[k][o] - idx[kb][o])
                wr.append(idx[k]["wall_s"] / max(idx[kb]["wall_s"], 1e-9))
            if n == 0:
                continue
            line = f"{arm:<15}"
            for o in OBS:
                m, lo, hi = boot_paired(diffs[o])
                star = " " if (np.isfinite(lo) and lo <= 0 <= hi) else "*"
                # do NOT truncate: the star is the whole point of the column and
                # a [:26] slice was silently cutting it off.
                line += f"  {m:>+9.5f}[{lo:>+8.5f},{hi:>+8.5f}]{star}"
            line += f"{n:>5}" + f"{np.median(wr):>12.2f}"
            print(line)
        print("   (* = CI excludes zero, i.e. the configuration DOES shift the observable)")
        print()

    print("=== GENEALOGY AND WALL BY ARM (diagnostic_only) ===")
    print(f"{'arm':<15}{'L':>5}{'zeta':>6}{'n_anc med':>11}{'gen_ESS med':>13}"
          f"{'n_resamp med':>14}{'ESS med':>9}{'wall med s':>12}")
    for arm in arms:
        for (L, z) in cells:
            s = [r for r in rows if r["arm"] == arm and r["L"] == L and r["zeta"] == z]
            if not s:
                continue
            g = [r.get("genealogical_ess", np.nan) for r in s]
            print(f"{arm:<15}{L:>5}{z:>6.2f}"
                  f"{np.median([r['n_distinct_ancestors'] for r in s]):>11.1f}"
                  f"{np.nanmedian(g):>13.2f}"
                  f"{np.median([r.get('n_resampling_events',-1) for r in s]):>14.0f}"
                  f"{np.median([r['eff_sample_size'] for r in s]):>9.1f}"
                  f"{np.median([r['wall_s'] for r in s]):>12.2f}")


def cmd_benchmark(a):
    rows = []
    for f in glob.glob(a.glob):
        try:
            rows += [r for r in json.load(open(f)) if r.get("status") == "ok"]
        except Exception as e:
            print("skip", f, e)
    if not rows:
        print("no rows"); return
    print("points: %d" % len(rows))
    sweeps = sorted({r["sweep"] for r in rows})

    def fit(x, y):
        x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            return np.nan, np.nan
        A = np.vstack([x[ok], np.ones(ok.sum())]).T
        b, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        r = y[ok] - A @ b
        se = np.sqrt((r @ r / max(ok.sum() - 2, 1)) * np.linalg.inv(A.T @ A)[0, 0])
        return float(b[0]), float(se)

    AX = {"L": "L", "N_c": "N_c", "T": "T", "dtau": "dtau_mult", "lam": "lam",
          "zeta": "zeta", "Lprod": "L"}
    for s in sweeps:
        sub = [r for r in rows if r["sweep"] == s]
        ax = AX.get(s)
        print("\n--- sweep %s (%d points) ---" % (s, len(sub)))
        if ax is None:
            for r in sorted(sub, key=lambda r: (r["solver"], r["jump"], r["stride"])):
                print("   %s/%-8s/s%d  wall=%8.3f  s/clone-step=%.3e"
                      % (r["solver"], r["jump"], r["stride"], r["wall_s"], r["s_per_clone_step"]))
            continue
        agg = {}
        for r in sub:
            agg.setdefault(r[ax], []).append(r)
        xs = sorted(agg)
        print(f"   {ax:>10}{'wall med':>12}{'s/cl-step':>13}{'/L^2':>13}{'n_anc':>8}")
        for x in xs:
            g = agg[x]
            print(f"   {x:>10}{np.median([r['wall_s'] for r in g]):>12.3f}"
                  f"{np.median([r['s_per_clone_step'] for r in g]):>13.3e}"
                  f"{np.median([r['s_per_clone_step_L2'] for r in g]):>13.3e}"
                  f"{np.median([r['n_distinct_ancestors'] for r in g]):>8.0f}")
        p, se = fit([x for x in xs], [np.median([r["wall_s"] for r in agg[x]]) for x in xs])
        print(f"   wall ~ {ax}^({p:.2f} +- {se:.2f})")
        if s in ("L", "Lprod"):
            q, qse = fit(xs, [np.median([r["s_per_clone_step_L2"] for r in agg[x]]) for x in xs])
            print(f"   s/(clone*step*L^2) ~ L^({q:.2f} +- {qse:.2f})   "
                  f"[0 => per-step work is exactly O(L^2)]")
        if s == "T":
            print("   [exponent 1.00 => cost linear in T, which is what licenses")
            print("    reading the short sweeps as production-T estimates]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validation"); v.add_argument("--dir", required=True)
    v.add_argument("--baseline", default="A_production"); v.set_defaults(f=cmd_validation)
    b = sp.add_parser("benchmark"); b.add_argument("--glob", required=True)
    b.set_defaults(f=cmd_benchmark)
    a = p.parse_args(); a.f(a)
