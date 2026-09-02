#!/usr/bin/env python3
"""Combined analysis — TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief section 11.

    python3 analysis/combined_analysis.py [--task DIR]

Produces, in order:

  A. the N_c convergence table at L = 128, lambda = 0.3032, including the DIRECT
     rung-to-rung differences and their bootstrap CIs;
  B. variance-scaling (gamma) diagnostics over a >=3-rung window scan;
  C. the lambda-stencil results at low L and at L = 128, with adjacent
     differences, the second finite difference and the S1-S4 consistency
     diagnostics;
  D. the per-L-class production recommendation, restricted to the three
     permitted verdicts.

and then evaluates the FROZEN falsification targets F1-F7.

Everything it applies is frozen in ../analysis_spec.yaml and
../SMOOTHNESS_CRITERION.md. Nothing here is tuned. Uncertainty always comes
from INDEPENDENT POPULATIONS; within-clone spread appears only as a VIF/N_eff
diagnostic and is never a standard error.

This script reads results and writes a report. It contains no scheduler call.
"""
import os, sys, csv, json, glob, math, argparse
import numpy as np

BOOT, SEED = 10000, 20260902
LAM_M, LAM_0, LAM_P = 0.2932, 0.3032, 0.3132
DLAM = 0.010
TAU_STEP = 0.0732      # one lambda-grid step in CMI at L=128; see analysis_spec
TAU_PLOT = 0.0146      # 0.2 of a grid step; pre-registered as NOT achievable
GAMMA_BAND = (0.5, 1.5)
R_SPACING_BAND = (2.0, 20.0)

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(task):
    """Every independent population in the programme at the frozen cell family.

    New results come from the arm result directories. Completed predecessor
    populations come from the TRACKED frozen_inputs/ snapshot, whose sha256s
    are recorded in INPUTS_LEDGER.md and which reproduces the published ARM1
    and ARM2 analyses digit for digit. The predecessor task archives are read
    only if present and are never required.
    """
    pops = {}          # (L, T, zeta, lam, N_c) -> list of dicts
    n_new = 0

    def add(L, T, z, lam, nc, m, s2, src, extra=None):
        k = (int(L), float(T), float(z), round(float(lam), 6), int(nc))
        d = dict(mean=float(m), within=float(s2), source=src)
        if extra:
            d.update(extra)
        pops.setdefault(k, []).append(d)

    frozen = os.path.join(task, "frozen_inputs", "predecessor_populations.csv")
    if os.path.isfile(frozen):
        for r in csv.DictReader(open(frozen)):
            if r["status"] not in ("ok", ""):
                continue
            add(r["L"], r["T"], r["zeta"], r["lam"], r["N_c"],
                r["cmi_weighted_mean"], r["cmi_within_var"], "frozen:" + r["arm"])
    else:
        print(f"WARNING: {frozen} is absent; the predecessor rungs will be "
              f"missing and F1 cannot be evaluated.")

    for arm in sorted(glob.glob(os.path.join(task, "arm*"))):
        for p in sorted(glob.glob(os.path.join(arm, "results", "*.json"))):
            r = json.load(open(p))
            if r.get("status") not in (None, "ok"):
                continue
            add(r["L"], r["T"], r["zeta"], r["lam"], r["N_c"],
                r["cmi_weighted_mean"], r["cmi_within_var"],
                "new:" + os.path.basename(arm),
                dict(anc=r.get("n_distinct_anc_final"),
                     wall=r.get("wall_s"),
                     nonfinite=r.get("n_nonfinite", 0)))
            n_new += 1
    return pops, n_new


def stats(rows):
    m = np.array([r["mean"] for r in rows])
    s2 = np.array([r["within"] for r in rows])
    return m, s2


def cell(pops, L, lam, nc):
    for k, v in pops.items():
        if k[0] == L and abs(k[3] - lam) < 1e-9 and k[4] == nc:
            return v
    return None


# --------------------------------------------------------------------------
# bootstrap helpers — always resampling INDEPENDENT POPULATIONS
# --------------------------------------------------------------------------
def boot_means(rng, m, n=BOOT):
    return m[rng.integers(0, m.size, (n, m.size))].mean(axis=1)


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


# --------------------------------------------------------------------------
def section_A(pops, rng, rep):
    print("\n" + "=" * 78)
    print("  A.  N_c CONVERGENCE AT L = 128, T = 128, zeta = 0.35, lambda = 0.3032")
    print("=" * 78)
    lad = {}
    for nc in (64, 128, 256, 512, 1024, 2048):
        rows = cell(pops, 128, LAM_0, nc)
        if rows and len(rows) >= 6:
            lad[nc] = rows
    if not lad:
        print("  no populations at this cell yet.")
        return {}, {}
    print(f"  {'N_c':>6}{'R':>5}{'mean CMI':>11}{'SEM':>10}{'variance':>12}"
          f"{'VIF':>9}{'N_eff':>8}{'anc_med':>9}")
    tab = {}
    for nc in sorted(lad):
        m, s2 = stats(lad[nc])
        V = float(np.var(m, ddof=1)); s2m = float(np.mean(s2))
        anc = [r.get("anc") for r in lad[nc] if r.get("anc") is not None]
        tab[nc] = dict(R=int(m.size), mean=float(m.mean()),
                       sem=math.sqrt(V / m.size), var=V, within=s2m,
                       vif=V * nc / s2m if s2m else float("nan"),
                       n_eff=s2m / V if V else float("nan"), m=m)
        print(f"  {nc:>6}{m.size:>5}{m.mean():>11.5f}{math.sqrt(V / m.size):>10.5f}"
              f"{V:>12.4e}{tab[nc]['vif']:>9.2f}{tab[nc]['n_eff']:>8.2f}"
              f"{(int(np.median(anc)) if anc else -1):>9}")

    print("\n  DIRECT rung-to-rung differences (the primary convergence test;")
    print("  a 1/N_c fit is NEVER used to define 'converged'):")
    print(f"  {'pair':>16}{'Delta':>10}{'SEM':>9}{'95% CI':>22}"
          f"{'|D|<tau_step?':>15}")
    deltas = {}
    ncs = sorted(tab)
    for a, b in zip(ncs, ncs[1:]):
        ma, mb = tab[a]["m"], tab[b]["m"]
        d = float(mb.mean() - ma.mean())
        bs = boot_means(rng, mb) - boot_means(rng, ma)
        lo, hi = ci(bs)
        s = float(bs.std(ddof=1))
        inside = (lo > -TAU_STEP and hi < TAU_STEP)
        deltas[(a, b)] = dict(d=d, sem=s, lo=lo, hi=hi, inside_tau_step=inside)
        print(f"  {f'{a}->{b}':>16}{d:>10.5f}{s:>9.5f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{('YES' if inside else 'no'):>15}")
    print(f"\n  tau_step = {TAU_STEP:.4f} (one lambda-grid step in CMI at L=128)")
    print(f"  tau_plot = {TAU_PLOT:.4f} (0.2 grid step) — pre-registered as NOT")
    print(f"           achievable at this R; the achieved half-widths above are")
    print(f"           the honest resolution and no tighter claim may be made.")

    print("\n  DESCRIPTIVE ONLY — I(N_c) = I_inf + B/N_c. This is reported as a")
    print("  description of the observed rungs. TASK-2026-08-31-SMCCERT killed")
    print("  the claim that it is a controlled universal asymptotic bias law, so")
    print("  it is NOT extrapolated, NOT used as ground truth, and NOT used to")
    print("  decide convergence.")
    if len(ncs) >= 3:
        x = np.array([1.0 / n for n in ncs])
        y = np.array([tab[n]["mean"] for n in ncs])
        B0 = float(np.polyfit(x, y, 1)[0])
        bs = np.array([float(np.polyfit(
            x, [boot_means(rng, tab[n]["m"], 1)[0] for n in ncs], 1)[0])
            for _ in range(1000)])
        lo, hi = ci(bs)
        print(f"    B = {B0:+.3f}  CI = [{lo:+.3f}, {hi:+.3f}]   (descriptive)")
    rep["A"] = dict(table={k: {kk: vv for kk, vv in v.items() if kk != "m"}
                           for k, v in tab.items()},
                    deltas={f"{a}->{b}": v for (a, b), v in deltas.items()})
    return tab, deltas


def section_B(tab, rng, rep):
    print("\n" + "=" * 78)
    print("  B.  VARIANCE SCALING  gamma = -dlogVar/dlogN_c   (L = 128)")
    print("=" * 78)
    ncs = sorted(tab)
    if len(ncs) < 3:
        print("  fewer than three rungs; no window scan possible.")
        return {}
    def windows(ns):
        ws = [tuple(ns)]
        for w in range(len(ns) - 1, 2, -1):
            for i in range(len(ns) - w + 1):
                t = tuple(ns[i:i + w])
                if t not in ws:
                    ws.append(t)
        return ws
    out = {}
    for win in windows(ncs):
        lx = np.log([float(n) for n in win])
        g0 = float(-np.polyfit(lx, [math.log(max(tab[n]["var"], 1e-300))
                                    for n in win], 1)[0])
        bs = []
        for _ in range(2000):
            ly = []
            for n in win:
                m = tab[n]["m"]
                ly.append(math.log(max(np.var(m[rng.integers(0, m.size, m.size)],
                                              ddof=1), 1e-300)))
            bs.append(float(-np.polyfit(lx, ly, 1)[0]))
        lo, hi = ci(np.array(bs))
        inside = (lo >= GAMMA_BAND[0] and hi <= GAMMA_BAND[1])
        out[win] = (g0, lo, hi, inside)
        print(f"    {'+'.join(map(str, win)):>26}  gamma={g0:+.3f} "
              f"CI=[{lo:+.3f},{hi:+.3f}] width={hi - lo:.3f}"
              f"{'  INSIDE [0.5,1.5]' if inside else ''}")
    print("\n  gamma is a VARIANCE diagnostic only. It has no authority over the")
    print("  MEAN convergence verdicts in section A: a SUPPORTED gamma never")
    print("  licenses a stabilization claim (analysis_spec.yaml, F3 note).")
    rep["B"] = {"+".join(map(str, k)): dict(gamma=v[0], lo=v[1], hi=v[2],
                                            inside=v[3]) for k, v in out.items()}
    return out


def stencil(pops, L, nc, R_expect, rng):
    """Return the three-point stencil analysis at one L, or None."""
    pts = {}
    for lam in (LAM_M, LAM_0, LAM_P):
        rows = cell(pops, L, lam, nc)
        if not rows:
            return None
        m, s2 = stats(rows)
        pts[lam] = dict(m=m, s2=s2, R=int(m.size), mean=float(m.mean()),
                        var=float(np.var(m, ddof=1)),
                        sem=math.sqrt(np.var(m, ddof=1) / m.size),
                        within=float(np.mean(s2)))
    bm = {lam: boot_means(rng, pts[lam]["m"]) for lam in pts}
    d_m = bm[LAM_0] - bm[LAM_M]
    d_p = bm[LAM_P] - bm[LAM_0]
    q = bm[LAM_P] - 2 * bm[LAM_0] + bm[LAM_M]
    res = dict(L=L, N_c=nc, pts=pts,
               d_minus=dict(v=pts[LAM_0]["mean"] - pts[LAM_M]["mean"],
                            sem=float(d_m.std(ddof=1)), ci=ci(d_m)),
               d_plus=dict(v=pts[LAM_P]["mean"] - pts[LAM_0]["mean"],
                           sem=float(d_p.std(ddof=1)), ci=ci(d_p)),
               q=dict(v=pts[LAM_P]["mean"] - 2 * pts[LAM_0]["mean"]
                        + pts[LAM_M]["mean"],
                      sem=float(q.std(ddof=1)), ci=ci(q)))

    # ---- S1: split-half stability, deterministic permutation --------------
    s1 = {}
    srng = np.random.default_rng(SEED)
    for lam in (LAM_M, LAM_0, LAM_P):
        m = pts[lam]["m"]
        if m.size < 4:
            # each half needs >= 2 populations. Unreachable at the budgeted
            # R (32-96); S1 must FAIL rather than silently pass if it happens.
            s1[lam] = dict(diff=float("nan"), sem=float("nan"),
                           z=float("inf"), ok=False)
            continue
        idx = srng.permutation(m.size)
        h = m.size // 2
        a, b = m[idx[:h]], m[idx[h:2 * h]]
        s = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
        z = abs(a.mean() - b.mean()) / s if s > 0 else float("inf")
        s1[lam] = dict(diff=float(a.mean() - b.mean()), sem=s, z=float(z),
                       ok=bool(z <= 2.5))
    res["S1"] = dict(per_lambda=s1, ok=all(v["ok"] for v in s1.values()))

    # ---- S2: adjacent increments resolved ---------------------------------
    res["S2"] = dict(ok=bool(res["d_minus"]["ci"][1] < 0 or res["d_minus"]["ci"][0] > 0) and
                        bool(res["d_plus"]["ci"][1] < 0 or res["d_plus"]["ci"][0] > 0))

    # ---- S3: compatible with a locally smooth curve ------------------------
    qlo, qhi = res["q"]["ci"]
    resolved = (qlo > 0 or qhi < 0)
    bound = max(abs(qlo), abs(qhi))
    trend = abs(res["d_minus"]["v"]) + abs(res["d_plus"]["v"])
    res["S3"] = dict(resolved=bool(resolved), bound=bound, trend=trend,
                     ok=bool(resolved or bound <= trend))

    # ---- S4: no single population dominates -------------------------------
    s4 = {}
    for lam in (LAM_M, LAM_0, LAM_P):
        m = pts[lam]["m"]
        sd = m.std(ddof=1)
        z = np.abs(m - m.mean()) / sd if sd > 0 else np.zeros_like(m)
        from math import erf, sqrt as _s
        # Phi^-1(1 - 0.01/(2R)) by bisection, no scipy dependency
        target = 1 - 0.01 / (2 * m.size)
        lo_, hi_ = 0.0, 8.0
        for _ in range(200):
            mid = 0.5 * (lo_ + hi_)
            if 0.5 * (1 + erf(mid / _s(2))) < target:
                lo_ = mid
            else:
                hi_ = mid
        zcrit = 0.5 * (lo_ + hi_)
        drop = np.delete(m, int(np.argmax(z)))
        shift = abs(drop.mean() - m.mean())
        s4[lam] = dict(zmax=float(z.max()), zcrit=float(zcrit),
                       loo_shift=float(shift), sem=pts[lam]["sem"],
                       ok=bool(z.max() <= zcrit and shift <= pts[lam]["sem"]))
    res["S4"] = dict(per_lambda=s4, ok=all(v["ok"] for v in s4.values()))

    # ---- F7: does the jaggedness survive the bootstrap? -------------------
    # H0 is "the three points lie on a straight line in lambda". The null
    # ensemble is therefore built by placing the WEIGHTED-LEAST-SQUARES LINE at
    # the three lambdas and adding each lambda's own bootstrap fluctuation about
    # its own mean. Centring the null on the observed points instead would bake
    # the observed jaggedness into the null and force p ~ 1 by construction.
    x = np.array([LAM_M, LAM_0, LAM_P])
    y = np.array([pts[l]["mean"] for l in x])
    w = np.array([1.0 / pts[l]["sem"] ** 2 for l in x])
    X = np.vstack([np.ones(3), x - LAM_0]).T
    W = np.diag(w)

    def fit(yv):
        return np.linalg.solve(X.T @ W @ X, X.T @ W @ yv)

    def chi2(yv):
        r = yv - X @ fit(yv)
        return float(np.sum(w * r ** 2))

    obs = chi2(y)
    yhat = X @ fit(y)
    dev = np.vstack([bm[l] - bm[l].mean() for l in x])       # 3 x BOOT
    null = np.array([chi2(yhat + dev[:, i]) for i in range(2000)])
    p = float((null >= obs).mean())
    res["F7"] = dict(chi2=obs, p=p,
                     verdict=("SUPPORTED (jaggedness is real)" if p < 0.05 else
                              "KILLED (consistent with sampling noise)" if p > 0.32
                              else "INCONCLUSIVE"))

    # ---- F6: spacing verdict ----------------------------------------------
    r_m = abs(res["d_minus"]["v"]) / res["d_minus"]["sem"] if res["d_minus"]["sem"] else 0
    r_p = abs(res["d_plus"]["v"]) / res["d_plus"]["sem"] if res["d_plus"]["sem"] else 0
    r = min(r_m, r_p)
    res["spacing"] = dict(r_minus=r_m, r_plus=r_p, r=r,
                          verdict=("unnecessarily fine" if r < R_SPACING_BAND[0]
                                   else "too coarse" if r > R_SPACING_BAND[1]
                                   else "approximately appropriate"))
    return res


def show_stencil(s, label):
    print(f"\n  --- {label}:  L = {s['L']}, N_c = {s['N_c']}, "
          f"delta_lambda = {DLAM:g} ---")
    print(f"  {'lambda':>9}{'R':>5}{'mean CMI':>11}{'SEM':>10}{'variance':>12}"
          f"{'VIF':>9}")
    for lam in (LAM_M, LAM_0, LAM_P):
        p = s["pts"][lam]
        vif = p["var"] * s["N_c"] / p["within"] if p["within"] else float("nan")
        print(f"  {lam:>9.4f}{p['R']:>5}{p['mean']:>11.5f}{p['sem']:>10.5f}"
              f"{p['var']:>12.4e}{vif:>9.2f}")
    for nm, k in (("d_- = I_0 - I_-", "d_minus"), ("d_+ = I_+ - I_0", "d_plus"),
                  ("q   = I_+ -2I_0 +I_-", "q")):
        v = s[k]
        print(f"  {nm:<22} {v['v']:+.5f}  SEM {v['sem']:.5f}  "
              f"95% CI [{v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}]")
    print(f"  S1 replicate stability      "
          f"{'PASS' if s['S1']['ok'] else 'FAIL'}   "
          + "  ".join(f"{l:.4f}:z={v['z']:.2f}"
                      for l, v in s["S1"]["per_lambda"].items()))
    print(f"  S2 increments resolved      {'PASS' if s['S2']['ok'] else 'UNDETERMINED'}")
    if s["S3"]["resolved"]:
        why = "curvature resolved"
    else:
        why = (f"curvature unresolved; |q| bounded by {s['S3']['bound']:.5f} "
               f"vs trend {s['S3']['trend']:.5f}")
    print(f"  S3 locally smooth           "
          f"{'PASS' if s['S3']['ok'] else 'FAIL'}   ({why})")
    print(f"  S4 no dominant population   "
          f"{'PASS' if s['S4']['ok'] else 'FAIL'}   "
          + "  ".join(f"{l:.4f}:zmax={v['zmax']:.2f}/{v['zcrit']:.2f}"
                      for l, v in s["S4"]["per_lambda"].items()))
    print(f"  F7 jaggedness bootstrap     chi2={s['F7']['chi2']:.3f} "
          f"p={s['F7']['p']:.4f}  ->  {s['F7']['verdict']}")
    print(f"  spacing r = min(|d|/SEM(d)) = {s['spacing']['r']:.2f}  ->  "
          f"{s['spacing']['verdict']}")


def section_C(pops, rng, rep):
    print("\n" + "=" * 78)
    print("  C.  LAMBDA STENCIL   lambda in {%.4f, %.4f, %.4f},  zeta = 0.35"
          % (LAM_M, LAM_0, LAM_P))
    print("=" * 78)
    out = {}
    for label, L, nc in (("ARM B, low L", 64, 1024),
                         ("ARM C + armA512 central point, high L", 128, 512)):
        s = stencil(pops, L, nc, None, rng)
        if s is None:
            print(f"\n  --- {label}: incomplete (not all three lambdas present "
                  f"at L={L}, N_c={nc}) ---")
            continue
        show_stencil(s, label)
        out[L] = s
    rep["C"] = {str(L): {k: v for k, v in s.items() if k != "pts"}
                for L, s in out.items()}
    return out


def section_D(tab, deltas, sten, rep):
    print("\n" + "=" * 78)
    print("  D.  RECOMMENDATION FOR FINAL PRODUCTION")
    print("=" * 78)
    print("  Permitted verdicts, and ONLY these three:")
    print("    (i)   smallest tested N_c clearly inadequate")
    print("    (ii)  smallest tested N_c plausibly adequate")
    print("    (iii) unresolved / higher rung required")
    print("  No N_c(L, zeta, lambda) law is inferred. Three cells cannot support")
    print("  one, and inventing one is explicitly out of scope.\n")
    rec = {}
    ncs = sorted(tab)
    for a, b in zip(ncs, ncs[1:]):
        d = deltas.get((a, b))
        if not d:
            continue
        moving = (d["lo"] > 0 or d["hi"] < 0) and abs(d["d"]) >= TAU_STEP
        settled = d["inside_tau_step"]
        v = ("(i) clearly inadequate" if moving else
             "(ii) plausibly adequate" if settled else
             "(iii) unresolved / higher rung required")
        rec[f"L=128 N_c={a} vs {b}"] = v
        print(f"    L = 128, N_c = {a} judged against {b}:  {v}")
    for L, s in sorted(sten.items()):
        clean = s["S1"]["ok"] and s["S3"]["ok"] and s["S4"]["ok"]
        v = ("(ii) plausibly adequate for a local CMI(lambda) curve"
             if clean and s["S2"]["ok"] else
             "(iii) unresolved / higher rung required" if clean else
             "(i) clearly inadequate")
        rec[f"L={L} N_c={s['N_c']} stencil"] = v
        print(f"    L = {L}, N_c = {s['N_c']}, lambda stencil:  {v}")
    if sten:
        vs = {s["spacing"]["verdict"] for s in sten.values()}
        print(f"\n    delta_lambda = {DLAM:g} verdict per L: "
              + ", ".join(f"L={L}: {s['spacing']['verdict']}"
                          for L, s in sorted(sten.items())))
        print(f"    -> for the FINAL production grid: "
              f"{'keep 0.010' if vs == {'approximately appropriate'} else 'see per-L verdicts above'}")
    rep["D"] = rec


def targets(deltas, gam, sten, rep):
    print("\n" + "=" * 78)
    print("  FROZEN FALSIFICATION TARGETS  (criteria fixed before any new datum)")
    print("=" * 78)
    v = {}

    def rung(name, pair, stmt):
        d = deltas.get(pair)
        if not d:
            v[name] = "NOT EVALUATED (rung missing)"
        elif (d["lo"] > 0 or d["hi"] < 0) and abs(d["d"]) >= TAU_STEP:
            v[name] = "SUPPORTED (mean still moving materially)"
        elif d["inside_tau_step"]:
            v[name] = "KILLED (drift bounded inside tau_step)"
        else:
            v[name] = "INCONCLUSIVE"
        print(f"  {name}  {stmt}\n        -> {v[name]}"
              + (f"   Delta={d['d']:+.5f} CI=[{d['lo']:+.4f},{d['hi']:+.4f}]"
                 if d else ""))

    rung("F1", (256, 512), "L=128 mean CMI still moving between N_c 256 and 512")
    rung("F2", (512, 1024), "L=128 mean CMI still moving between N_c 512 and 1024")

    if gam:
        full = max(gam, key=len)
        g0, lo, hi, inside = gam[full]
        drop = tuple(sorted(full)[1:])
        v["F3"] = ("SUPPORTED (gamma CI inside [0.5,1.5])" if inside else
                   "KILLED (gamma useless)" if hi < 0.5 and
                   gam.get(drop, (0, 0, 9, 0))[2] < 0.5 else "INCONCLUSIVE")
        print(f"  F3  variance still falls usefully with N_c at L=128\n"
              f"        -> {v['F3']}   gamma={g0:+.3f} CI=[{lo:+.3f},{hi:+.3f}]"
              f"  (the exponent is NOT required to equal 1)")
    else:
        v["F3"] = "NOT EVALUATED"

    for name, L, stmt in (("F4", 64, "high-N_c low-L stencil gives a reproducible curve"),
                          ("F5", 128, "the same spacing remains usable at L=128")):
        s = sten.get(L)
        if not s:
            v[name] = "NOT EVALUATED (stencil incomplete)"
        elif not s["S1"]["ok"] or not s["S4"]["ok"]:
            v[name] = "KILLED (replicates do not reproduce)"
        elif s["S1"]["ok"] and s["S2"]["ok"] and s["S3"]["ok"] and s["S4"]["ok"]:
            v[name] = "SUPPORTED (S1-S4 all pass)"
        else:
            v[name] = "INCONCLUSIVE (S2 or S3 undetermined at the achieved SEM)"
        print(f"  {name}  {stmt}\n        -> {v[name]}")

    if sten:
        rs = {L: s["spacing"] for L, s in sten.items()}
        allin = all(R_SPACING_BAND[0] <= x["r"] <= R_SPACING_BAND[1]
                    for x in rs.values())
        anybad = any(x["r"] < R_SPACING_BAND[0] for x in rs.values())
        v["F6"] = ("SUPPORTED (2 <= r <= 20 at both L)" if allin else
                   "KILLED (spacing buried or far too coarse)" if anybad or
                   all(x["r"] > R_SPACING_BAND[1] for x in rs.values())
                   else "INCONCLUSIVE (band met at one L only)")
        print(f"  F6  the chosen delta_lambda is usable\n        -> {v['F6']}   "
              + ", ".join(f"L={L}: r={x['r']:.2f} ({x['verdict']})"
                          for L, x in sorted(rs.items())))
        ps = {L: s["F7"] for L, s in sten.items()}
        v["F7"] = {str(L): x["verdict"] for L, x in ps.items()}
        print(f"  F7  jaggedness survives the independent-population bootstrap\n"
              + "".join(f"        -> L={L}: p={x['p']:.4f}  {x['verdict']}\n"
                        for L, x in sorted(ps.items())), end="")
    else:
        v["F6"] = v["F7"] = "NOT EVALUATED"
    rep["targets"] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASK)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    pops, n_new = load(a.task)
    print("=" * 78)
    print("  TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA — combined analysis")
    print("  frozen rules: analysis_spec.yaml + SMOOTHNESS_CRITERION.md")
    print("=" * 78)
    print(f"  populations loaded: {sum(len(v) for v in pops.values())} "
          f"({n_new} new, {sum(len(v) for v in pops.values()) - n_new} frozen "
          f"predecessor)")
    print(f"  cells: {len(pops)}")
    print("  every error bar below is ACROSS INDEPENDENT POPULATIONS; within-clone")
    print("  spread appears only as VIF/N_eff and is never a standard error.")
    rep = {}
    tab, deltas = section_A(pops, rng, rep)
    gam = section_B(tab, rng, rep) if tab else {}
    sten = section_C(pops, rng, rep)
    section_D(tab, deltas, sten, rep)
    targets(deltas, gam, sten, rep)
    dest = os.path.join(a.task, "COMBINED_RESULTS.json")
    json.dump(rep, open(dest, "w"), indent=1, default=float)
    print("\n" + "=" * 78)
    print(f"  wrote {dest}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
