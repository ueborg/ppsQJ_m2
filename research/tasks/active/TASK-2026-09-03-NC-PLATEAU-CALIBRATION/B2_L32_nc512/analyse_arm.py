#!/usr/bin/env python3
"""Per-arm analysis and quality control for TASK-2026-09-03-NC-PLATEAU-CALIBRATION.

    python3 analyse_arm.py [results_dir]

This reports THIS ARM ONLY: per-cell population statistics, the exclusion
accounting frozen in ../ANALYSIS_SPEC.yaml, the genealogy/VIF diagnostics, and a
split-half reproducibility check over independent populations.

It deliberately does NOT compute Delta_N, B_eff, plateau criteria, lambda
stencils or crossings. Those are cross-arm quantities -- every ladder in this
campaign mixes fresh populations with exact-compatible reused ones -- and they
belong to ../analysis/nc_plateau_analysis.py, which is the single place the
frozen criteria P1-P5, H1-H3, E1-E2 and the tolerances tau_I / tau_lambda are
evaluated.

VIF is printed as a VARIANCE-EQUIVALENCE diagnostic only. It is never a bias
diagnostic and never licenses a statement about finite-N_c drift.

Every uncertainty here is across INDEPENDENT POPULATIONS. Within-clone spread is
reported as a diagnostic and is NEVER used as the standard error of a population
mean — that is the parent programme's headline recorded lesson.
"""
import os, sys, json, glob, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")

rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(res, "*.json")))]
print(f"loaded {len(rows)} result file(s) from {res}")
if not rows:
    sys.exit("nothing to analyse yet.")


def key(r):
    # dtau_mult is part of the cell identity: campaign E deliberately varies it,
    # and pooling two discretisations into one cell would be the single easiest
    # way to manufacture a wrong answer here.
    return (int(r["L"]), float(r["T"]), float(r["zeta"]),
            round(float(r["lam"]), 6), int(r["N_c"]), float(r["dtau_mult"]))


bad = [r for r in rows if r.get("status") not in (None, "ok")]
ok = [r for r in rows if r.get("status") in (None, "ok")]
print(f"status != 'ok': {len(bad)} of {len(rows)} "
      f"({100.0 * len(bad) / len(rows):.1f} %)"
      f"{'   ** above the frozen 5 % cell-exclusion threshold **' if len(bad) > 0.05 * len(rows) else ''}")

cells = {}
for r in ok:
    cells.setdefault(key(r), []).append(r)

rng = np.random.default_rng(20260903)
out = []
for k in sorted(cells):
    rs = cells[k]
    m = np.array([float(r["cmi_weighted_mean"]) for r in rs])
    s2 = np.array([float(r["cmi_within_var"]) for r in rs])
    nc = k[4]
    dm = k[5]
    V = float(np.var(m, ddof=1))
    s2m = float(np.mean(s2))
    sem = math.sqrt(V / m.size)
    vif = V * nc / s2m if s2m > 0 else float("nan")
    neff = s2m / V if V > 0 else float("nan")

    nonfin = sum(int(r.get("n_nonfinite", 0)) for r in rs)
    clones = sum(int(r["N_c"]) for r in rs)
    fbk = sum(int(r.get("brentq_fallbacks", 0)) for r in rs)
    anc = np.array([int(r.get("n_distinct_anc_final", 0)) for r in rs])
    essf = np.array([float(r.get("ess_frac_mean", np.nan)) for r in rs])
    wall = np.array([float(r.get("wall_s", np.nan)) for r in rs])

    # split-half over independent populations: two disjoint halves of the R
    # replicates must agree within their own joint error. This is a
    # reproducibility check, not a convergence claim.
    if m.size >= 4:
        idx = rng.permutation(m.size)
        a, b = m[idx[:m.size // 2]], m[idx[m.size // 2:2 * (m.size // 2)]]
        dsplit = float(a.mean() - b.mean())
        sd = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    else:
        # each half needs >= 2 populations for its own variance to exist. Every
        # cell here has R = 24, so this branch is only reachable in a smoke run
        # -- but returning NaN silently would look like a diagnostic result
        # rather than an unmet precondition.
        dsplit, sd = float("nan"), float("nan")

    K = [int(r["n_steps"]) for r in rs]
    print(f"\n[L={k[0]} T={k[1]:g} zeta={k[2]:g} lambda={k[3]:g} N_c={nc} "
          f"dtau_mult={dm:g} K={K[0]}]")
    assert len(set(K)) == 1, f"n_steps not constant within a cell: {sorted(set(K))}"
    print(f"  R (independent populations)   {m.size}")
    print(f"  mean CMI                      {m.mean():.5f}")
    print(f"  across-population SEM         {sem:.5f}      <- the only valid error bar")
    print(f"  across-population variance    {V:.4e}")
    print(f"  mean within-clone variance    {s2m:.4e}      (diagnostic only)")
    print(f"  VIF = V*N_c/s2_within         {vif:.2f}")
    print(f"  N_eff = s2_within/V           {neff:.2f}")
    print(f"  non-finite clones             {nonfin} of {clones} "
          f"({100.0 * nonfin / clones:.3f} %)"
          f"{'   ** cell SUSPECT: above the frozen 1 % **' if nonfin > 0.01 * clones else ''}")
    print(f"  brentq fallbacks (reported,   {fbk}")
    print(f"    never an exclusion)")
    print(f"  n_distinct_ancestors final    min {anc.min()} med {int(np.median(anc))} "
          f"max {anc.max()} of {nc}")
    print(f"  ess_frac_mean                 min {np.nanmin(essf):.4f} "
          f"med {np.nanmedian(essf):.4f}")
    print(f"    (genealogy may collapse completely without implying an")
    print(f"     information ceiling — this is a diagnostic, not a verdict)")
    print(f"  wall_s                        med {np.nanmedian(wall):.1f} "
          f"min {np.nanmin(wall):.1f} max {np.nanmax(wall):.1f}")
    if math.isnan(sd):
        print(f"  split-half difference         not computed (R = {m.size} < 4)")
    else:
        print(f"  split-half difference         {dsplit:+.5f} +- {sd:.5f} "
              f"({abs(dsplit) / sd if sd > 0 else float('inf'):.2f} sigma)")
    lwv = [float(r["logw_carry_var_final"]) for r in rs if "logw_carry_var_final" in r]
    if lwv:
        print(f"  Var(log carried weight)       med {np.median(lwv):.4f} "
              f"min {min(lwv):.4f} max {max(lwv):.4f}")
        print(f"    (recorded in 0 % of the pre-existing production corpus;")
        print(f"     see ../INSTRUMENTATION.md)")
    out.append(dict(L=k[0], T=k[1], zeta=k[2], lam=k[3], N_c=nc, dtau_mult=dm,
                    K=K[0], logw_carry_var_med=(float(np.median(lwv)) if lwv else None),
                    R=int(m.size),
                    mean=float(m.mean()), sem=sem, var_across=V,
                    within_mean=s2m, vif=vif, n_eff=neff,
                    n_nonfinite=nonfin, n_clones=clones,
                    brentq_fallbacks=fbk,
                    anc_min=int(anc.min()), anc_med=float(np.median(anc)),
                    wall_med=float(np.nanmedian(wall)),
                    split_half_diff=dsplit, split_half_sem=sd,
                    pop_means=[float(x) for x in m],
                    within_vars=[float(x) for x in s2]))

dest = os.path.join(HERE, "arm_summary.json")
json.dump(out, open(dest, "w"), indent=1)
print(f"\nwrote {dest}")
print("Cross-arm quantities -- Delta_N, B_eff, the plateau criteria P1-P5, the")
print("additive/multiplicative tests H1-H3, the discretisation hypotheses E1-E2,")
print("the locator convergence diagnostic and every crossing statistic -- are")
print("evaluated ONLY by ../analysis/nc_plateau_analysis.py.")
