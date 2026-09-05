#!/usr/bin/env python3
"""Analysis for TASK-2026-09-06-NC-ZETA-MOCKPROD.

Implements ANALYSIS_SPEC.yaml and nothing else. The spec is FROZEN; if this
script needs to do something the spec does not describe, the answer is a child
task (research/tools/child_task.py propose), not an edit to either file.

    .venv/bin/python3 analysis/mockprod_analysis.py [--task DIR]

Read-only with respect to everything except analysis/ outputs. Runs on whatever
has returned: missing cells are FLAGGED AND COUNTED, never imputed, and a rung
with missing populations classifies INCONCLUSIVE rather than being skipped.

Writes MOCKPROD_ANALYSIS.txt, MOCKPROD_RESULTS.json and analysis/figures/.
"""
from __future__ import annotations
import argparse, collections, glob, json, math, os, sys
from statistics import NormalDist

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))

# ---------------------------------------------------------------------------
# The frozen constants. Duplicated from ANALYSIS_SPEC.yaml deliberately: the
# spec is the authority and check_spec() below fails if the two ever disagree.
# ---------------------------------------------------------------------------
GRID = {0.10: [0.040, 0.055, 0.070, 0.085, 0.100, 0.115, 0.130, 0.145, 0.160],
        0.20: [0.080, 0.1025, 0.125, 0.1475, 0.170, 0.1925, 0.215, 0.2375, 0.260],
        0.70: [0.250, 0.284, 0.318, 0.351, 0.385, 0.419, 0.453, 0.486, 0.520]}
LS = [32, 48, 64]
NCS = [128, 256, 512, 1024, 2048]
RUNGS = [(128, 256), (256, 512), (512, 1024), (1024, 2048)]
PRIMARY = (48, 64)
SUPPORTING = [(32, 48), (32, 64)]
R_DESIGN = 16
NBOOT = 10000
SEED = 20260906          # fixed: the bootstrap is reproducible, not stochastic

# The zeta = 0.35 REFERENCE ROW. Quoted, never recomputed, never pooled.
REF035 = dict(source="TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION",
              pair="L48-L64", crossing=0.23691, N_c=1024, R=24,
              grid_points=17, outcome="INTERIOR",
              matched=False,
              why_not_matched="different R (24 vs 16) and a different lambda grid")

NOT_CONVERGENCE = (
    "ROUGHLY STABLE means the curves overlap within the statistical resolution "
    "of THIS survey at R = 16. It does NOT mean finite-N_c bias is bounded, "
    "small, or converged: the N_c -> infinity target is unknown, so what is "
    "measured here is DRIFT, never bias.")


# ---------------------------------------------------------------------------
def load(task):
    """Every returned population, keyed by cell. Nothing is imputed."""
    cells = collections.defaultdict(list)
    nbad, nfiles = 0, 0
    for p in glob.glob(os.path.join(task, "**", "results", "*.json"), recursive=True):
        # scratch/ is synthetic and throwaway by construction. Without this the
        # analysis would silently ingest fabricated populations sitting in the
        # task's own scratch directory and report them as measurements. Found
        # when the red team's scratch data made preflight P14 report 27 false
        # duplicates; the same glob is used here, so the same file would have
        # entered the real curves.
        if os.sep + "scratch" + os.sep in p:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            nbad += 1
            continue
        nfiles += 1
        if d.get("status") != "ok":
            nbad += 1
            continue
        k = (float(d["zeta"]), int(d["L"]), int(d["N_c"]),
             round(float(d["lam"]), 6), float(d["dtau_mult"]))
        cells[k].append(d)
    # Deduplicate by seed. A cell must contain R INDEPENDENT populations; two
    # files carrying the same seed are the same population, and counting both
    # would shrink the SEM without adding information. Requeued and re-run packs
    # make duplicate files a normal occurrence, not an exotic one.
    ndup = 0
    for k, v in cells.items():
        seen, keep = set(), []
        for d in sorted(v, key=lambda x: x.get("seed", 0)):
            sd = d.get("seed")
            if sd in seen:
                ndup += 1
                continue
            seen.add(sd)
            keep.append(d)
        cells[k] = keep
    return cells, nfiles, nbad, ndup


def cellstat(pops, R=None):
    """Across-population mean and SEM. DEC-MASTER-METRIC-001: the uncertainty
    is over independent populations and nothing else."""
    if not pops:
        return None
    pops = sorted(pops, key=lambda d: d["seed"])
    if R is not None:
        pops = pops[:R]                      # matched R, in SEED order
    m = np.array([p["cmi_weighted_mean"] for p in pops], float)
    n = len(m)
    if n < 2:
        return dict(n=n, mean=float(m[0]), sem=float("nan"), vif=float("nan"))
    within = np.array([p["cmi_within_var"] for p in pops], float)
    across = float(np.var(m, ddof=1))
    Nc = int(pops[0]["N_c"])
    vif = float(across * Nc / np.mean(within)) if np.mean(within) > 0 else float("nan")
    return dict(n=n, mean=float(np.mean(m)), sem=float(np.std(m, ddof=1) / math.sqrt(n)),
                vif=vif, values=m)


def curve(cells, z, L, nc, R=R_DESIGN):
    """The nine-point CMI(lambda) curve. Missing points stay None."""
    out = []
    for lam in GRID[z]:
        s = cellstat(cells.get((z, L, nc, round(lam, 6), 6.0), []), R)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
def sign_changes(lams, D):
    """Indices i where D[i] and D[i+1] straddle zero. No smoothing."""
    idx = []
    for i in range(len(D) - 1):
        a, b = D[i], D[i + 1]
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0 or (a < 0) != (b < 0):
            idx.append(i)
    return idx


def interp(lams, D, i):
    a, b = D[i], D[i + 1]
    if a == b:
        return lams[i]
    return lams[i] + (lams[i + 1] - lams[i]) * (-a) / (b - a)


def classify_crossing(lams, D, boot_locs, boot_counts):
    """The XV1 validity rule and the six frozen outcome classes."""
    idx = sign_changes(lams, D)
    step = np.median(np.diff(lams))
    res = dict(n_sign_changes=len(idx), step=float(step),
               boot_count_hist={str(k): int(v) for k, v in
                                sorted(collections.Counter(boot_counts).items())})
    if len(idx) > 1:
        res.update(outcome="MULTIPLE", location=None,
                   locations=[interp(lams, D, i) for i in idx])
        return res
    if len(idx) == 0:
        finite = [d for d in D if d is not None and np.isfinite(d)]
        if not finite:
            res.update(outcome="NONE", location=None)
            return res
        # Which end is the mass piling against? Use |D| smallest end.
        lo_end, hi_end = abs(finite[0]), abs(finite[-1])
        res.update(outcome="BELOW_GRID" if lo_end < hi_end else "ABOVE_GRID",
                   location=None,
                   nearest_zero_end="low" if lo_end < hi_end else "high")
        return res
    i = idx[0]
    loc = interp(lams, D, i)
    res["location_raw"] = float(loc)
    res["interval_index"] = i
    if boot_locs:
        b = np.array(boot_locs, float)
        ci = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    else:
        ci = (float("nan"), float("nan"))
    res["bootstrap_ci"] = ci
    res["fraction_exactly_one"] = float(np.mean(np.array(boot_counts) == 1)) \
        if len(boot_counts) else float("nan")

    end_interval = (i == 0 or i == len(lams) - 2)
    ci_clear = (np.isfinite(ci[0]) and ci[0] - lams[0] >= step / 2
                and lams[-1] - ci[1] >= step / 2)
    if end_interval or not ci_clear:
        res.update(outcome="ENDPOINT_INDUCED", location=None,
                   why=("sign change is in an end interval" if end_interval
                        else "bootstrap interval is not clear of both grid ends"))
    else:
        res.update(outcome="INTERIOR", location=float(loc))
    return res


def crossing(cA, cB, lams, rng):
    """Crossing of CMI_B - CMI_A, with the frozen bootstrap."""
    D = [None if (a is None or b is None) else b["mean"] - a["mean"]
         for a, b in zip(cA, cB)]
    have = [j for j in range(len(lams)) if D[j] is not None]
    if len(have) < 3:
        return dict(outcome="NONE", location=None, n_sign_changes=0,
                    note="fewer than 3 usable lambda points")
    boot_locs, boot_counts = [], []
    for _ in range(NBOOT):
        Db = []
        for a, b in zip(cA, cB):
            if a is None or b is None or "values" not in a or "values" not in b:
                Db.append(None); continue
            ra = rng.choice(a["values"], size=len(a["values"]), replace=True)
            rb = rng.choice(b["values"], size=len(b["values"]), replace=True)
            Db.append(float(np.mean(rb) - np.mean(ra)))
        ix = sign_changes(lams, Db)
        boot_counts.append(len(ix))
        if len(ix) == 1:
            boot_locs.append(interp(lams, Db, ix[0]))
    out = classify_crossing(lams, D, boot_locs, boot_counts)
    out["D"] = [None if d is None else float(d) for d in D]
    return out


def roughness(c):
    """mean over interior points of (second difference / its SEM)^2."""
    v = []
    for i in range(1, len(c) - 1):
        a, b, d = c[i - 1], c[i], c[i + 1]
        if None in (a, b, d):
            continue
        q = a["mean"] - 2 * b["mean"] + d["mean"]
        s = math.sqrt(a["sem"] ** 2 + 4 * b["sem"] ** 2 + d["sem"] ** 2)
        if s > 0:
            v.append((q / s) ** 2)
    return float(np.mean(v)) if v else float("nan")



def smallest_adequate(evs):
    """ANALYSIS_SPEC.yaml -> recommendation_rule (amendment_2, R1).

    `evs` is [(nlo, nhi, ev_or_None), ...] in ladder order. Returns
    (pick, worst_mde_pct, inconclusive_rungs).

    The rule: the smallest N_c such that THAT RUNG AND EVERY HIGHER RUNG
    classifies ROUGHLY_STABLE. Every starting rung is examined -- the scan does
    NOT stop at the first rung that moves, because finite-N_c drift is not
    required to be monotone in N_c (FALSIFICATION_PLAN F12), and the measured
    rate ladder is itself non-monotone. An INCONCLUSIVE rung at or below the
    pick BREAKS the chain: it is not evidence of stability and may not be
    skipped over to reach a stable rung above it.
    """
    inc = [(a, b) for a, b, ev in evs if ev and ev["class"] == "INCONCLUSIVE"]
    pick, mde = None, None
    for i, (nlo, _nhi, ev) in enumerate(evs):
        tail = [e[2] for e in evs[i:]]
        if any(t is None for t in tail):
            continue
        if all(t["class"] == "ROUGHLY_STABLE" for t in tail):
            pick = nlo
            mde = max(t.get("mde_pct", float("nan")) for t in tail)
            break
    if pick is not None and any(a <= pick for a, _b in inc):
        pick, mde = None, None
    return pick, mde, inc


# ---------------------------------------------------------------------------
_ZCRIT_CACHE = {}


def z_crit(n_cmp, R=None, nsim=20000, alpha=0.01):
    """Critical value for max|z| over n_cmp comparisons, SIMULATED at the
    actual R.

    Not Phi^-1((1+(1-alpha)**(1/n))/2). That is the Sidak point for the maximum
    of n STANDARD NORMALS, and z here divides by SEMs estimated from R = 16
    populations, so each z is t-like and the maximum has a heavier tail. The
    normal point 3.451 has a true null exceedance of 3.05 %, not 1 %.
    Independently found by the red team and reproduced before this changed.

    Fixed seed: the threshold is a constant of the design, not a random draw.
    """
    R = R_DESIGN if R is None else R
    key = (int(n_cmp), int(R), int(nsim), float(alpha))
    if key in _ZCRIT_CACHE:
        return _ZCRIT_CACHE[key]
    g = np.random.default_rng(90210)
    a = g.standard_normal((nsim, n_cmp, R))
    b = g.standard_normal((nsim, n_cmp, R))
    sa = a.std(axis=2, ddof=1) / math.sqrt(R)
    sb = b.std(axis=2, ddof=1) / math.sqrt(R)
    z = np.abs(b.mean(axis=2) - a.mean(axis=2)) / np.sqrt(sa ** 2 + sb ** 2)
    v = float(np.percentile(z.max(axis=1), 100 * (1 - alpha)))
    _ZCRIT_CACHE[key] = v
    return v


# ---------------------------------------------------------------------------
def classify_rung(z, nlo, nhi, curves, xs, step):
    """The four frozen qualitative classes. INCONCLUSIVE is evaluated FIRST so
    a noisy rung can never be reported as stable by default."""
    ev = {}
    zmax, ncomp = 0.0, 0
    rel_worst = {}
    sem_comb, cmi_scale = [], []
    for L in (48, 64):
        clo, chi = curves[(nlo, L)], curves[(nhi, L)]
        # ANALYSIS_SPEC amendment_2 R2: the relative SEM is evaluated on BOTH
        # rungs and the WORST is taken. The first implementation looked only at
        # the higher rung, so an unusable lower rung could not trigger the gate.
        med_lo, med_hi = [], []
        for a, b in zip(clo, chi):
            if a is None or b is None or not np.isfinite(a["sem"]) or not np.isfinite(b["sem"]):
                continue
            s = math.sqrt(a["sem"] ** 2 + b["sem"] ** 2)
            if s > 0:
                zmax = max(zmax, abs(b["mean"] - a["mean"]) / s)
                ncomp += 1
                sem_comb.append(s)
            if abs(a["mean"]) > 0:
                med_lo.append(a["sem"] / abs(a["mean"]))
            if abs(b["mean"]) > 0:
                med_hi.append(b["sem"] / abs(b["mean"]))
                cmi_scale.append(abs(b["mean"]))
        r = [float(np.median(m)) for m in (med_lo, med_hi) if m]
        if r:
            rel_worst["L%d" % L] = max(r)
            ev["rel_sem_L%d" % L] = max(r)
    ev["max_abs_z"] = float(zmax)
    ev["n_comparable_lambda"] = ncomp // 2

    # crossing shift, only when BOTH crossings are valid under XV1
    a, b = xs.get(nlo), xs.get(nhi)
    shift = None
    if a and b and a.get("location") is not None and b.get("location") is not None:
        shift = b["location"] - a["location"]
    ev["crossing_shift"] = shift
    ev["crossing_shift_in_steps"] = None if shift is None else abs(shift) / step

    # z_crit, SIMULATED at the actual R (amendment_2 R4).
    n_cmp = max(1, ev["n_comparable_lambda"] * 2)
    zc = z_crit(n_cmp)
    ev["z_crit"] = float(zc)
    ev["z_crit_normal_sidak"] = float(
        NormalDist().inv_cdf(0.5 * (1 + 0.99 ** (1.0 / n_cmp))))
    ev["n_comparisons"] = n_cmp

    # minimum detectable effect (amendment_2 R3): the uniform per-point rung
    # shift that would reach the STILL_CHANGING threshold, as a percentage of
    # the curve. Without it, ROUGHLY_STABLE and "we could not have seen it"
    # are indistinguishable in the output.
    if sem_comb and cmi_scale:
        ev["mde_abs"] = float(zc * np.median(sem_comb))
        ev["mde_pct"] = float(100 * ev["mde_abs"] / np.median(cmi_scale))
    else:
        ev["mde_abs"] = ev["mde_pct"] = float("nan")

    rel = [v for v in rel_worst.values() if v is not None and np.isfinite(v)]
    missing = any(curves[(n, L)][j] is None or curves[(n, L)][j]["n"] < R_DESIGN
                  for n in (nlo, nhi) for L in LS for j in range(len(GRID[z])))
    # amendment_2 R2: "at L=48 OR L=64 exceeds 0.05" fires when EITHER does,
    # i.e. on the MAXIMUM. The first implementation used the minimum and so
    # required BOTH L to be unusable before firing.
    if missing or not rel or max(rel) > 0.05 or ev["n_comparable_lambda"] < 6:
        ev["class"] = "INCONCLUSIVE"
        ev["why"] = ("populations missing or short of R = %d" % R_DESIGN if missing
                     else "R = %d does not resolve this rung: worst median "
                          "SEM/|CMI| %.3f over L in {48, 64} and over both rungs, "
                          "%d of 9 lambda comparable"
                          % (R_DESIGN, max(rel) if rel else float("nan"),
                             ev["n_comparable_lambda"]))
        return ev
    s_ = ev["crossing_shift_in_steps"]
    if zmax >= 2 * zc or (s_ is not None and s_ >= 2):
        ev["class"] = "CLEARLY_TOO_SMALL"
    elif zmax >= zc or (s_ is not None and s_ >= 1):
        ev["class"] = "STILL_CHANGING"
    else:
        ev["class"] = "ROUGHLY_STABLE"
        ev["caveat"] = ("verdict rests on the curves alone; no usable crossing "
                        "at one or both rungs" if s_ is None else
                        "crossing shift %.4f = %.2f grid spacings" % (shift, s_))
        ev["not_convergence"] = NOT_CONVERGENCE
    return ev

# ---------------------------------------------------------------------------
def dtau_control(cells, out):
    """The one-cell discretisation sanity check. May conclude nothing global."""
    cell = dict(zeta=0.10, L=64, N_c=512, lam=0.100)
    legs = {}
    for dt in (3.0, 6.0, 12.0):
        s = cellstat(cells.get((0.10, 64, 512, 0.1, dt), []), R_DESIGN)
        legs[dt] = s
    out.append("")
    out.append("=" * 78)
    out.append("  DISCRETISATION CONTROL  zeta=0.10 L=64 N_c=512 lambda=0.100")
    out.append("=" * 78)
    if any(v is None for v in legs.values()):
        have = [k for k, v in legs.items() if v is not None]
        out.append("  legs returned: %s of [3.0, 6.0, 12.0] -- NOT EVALUABLE." % have)
        out.append("  The dtau_mult = 6 leg lives in M_z010_nc512; the control cannot")
        out.append("  be read until that arm has returned. This is by design.")
        return dict(cell=cell, status="incomplete", legs_present=have)
    out.append("  dtau_mult    R    mean CMI       SEM")
    for dt in (3.0, 6.0, 12.0):
        v = legs[dt]
        out.append("  %8.1f  %4d  %10.5f  %8.5f" % (dt, v["n"], v["mean"], v["sem"]))
    worst, pair = 0.0, None
    for a in (3.0, 6.0, 12.0):
        for b in (3.0, 6.0, 12.0):
            if a >= b:
                continue
            s = math.sqrt(legs[a]["sem"] ** 2 + legs[b]["sem"] ** 2)
            t = abs(legs[a]["mean"] - legs[b]["mean"]) / s if s > 0 else float("nan")
            if t > worst:
                worst, pair = t, (a, b)
    verdict = ("AN OBVIOUS dtau DEPENDENCE IS DETECTED AT THIS ONE CELL"
               if worst >= 3 else
               "no obvious dtau dependence is detected at this one cell")
    out.append("  worst pair %s: %.2f combined SEM" % (str(pair), worst))
    out.append("  -> %s" % verdict)
    out.append("  THIS LICENSES NOTHING GLOBAL. One cell, one L, one N_c, one lambda.")
    out.append("  No discretisation theorem may be inferred from it, and dtau_mult != 6")
    out.append("  rows are never pooled with the production corpus.")
    return dict(cell=cell, status="complete", worst_t=worst, pair=pair,
                verdict=verdict, legs={str(k): {kk: vv for kk, vv in v.items()
                                                if kk != "values"}
                                       for k, v in legs.items()})


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASK)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    cells, nfiles, nbad, ndup = load(a.task)

    out = []
    out.append("=" * 78)
    out.append("  TASK-2026-09-06-NC-ZETA-MOCKPROD  --  mock-production survey")
    out.append("  frozen rules: ANALYSIS_SPEC.yaml   observable: OBS-CMI-001")
    out.append("=" * 78)
    out.append("  result files read: %d   runs with status != ok or unreadable: %d"
               % (nfiles, nbad))
    out.append("  duplicate seeds discarded (same population, two files): %d" % ndup)
    out.append("  Every error bar below is ACROSS INDEPENDENT POPULATIONS.")
    out.append("  VIF is a diagnostic (DEC-MASTER-METRIC-001) and is never a")
    out.append("  standard error. R = %d, matched at every compared cell." % R_DESIGN)
    out.append("  THIS IS A SURVEY. It certifies nothing and no verdict below may")
    out.append("  be read as convergence. " + NOT_CONVERGENCE)

    results = dict(reference_zeta035=REF035, zetas={},
                   files_read=nfiles, files_rejected=nbad,
                   duplicate_seeds_discarded=ndup,
                   audit=dict(smoothing_applied=False, value_based_exclusions=0,
                              lambda_points_removed=0, populations_removed=0,
                              grid_extended=False))

    summary_rows, recommend_rows = [], []

    for z in sorted(GRID):
        lams = GRID[z]
        step = float(np.median(np.diff(lams)))
        curves = {}
        for nc in NCS:
            for L in LS:
                curves[(nc, L)] = curve(cells, z, L, nc)

        out.append("")
        out.append("=" * 78)
        out.append("  zeta = %.2f   grid %.3f .. %.3f, step %.4f, 9 points"
                   % (z, lams[0], lams[-1], step))
        out.append("=" * 78)

        zres = dict(grid=lams, step=step, rungs={}, curves={}, crossings={})

        # --- curves -----------------------------------------------------
        for nc in NCS:
            present = sum(1 for L in LS for c in curves[(nc, L)] if c is not None)
            out.append("")
            out.append("  [N_c = %d]  %d of %d cells returned" % (nc, present, 3 * 9))
            if present == 0:
                out.append("      nothing returned for this rung yet.")
                continue
            for L in LS:
                c = curves[(nc, L)]
                out.append("   L=%-3d lambda      R    mean CMI       SEM       VIF"
                           % L)
                for lam, s in zip(lams, c):
                    if s is None:
                        out.append("        %.4f    --    MISSING (flagged, not imputed)"
                                   % lam)
                    else:
                        out.append("        %.4f  %4d  %10.5f  %8.5f  %8.2f"
                                   % (lam, s["n"], s["mean"], s["sem"], s["vif"]))
                out.append("        roughness = %.3f" % roughness(c))
                zres["curves"]["nc%d_L%d" % (nc, L)] = [
                    None if s is None else dict(lam=lam, R=s["n"], mean=s["mean"],
                                                sem=s["sem"], vif=s["vif"])
                    for lam, s in zip(lams, c)]

        # --- crossings ---------------------------------------------------
        out.append("")
        out.append("  CROSSINGS.  LOCATOR QUALITY ONLY. Nothing here is")
        out.append("  lambda_c(zeta) and nothing here may become a phase boundary.")
        xs = {}
        for nc in NCS:
            for (La, Lb), tag in [(PRIMARY, "PRIMARY")] + \
                                 [(p, "supporting") for p in SUPPORTING]:
                cA, cB = curves[(nc, La)], curves[(nc, Lb)]
                if all(x is None for x in cA) or all(x is None for x in cB):
                    continue
                r = crossing(cA, cB, lams, rng)
                key = "nc%d_L%d-L%d" % (nc, La, Lb)
                zres["crossings"][key] = {k: v for k, v in r.items() if k != "D"}
                if (La, Lb) == PRIMARY:
                    xs[nc] = r
                out.append("   N_c=%-5d %s  L%d-L%d  outcome=%-16s %s"
                           % (nc, tag, La, Lb, r["outcome"],
                              ("location %.5f  boot95 [%.5f, %.5f]  frac-one %.3f"
                               % (r["location"], r["bootstrap_ci"][0],
                                  r["bootstrap_ci"][1], r["fraction_exactly_one"]))
                              if r.get("location") is not None else
                              "(no location quoted: %s)" % r.get("why", r["outcome"])))

        # --- rung classification ----------------------------------------
        out.append("")
        out.append("  N_c RUNG COMPARISON")
        for nlo, nhi in RUNGS:
            if all(c is None for L in LS for c in curves[(nlo, L)]) or \
               all(c is None for L in LS for c in curves[(nhi, L)]):
                out.append("   %5d -> %-5d  not evaluable: a rung has returned nothing"
                           % (nlo, nhi))
                continue
            ev = classify_rung(z, nlo, nhi, curves, xs, step)
            zres["rungs"]["%d->%d" % (nlo, nhi)] = ev
            out.append("   %5d -> %-5d  %-18s max|z| = %.2f (z_crit %.2f sim, n=%d) "
                   "MDE %.1f %% of CMI   %s"
                       % (nlo, nhi, ev["class"], ev["max_abs_z"],
                          ev.get("z_crit", float("nan")), ev.get("n_comparisons", 0),
                          ev.get("mde_pct", float("nan")),
                          ("crossing shift %.4f (%.2f steps)"
                           % (ev["crossing_shift"], ev["crossing_shift_in_steps"]))
                          if ev["crossing_shift"] is not None else
                          "crossing shift n/a (crossing not usable at one or both rungs)"))
            if ev["class"] == "ROUGHLY_STABLE":
                out.append("        %s" % ev.get("caveat", ""))
                out.append("        %s" % NOT_CONVERGENCE)
            if ev["class"] == "INCONCLUSIVE":
                out.append("        why: %s" % ev["why"])
            summary_rows.append((z, nhi, xs.get(nhi), ev))

        results["zetas"]["%.2f" % z] = zres

    # --- the two required tables ----------------------------------------
    out.append("")
    out.append("=" * 78)
    out.append("  SUMMARY TABLE")
    out.append("=" * 78)
    out.append("  zeta |   N_c | L48-L64 rough crossing | quality        | change from previous rung          | status")
    for z, nc, x, ev in summary_rows:
        loc = ("%.5f" % x["location"]) if x and x.get("location") is not None else "n/a"
        qual = x["outcome"] if x else "not computed"
        out.append("  %.2f | %5d | %-22s | %-14s | max|z| %-5.2f / crit %.2f, "
                   "MDE %4.1f %% | %s"
                   % (z, nc, loc, qual, ev["max_abs_z"],
                      ev.get("z_crit", float("nan")),
                      ev.get("mde_pct", float("nan")), ev["class"]))
    out.append("  0.35 |  1024 | %.5f (REFERENCE)   | %-14s | not measured here         | quoted from %s"
               % (REF035["crossing"], REF035["outcome"], REF035["source"]))
    out.append("       |       | R = 24, 17-point grid -- NOT matched to this survey's R = 16")
    out.append("       |       | and lambda grid, so it is a reference point, never a")
    out.append("       |       | row this task measured and never pooled into any statistic.")

    out.append("")
    out.append("=" * 78)
    out.append("  PRACTICAL RECOMMENDATION")
    out.append("=" * 78)
    out.append("  zeta | smallest N_c that LOOKS adequate | confidence | caveat")
    out.append("  RULE: the smallest N_c such that THAT RUNG AND EVERY HIGHER RUNG is")
    out.append("  ROUGHLY_STABLE. An INCONCLUSIVE rung breaks the chain and is never")
    out.append("  skipped over. Confidence is \"low\" at best, always.")
    for z in sorted(GRID):
        rr = results["zetas"].get("%.2f" % z, {}).get("rungs", {})
        # ANALYSIS_SPEC.yaml -> recommendation_rule (amendment_2, R1):
        # the smallest N_c such that THAT RUNG AND EVERY HIGHER RUNG classify
        # ROUGHLY_STABLE. An INCONCLUSIVE rung BREAKS the chain and may not be
        # skipped over to reach a stable rung above it.
        #
        # The first implementation was first-stable-wins, never revised by a
        # higher rung. It would have printed the cheapest N_c while the rung
        # table above it said STILL_CHANGING -- and directionally so, because
        # relative SEM falls with N_c, so the lowest rung is always the most
        # likely to read stable for want of power. Found by the red team.
        evs = [(nlo, nhi, rr.get("%d->%d" % (nlo, nhi))) for nlo, nhi in RUNGS]
        pick, conf, cav, mde = None, "none", "no rung evaluable yet", None
        if any(e[2] for e in evs):
            pick, mde, inc = smallest_adequate(evs)
            if pick is not None:
                conf = "low"
                cav = ("stable at this rung and every rung above it; worst MDE "
                       "over that chain %.1f %% of CMI" % mde)
            elif inc:
                cav = ("chain broken: rung(s) %s are INCONCLUSIVE, which is not "
                       "evidence of stability and may not be skipped over"
                       % ", ".join("%d->%d" % r for r in inc))
            else:
                top = rr.get("1024->2048")
                if top and top["class"] in ("CLEARLY_TOO_SMALL", "STILL_CHANGING"):
                    pick, conf = "> 2048", "low"
                    cav = ("curves still moving at the top of the ladder; this "
                           "survey cannot size it")
                else:
                    cav = "no rung, and no chain of rungs, classifies ROUGHLY_STABLE"
        out.append("  %.2f | %-32s | %-10s | %s"
                   % (z, str(pick), conf, cav[:150]))
        recommend_rows.append(dict(zeta=z, smallest_nc=pick, confidence=conf,
                                   caveat=cav, worst_mde_pct=mde))
    out.append("  0.35 | REFERENCE ROW -- not measured here. The predecessor's own")
    out.append("       | finding at this zeta was CONSISTENT BUT R-LIMITED, i.e. R and")
    out.append("       | not N_c was the binding budget.")
    nrung = sum(len(results["zetas"].get("%.2f" % z, {}).get("rungs", {}))
                for z in GRID)
    out.append("")
    out.append("  MULTIPLICITY. z_crit is corrected WITHIN a rung (alpha = 0.01 over")
    out.append("  the ~18 comparisons a rung makes) and NOT across rungs, because a")
    out.append("  rung verdict is a statement about that rung. Over the %d rung" % nrung)
    out.append("  classifications in this run the expected number of STILL_CHANGING")
    out.append("  verdicts arising from noise alone is about %.2f." % (0.01 * nrung))
    out.append("  z_crit is SIMULATED at R = %d, not read off a normal quantile: the" % R_DESIGN)
    out.append("  normal Sidak point has a true null exceedance of 3.05 %, not 1 %.")
    out.append("")
    out.append("  Every entry above is a QUALITATIVE, resolution-limited judgement.")
    out.append("  " + NOT_CONVERGENCE)

    results["recommendation"] = recommend_rows
    results["dtau_control"] = dtau_control(cells, out)

    out.append("")
    out.append("=" * 78)
    out.append("  HANDLING AUDIT (ANALYSIS_SPEC.yaml -> handling_of_data)")
    out.append("=" * 78)
    for k, v in results["audit"].items():
        out.append("  %-28s %s" % (k, v))
    out.append("  Missing cells are printed as MISSING above and are never imputed.")

    out.append("")
    out.append("=" * 78)
    out.append("  WHAT THIS ANALYSIS MAY NOT SAY")
    out.append("=" * 78)
    for s in ("No crossing above is lambda_c(zeta) or a finite-size estimate of it.",
              "No boundary exponent phi is measured; DISP-PHI-001 stays open.",
              "No N_c^req(zeta) is produced and none may be fitted from these numbers.",
              "No zeta = 0.35 coefficient is transferred to any other zeta.",
              "Nothing here is certified, and no N_c here is converged.",
              "Nothing here extends to L > 64 or N_c > 2048."):
        out.append("  * " + s)

    txt = "\n".join(out)
    print(txt)
    open(os.path.join(a.task, "MOCKPROD_ANALYSIS.txt"), "w").write(txt + "\n")
    json.dump(results, open(os.path.join(a.task, "MOCKPROD_RESULTS.json"), "w"),
              indent=1, default=float)
    make_figures(a.task, cells)
    return 0


# ---------------------------------------------------------------------------
def make_figures(task, cells):
    """Validation figures, NOT manuscript figures. Every point carries its
    across-population SEM."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("\n  [figures skipped: %s]" % e)
        return
    fdir = os.path.join(task, "analysis", "figures")
    os.makedirs(fdir, exist_ok=True)
    made = []
    for z in sorted(GRID):
        lams = GRID[z]
        # primary: one figure per zeta, panelled by L, N_c overlaid
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
        any_pt = False
        for ax, L in zip(axes, LS):
            for nc in NCS:
                c = curve(cells, z, L, nc)
                xs = [l for l, s in zip(lams, c) if s is not None]
                ys = [s["mean"] for s in c if s is not None]
                es = [s["sem"] for s in c if s is not None]
                if xs:
                    any_pt = True
                    ax.errorbar(xs, ys, yerr=es, marker="o", ms=3, capsize=2,
                                label="N_c=%d" % nc)
            ax.set_title("zeta=%.2f  L=%d  (T=L)" % (z, L))
            ax.set_xlabel("lambda")
        axes[0].set_ylabel("CMI  (OBS-CMI-001)")
        axes[0].legend(fontsize=7)
        fig.suptitle("SURVEY, R=16, across-population SEM. Not a certification; "
                     "no crossing here is lambda_c(zeta).", fontsize=8)
        fig.tight_layout()
        p = os.path.join(fdir, "figureA_curves_z%03d.png" % round(z * 100))
        if any_pt:
            fig.savefig(p, dpi=130); made.append(p)
        plt.close(fig)

        # difference figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
        any_pt = False
        for ax, L in zip(axes, LS):
            for nlo, nhi in RUNGS:
                a_, b_ = curve(cells, z, L, nlo), curve(cells, z, L, nhi)
                xs, ys, es = [], [], []
                for lam, u, v in zip(lams, a_, b_):
                    if u is None or v is None:
                        continue
                    xs.append(lam); ys.append(v["mean"] - u["mean"])
                    es.append(math.sqrt(u["sem"] ** 2 + v["sem"] ** 2))
                if xs:
                    any_pt = True
                    ax.errorbar(xs, ys, yerr=es, marker="s", ms=3, capsize=2,
                                label="%d->%d" % (nlo, nhi))
            ax.axhline(0, lw=0.7, color="k")
            ax.set_title("zeta=%.2f  L=%d" % (z, L)); ax.set_xlabel("lambda")
        axes[0].set_ylabel("Delta_N = CMI_2Nc - CMI_Nc")
        axes[0].legend(fontsize=7)
        fig.suptitle("Finite-N_c DRIFT (the N_c->infinity target is unknown, so "
                     "this is not bias).", fontsize=8)
        fig.tight_layout()
        p = os.path.join(fdir, "figureC_delta_z%03d.png" % round(z * 100))
        if any_pt:
            fig.savefig(p, dpi=130); made.append(p)
        plt.close(fig)
    print("\n  figures written: %s" % (", ".join(os.path.basename(m) for m in made)
                                       or "none (no data yet)"))


if __name__ == "__main__":
    sys.exit(main())
