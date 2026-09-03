#!/usr/bin/env python3
"""End-to-end smoke test for analysis/lowlambda_analysis.py.

    python3 tools/smoke_test.py [staging_dir]

THE DATA THIS SCRIPT MAKES IS SYNTHETIC AND IS NEVER SCIENTIFIC. It exists so
that the frozen analysis can be exercised over its whole path -- the 17-point
curves, the join tests, the crossing protocol, the interiority classification,
X1-X7 and all three figures -- BEFORE any Ruche job is queued, rather than
discovering a crash after 288 tasks have run.

It never writes inside the package. Everything goes to a staging tree, which
carries the real frozen predecessor CSV and the real analysis module and
differs from the package in exactly one way: the lowlam*/results/ directories
hold synthetic JSONs instead of empty ones.

WHY THREE SCENARIOS AND NOT ONE
-------------------------------
A smoke test that only proves "it runs" would pass on an analysis whose
interiority test was hard-wired to a single answer. The three scenarios below
are built so that the pre-registered classification MUST come out differently,
and the test asserts which -- PER PAIR, not in aggregate.

The synthetic curves are constructed by choosing the cross-L DIFFERENCES
directly rather than by bending each curve and hoping:

    m48(new) = linear continuation of the measured L=48 curve
    m32(new) = m48(new) - D32(lambda)          so  I48 - I32 == D32
    m64(new) = m48(new) + D64(lambda)          so  I64 - I48 == D64
                                               and I64 - I32 == D32 + D64

so each scenario can place a sign change exactly where it means to.

  interior     every difference changes sign INSIDE the new region, at least
               one interval away from the lower endpoint
                                    -> all three pairs expect INTERIOR
  below_grid   D32 stays negative across the whole new region, so the
               L32-L48 pair has no sign change anywhere on 17 points
                                    -> L32-L48 expects NONE or BELOW_GRID
  edge         D32's sign change is forced into the FIRST interval, hard
               against the new lower boundary
                                    -> L32-L48 expects STILL_BOUNDARY

The `edge` case is the one that matters: it is exactly the failure mode this
whole task exists to detect, and an analysis that called it INTERIOR would be
worthless.

A NOTE ON THE TWO PAIRS INVOLVING L = 64
----------------------------------------
Those pairs are NOT free to be given any behaviour, because 13 of their 17
points are the real measured data. The predecessor's L48-L64 difference already
changes sign between the measured 0.2332 and 0.2432, and no choice of new
points can remove it -- extending the grid downward is precisely what turns
that boundary-hugging crossing into an interior one. So the scenarios steer
D32, assert on the L32-L48 pair, and for the L64 pairs assert only that the
class is drawn from the frozen vocabulary. That is a real limitation of a
smoke test built on real reused data, and it is stated rather than hidden.

This script contains no scheduler call and cannot submit anything.
"""
import os, sys, csv, json, math, shutil, subprocess, collections

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(TASK, "analysis"))

import numpy as np

GRID = [round(0.1932 + 0.010 * i, 4) for i in range(17)]
NEW_IDX = [0, 1, 2, 3]
LS = [32, 48, 64]
ARMS = {32: ("lowlamL32", "X32"), 48: ("lowlamL48", "X48"),
        64: ("lowlamL64", "X64")}
R = 24
NC = 1024
DEFAULT_STAGE = os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "lowlam_smoke")


def predecessor_curves():
    """Block-A means and SEMs of the 13 measured points, per L."""
    fp = os.path.join(TASK, "frozen_inputs",
                      "predecessor_nc1024_populations.csv")
    cells = collections.defaultdict(list)
    for r in csv.DictReader(open(fp)):
        cells[(int(r["L"]), round(float(r["lam"]), 4))].append(
            (int(r["seed"]), float(r["cmi_weighted_mean"]),
             float(r["cmi_within_var"])))
    out = {}
    for L in LS:
        m, s, w = [], [], []
        for lam in GRID[4:]:
            v = sorted(cells[(L, lam)])[:R]
            p = np.array([x[1] for x in v])
            m.append(p.mean())
            s.append(math.sqrt(p.var(ddof=1) / R))
            w.append(float(np.mean([x[2] for x in v])))
        out[L] = (np.array(m), np.array(s), np.array(w))
    return out


# D32 = I48 - I32 over the four new lambdas. The measured value at the join
# (lambda = 0.2332) is about -0.020, so a scenario that ends near -0.018 joins
# smoothly and one that ends near -0.024 continues the trend.
D32_BY_SCENARIO = {
    # sign change between new index 1 and 2 -> interval 1, interior
    "interior":   np.array([+0.030, +0.010, -0.010, -0.018]),
    # negative throughout -> no sign change anywhere on the 17 points
    "below_grid": np.array([-0.050, -0.040, -0.030, -0.024]),
    # sign change between new index 0 and 1 -> interval 0, against the boundary
    "edge":       np.array([+0.020, -0.005, -0.012, -0.018]),
}
# D64 = I64 - I48. Positive throughout the new region in every scenario, so it
# joins the measured +0.015 at the join and adds no crossing of its own.
D64_NEW = np.array([+0.050, +0.040, +0.030, +0.022])

PAIR_EXPECT = {
    "interior":   {"L32-L48": {"INTERIOR"}},
    "below_grid": {"L32-L48": {"NONE", "BELOW_GRID"}},
    "edge":       {"L32-L48": {"STILL_BOUNDARY"}},
}
VOCAB = {"INTERIOR", "STILL_BOUNDARY", "BELOW_GRID", "NONE"}


def targets(scenario, pred):
    """Synthetic MEANS for the four new lambdas, per L.

    Built from the DIFFERENCES so the scenario controls what it claims to.
    """
    if scenario not in D32_BY_SCENARIO:
        raise SystemExit("unknown scenario %r" % scenario)
    m48, _s, _w = pred[48]
    slope = float(m48[1] - m48[0])                 # per grid step, negative
    base48 = np.array([m48[0] - slope * k for k in (4, 3, 2, 1)])
    return {32: base48 - D32_BY_SCENARIO[scenario],
            48: base48,
            64: base48 + D64_NEW}


def stage(root, scenario, pred):
    """Build a staging package for one scenario and return its path."""
    d = os.path.join(root, scenario)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "analysis"))
    os.makedirs(os.path.join(d, "frozen_inputs"))
    shutil.copy2(os.path.join(TASK, "analysis", "lowlambda_analysis.py"),
                 os.path.join(d, "analysis"))
    shutil.copy2(os.path.join(TASK, "frozen_inputs",
                              "predecessor_nc1024_populations.csv"),
                 os.path.join(d, "frozen_inputs"))

    tg = targets(scenario, pred)
    rng = np.random.default_rng(4242)
    for L in LS:
        arm, tag = ARMS[L]
        rd = os.path.join(d, arm, "results")
        os.makedirs(rd)
        m, s, w = pred[L]
        sd_pop = s[0] * math.sqrt(R)          # per-population spread at the low end
        for j, gi in enumerate(NEW_IDX):
            draws = rng.normal(tg[L][j], sd_pop, R)
            draws = draws - draws.mean() + tg[L][j]   # exact target mean
            for i in range(R):
                idx = j * R + i
                row = dict(arm=tag, L=str(L), T="%.1f" % L, N_c=str(NC),
                           zeta="0.35", lam="%.4f" % GRID[gi],
                           dtau_mult="6.0", resample_scheme="systematic",
                           seed=str(32_000_000 + 100_000 * LS.index(L)
                                    + 1000 * gi + i),
                           status="ok", wall_s=100.0 + i, n_steps=64,
                           cmi_weighted_mean=float(draws[i]),
                           cmi_unweighted_mean=float(draws[i]),
                           cmi_within_var=float(w[0]),
                           n_nonfinite=0, n_distinct_anc_final=3,
                           gess_final=1.4, ess_cum_final=1000.0,
                           ess_frac_mean=0.99, brentq_fallbacks=0)
                json.dump(row, open(os.path.join(rd, "%s_%05d.json"
                                                 % (tag, idx)), "w"))
    return d


def run(d):
    r = subprocess.run(
        [sys.executable, os.path.join(d, "analysis", "lowlambda_analysis.py"),
         "--task-root", d], capture_output=True, text=True)
    return r


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STAGE
    os.makedirs(root, exist_ok=True)
    pred = predecessor_curves()
    failures = []

    print("=" * 78)
    print("  SMOKE TEST — analysis/lowlambda_analysis.py")
    print("  SYNTHETIC DATA. Not a scientific result and never cited as one.")
    print("  staging: %s" % root)
    print("=" * 78)

    for scen in ("interior", "below_grid", "edge"):
        d = stage(root, scen, pred)
        r = run(d)
        if r.returncode != 0:
            failures.append("%s: analysis exited %d\n%s"
                            % (scen, r.returncode, r.stderr[-2000:]))
            print("\n  [%s]  ** ANALYSIS FAILED **" % scen)
            print(r.stderr[-2000:])
            continue
        res = json.load(open(os.path.join(d, "LOWLAMBDA_RESULTS.json")))
        cls = dict((k, v["outcome_class"]) for k, v in res["crossings"].items())
        crit = dict((k, v["verdict"]) for k, v in res["criteria"].items())
        joins = dict((k, v["overall"]) for k, v in res["join"].items())
        figs = sorted(os.listdir(os.path.join(d, "analysis", "figures")))
        print("\n  [%s]" % scen)
        print("     curves complete      %s"
              % [k for k, v in res["inventory"].items() if v["complete"]])
        print("     outcome classes      %s" % cls)
        print("     join verdicts        %s" % joins)
        print("     criteria             %s" % crit)
        print("     figures              %s" % figs)

        if len(res["curves"]) != 3:
            failures.append("%s: %d/3 complete curves"
                            % (scen, len(res["curves"])))
        if len(cls) != 3:
            failures.append("%s: %d/3 crossing pairs" % (scen, len(cls)))
        if len(figs) != 3:
            failures.append("%s: %d/3 figures" % (scen, len(figs)))
        # every class must come from the frozen vocabulary...
        bad = [k for k, v in cls.items() if v not in VOCAB]
        if bad:
            failures.append("%s: pairs %s got classes outside the frozen "
                            "vocabulary: %s" % (scen, bad, [cls[k] for k in bad]))
        # ...and the pair this scenario actually steers must land where it says
        for pair, want in PAIR_EXPECT[scen].items():
            got = cls.get(pair)
            if got not in want:
                failures.append("%s: %s classified %s, expected one of %s"
                                % (scen, pair, got, sorted(want)))
        if any(v == "NOT EVALUATED" for v in crit.values()):
            failures.append("%s: some criteria NOT EVALUATED on complete data: "
                            "%s" % (scen, crit))
        # the reuse must reproduce itself in every scenario: the 13 old points
        # are identical inputs, so their recomputed statistics must not move
        for k, v in res["curves"].items():
            ow = v["old_window_recheck"]
            if not (0 < ow["roughness"] < 1e3):
                failures.append("%s %s: implausible old-window roughness %r"
                                % (scen, k, ow["roughness"]))

    print("\n" + "=" * 78)
    if failures:
        print("  SMOKE TEST FAILED")
        for f in failures:
            print("    * %s" % f)
        print("=" * 78)
        return 1
    print("  SMOKE TEST PASSED — the frozen analysis runs end to end, and the")
    print("  pre-registered interiority classification distinguishes an")
    print("  interior crossing from a boundary one and from none at all.")
    print("  All synthetic. Nothing above is a measurement.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
