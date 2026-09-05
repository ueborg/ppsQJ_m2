#!/usr/bin/env python3
"""Smoke test for analysis/mockprod_analysis.py.

An analysis whose classifications cannot be seen to distinguish the cases they
name is worthless, so each case below is constructed to land in a KNOWN class
and the test fails if the analysis says something else. Several cases are
constructed to FAIL on purpose -- INCONCLUSIVE, ENDPOINT_INDUCED, ABOVE_GRID --
because a classifier that only ever returns the good answer has not been shown
to be a classifier.

Synthetic populations are written into a temporary directory. Nothing in the
task directory is touched and no real datum is involved.

    .venv/bin/python3 tools/smoke_test.py
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(TASK, "analysis"))
import mockprod_analysis as A          # noqa: E402

R = A.R_DESIGN


def write_cell(root, arm, z, L, nc, lam, mean, noise, rng, dtau=6.0, R_=R):
    d = os.path.join(root, arm, "results")
    os.makedirs(d, exist_ok=True)
    for i in range(R_):
        v = float(mean + rng.normal(0.0, noise))
        json.dump(dict(arm=arm, L=L, T=float(L), N_c=nc, zeta=z, lam=lam,
                       dtau_mult=dtau, resample_scheme="systematic",
                       seed=int(abs(hash((arm, z, L, nc, round(lam, 6), dtau, i)))
                                % 10 ** 9),
                       status="ok", wall_s=1.0, n_steps=10,
                       cmi_weighted_mean=v, cmi_unweighted_mean=v,
                       cmi_within_var=max(noise, 1e-9) ** 2 * nc,
                       n_nonfinite=0, n_distinct_anc_final=nc, gess_final=1.0,
                       ess_cum_final=1.0, ess_frac_mean=0.99, brentq_fallbacks=0),
                  open(os.path.join(d, "%s_%05d_%s.json"
                                    % (arm, i, str(abs(hash((z, L, nc, lam, dtau))))[:8])),
                       "w"))


def build(root, spec, rng):
    """spec: (zeta) -> (nc, L, lam) -> (mean, noise). Curves are built from a
    simple model whose L48-L64 crossing sits exactly where we want it."""
    for z, fn in spec.items():
        for nc in A.NCS:
            for L in A.LS:
                for lam in A.GRID[z]:
                    m, s = fn(nc, L, lam)
                    if m is None:
                        continue
                    write_cell(root, "S_z%03d_nc%d" % (round(z * 100), nc),
                               z, L, nc, lam, m, s, rng)


def curves_from(root, z, nc, L):
    cells, _n, _b, _d = A.load(root)
    return A.curve(cells, z, L, nc)


def run_case(name, z, fn, seed, expect_class=None, expect_outcome=None,
             rung=(512, 1024)):
    """Each case gets its OWN seeded generator, so a change to one case cannot
    silently move another by advancing a shared stream."""
    rng = np.random.default_rng(seed)
    root = tempfile.mkdtemp(prefix="smoke_")
    try:
        build(root, {z: fn}, rng)
        cells, _n, _b, _d = A.load(root)
        lams = A.GRID[z]
        step = float(np.median(np.diff(lams)))
        curves = {(nc, L): A.curve(cells, z, L, nc) for nc in A.NCS for L in A.LS}
        xs = {}
        for nc in A.NCS:
            xs[nc] = A.crossing(curves[(nc, 48)], curves[(nc, 64)], lams,
                                np.random.default_rng(7))
        ev = A.classify_rung(z, rung[0], rung[1], curves, xs, step)
        got_class = ev["class"]
        got_outcome = xs[rung[1]]["outcome"]
        okc = expect_class is None or got_class == expect_class
        oko = expect_outcome is None or got_outcome == expect_outcome
        status = "ok  " if (okc and oko) else "FAIL"
        print("  %s %-34s class=%-18s crossing=%-17s%s"
              % (status, name, got_class, got_outcome,
                 "" if (okc and oko) else
                 "   expected class=%s crossing=%s" % (expect_class, expect_outcome)))
        return okc and oko
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    print("=" * 78)
    print("  SMOKE TEST  analysis/mockprod_analysis.py")
    print("  Synthetic data only. Cases are built to land in a KNOWN class, and")
    print("  several are built to FAIL on purpose.")
    print("=" * 78)
    z = 0.20
    lams = A.GRID[z]
    XC = lams[4]            # a crossing at the middle grid point
    XLOW = lams[0] + 0.2 * (lams[1] - lams[0])     # inside the FIRST interval

    def base(lam):
        return 1.0 - 2.0 * (lam - lams[0])

    def make(slopeL, xcross, nc_shift, noise):
        """CMI = base(lam) + slopeL*(L-48)/16*(lam-xcross) + nc_shift(nc)."""
        def f(nc, L, lam):
            return (base(lam) + slopeL * (L - 48) / 16.0 * (lam - xcross)
                    + nc_shift(nc), noise)
        return f

    results = []

    # S1  a large rung shift -> CLEARLY TOO SMALL
    results.append(run_case(
        "S1 large rung shift", z,
        make(1.0, XC, lambda nc: 0.05 if nc <= 512 else 0.0, 0.004),
        101, expect_class="CLEARLY_TOO_SMALL"))

    # S2  a moderate rung shift -> STILL CHANGING.
    #     SIZING, and it is not obvious. max|z| is a maximum over 18
    #     comparisons and a UNIFORM rung offset gives all 18 the same
    #     expectation, so max|z| sits roughly 2.3 above the per-point |z|.
    #     The band between z_crit = 3.45 and 2*z_crit = 6.90 is therefore a
    #     band in per-point z of about [1.2, 4.6], not [3.45, 6.90].
    #     Calibrated across five seeds at noise = 0.004, R = 16:
    #         shift 0.0020 -> max|z| 2.85-4.99, class flips STABLE/CHANGING
    #         shift 0.0025 -> max|z| 3.26-5.45, still flips
    #         shift 0.0030 -> max|z| 3.67-5.91, STILL_CHANGING at all five
    #         shift 0.0035 -> max|z| 4.01-6.37, STILL_CHANGING at all five
    #     0.0030 is adopted. Earlier attempts at 0.0045 and 0.006 both came
    #     out CLEARLY_TOO_SMALL, which is the fact worth recording here: the
    #     max statistic is far more sensitive to a uniform offset than the
    #     per-point z suggests.
    results.append(run_case(
        "S2 moderate rung shift", z,
        make(1.0, XC, lambda nc: 0.0030 if nc <= 512 else 0.0, 0.004),
        102, expect_class="STILL_CHANGING"))

    # S3  no rung shift, small noise -> ROUGHLY STABLE
    results.append(run_case(
        "S3 no shift, small noise", z,
        make(1.0, XC, lambda nc: 0.0, 0.004),
        103, expect_class="ROUGHLY_STABLE"))

    # S4  no rung shift, huge noise -> INCONCLUSIVE, not "stable"
    #     This is the case that matters: a noisy rung must NOT read as stable.
    results.append(run_case(
        "S4 no shift, huge noise", z,
        make(1.0, XC, lambda nc: 0.0, 0.35),
        104, expect_class="INCONCLUSIVE"))

    # S5  crossing forced into the FIRST interval -> ENDPOINT_INDUCED
    results.append(run_case(
        "S5 crossing in first interval", z,
        make(1.0, XLOW, lambda nc: 0.0, 0.002),
        105, expect_outcome="ENDPOINT_INDUCED"))

    # S6  crossing pushed above the grid -> ABOVE_GRID, no location quoted
    results.append(run_case(
        "S6 crossing above the grid", z,
        make(1.0, lams[-1] + 5 * (lams[1] - lams[0]), lambda nc: 0.0, 0.002),
        109, expect_outcome="ABOVE_GRID"))

    # S7  clean interior crossing, no rung shift
    results.append(run_case(
        "S7 clean interior crossing", z,
        make(1.0, XC, lambda nc: 0.0, 0.002),
        106, expect_class="ROUGHLY_STABLE", expect_outcome="INTERIOR"))

    # S8  a rung with populations MISSING must be INCONCLUSIVE, never stable
    def missing(nc, L, lam):
        if nc == 1024 and abs(lam - lams[3]) < 1e-9:
            return None, None
        return base(lam) + 1.0 * (L - 48) / 16.0 * (lam - XC), 0.002
    results.append(run_case(
        "S8 missing populations", z, missing, 107,
        expect_class="INCONCLUSIVE"))

    # S9  a fabricated population sitting in scratch/ must be INVISIBLE to the
    #     analysis loader. This is not hypothetical: the red team's own scratch
    #     data made preflight P14 report 27 false duplicates, and the analysis
    #     loader used the same glob, so the same files would have entered the
    #     real curves as measurements.
    root = tempfile.mkdtemp(prefix="smoke_")
    try:
        r9 = np.random.default_rng(110)
        write_cell(root, "R_real", z, 64, 512, lams[0], 1.0, 0.002, r9)
        write_cell(os.path.join(root, "scratch", "fake"), "R_fake",
                   z, 64, 512, lams[1], 99.0, 0.002, r9)
        cells, nfiles, _bad, _dup = A.load(root)
        seen = sorted({round(k[3], 4) for k in cells})
        good = (nfiles == R and seen == [round(lams[0], 4)])
        print("  %s %-34s loader read %d files, lambdas %s"
              % ("ok  " if good else "FAIL", "S9 scratch data is invisible",
                 nfiles, seen))
        results.append(good)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # ---------------------------------------------------------------------
    # S10-S13 exist because the first version of this suite MISSED the two
    # defects the red team found by executing the shipped script. A smoke test
    # that only exercises classify_rung on clean cases cannot see a wrong
    # recommendation rule or a gate that fires on the wrong side of an "or".
    # ---------------------------------------------------------------------

    def rungs_from(classes):
        """Build a rung dict of the shape main() consumes."""
        return {"%d->%d" % (a, b): dict(**{"class": c}, mde_pct=1.0)
                for (a, b), c in zip(A.RUNGS, classes)}

    def recommend(classes):
        """Call the SHIPPED rule, not a copy of it. A mirrored implementation
        in a test drifts from the thing it is testing and then agrees with it
        for the wrong reason."""
        r = rungs_from(classes)
        evs = [(a, b, r.get("%d->%d" % (a, b))) for a, b in A.RUNGS]
        return A.smallest_adequate(evs)[0]

    # S10  a rung that is stable ONLY at the bottom must NOT be recommended.
    #      This is the exact bug: first-stable-wins printed 128 while the table
    #      above it said 256 -> 512 STILL_CHANGING.
    got = recommend(["ROUGHLY_STABLE", "STILL_CHANGING",
                     "ROUGHLY_STABLE", "ROUGHLY_STABLE"])
    good = (got == 512)
    print("  %s %-34s recommended N_c = %s (must be 512, not 128)"
          % ("ok  " if good else "FAIL", "S10 stable-then-moving chain", got))
    results.append(good)

    # S11  every rung stable -> the bottom of the ladder is the answer.
    got = recommend(["ROUGHLY_STABLE"] * 4)
    good = (got == 128)
    print("  %s %-34s recommended N_c = %s"
          % ("ok  " if good else "FAIL", "S11 all rungs stable", got))
    results.append(good)

    # S12  an INCONCLUSIVE rung BREAKS the chain and may not be skipped over.
    got = recommend(["ROUGHLY_STABLE", "INCONCLUSIVE",
                     "ROUGHLY_STABLE", "ROUGHLY_STABLE"])
    good = (got is None)
    print("  %s %-34s recommended N_c = %s (must be None)"
          % ("ok  " if good else "FAIL", "S12 inconclusive breaks the chain", got))
    results.append(good)

    # S13  ONE unusable L must trigger INCONCLUSIVE. The gate read "at L=48 or
    #      L=64 exceeds 0.05" and was implemented as a MINIMUM, so it required
    #      BOTH to be unusable; an L = 64 curve at six times the threshold was
    #      classified ROUGHLY_STABLE.
    def one_bad_L(nc, L, lam):
        return (base(lam) + 1.0 * (L - 48) / 16.0 * (lam - XC),
                0.30 if L == 64 else 0.002)
    results.append(run_case(
        "S13 one unusable L", z, one_bad_L, 111,
        expect_class="INCONCLUSIVE"))

    print("-" * 78)
    n = sum(1 for r in results if r)
    print("  %d of %d cases classified as constructed" % (n, len(results)))
    if n != len(results):
        print("  A classifier that does not separate these cases cannot support")
        print("  the survey's verdicts, and the analysis must be fixed before use.")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
