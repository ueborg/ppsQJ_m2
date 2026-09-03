#!/usr/bin/env python3
"""Cost model for TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION.

EVERY number here is fitted to `wall_s` ACTUALLY RECORDED BY COMPLETED RUCHE
JOBS of this identical code path, at this identical N_c, on this identical
lambda family. Never a requested Slurm `--time`. Never a predecessor projection.

WHY THIS FILE DOES NOT INHERIT THE PREDECESSOR'S MODEL
-----------------------------------------------------
TASK-2026-09-02-MOCK-PRODUCTION had to PREDICT its own cost, and its model was

    wall_s = BASE_MS[L] * NC_FACTOR[N_c] * 1e-3 * N_c * n_steps

i.e. strictly proportional to n_steps with no intercept, with BASE_MS[32] and
BASE_MS[48] DERIVED by downward L-scaling rather than measured. That campaign
has now returned 864 completed N_c = 1024 runs, so this task does not have to
predict anything: it can fit.

The returned data falsify the zero-intercept form. Per-clone-window rates over
each arm's own 13-lambda span are not constant:

    L = 32   1.294 - 1.888 ms   (median 1.479)
    L = 48   2.330 - 3.354 ms   (median 2.839)
    L = 64   3.840 - 5.527 ms   (median 4.809)

and they drift SYSTEMATICALLY with n_steps rather than scattering, because a run
carries a fixed per-run cost -- interpreter start, the pps_qj import, lattice and
population construction, the final observable pass -- that a proportional model
must smear into the per-window rate. The measured relation is affine:

    L        wall_s = a * n_steps + b        resid sd    n
    32       0.815551 * n +  68.43              7.1 s   312
    48       1.588743 * n + 286.09             24.3 s   312
    64       2.723572 * n + 850.23             99.9 s   528

Fitted by least squares over every N_c = 1024 population in
frozen_inputs/predecessor_nc1024_populations.csv, with n_steps read from the
JSON the run itself wrote (not recomputed), so the regressor is exactly what
the sampler did. The L = 64 fit has n = 528 rather than 240 because the frozen
snapshot carries the reused ARM-B centre triple at its full R = 96 alongside
mockL64's 240; those runs are the same code path at the same L and N_c on the
same cluster, so excluding them would be discarding measurement, not noise.
fit_from_frozen() below recomputes all three and the preflight fails on drift.

WHICH DIRECTION THE CHOICE ERRS
-------------------------------
This task's four new lambdas are BELOW the fitted range, so n_steps is an
extrapolation DOWNWARD:

    L = 32   fitted over n_steps 78-117    used at 64-74
    L = 48   fitted over n_steps 176-266   used at 146-168
    L = 64   fitted over n_steps 314-475   used at 260-300

A positive intercept makes the affine model predict MORE time at low n_steps
than the proportional model does. At L = 64, lambda = 0.1932 the affine model
says 1558 s and the median-rate proportional model says 1280 s -- the affine
model is 22 % more conservative, in the direction that matters for a `--time`
limit. That is deliberate. Where the two models disagree this file takes the
larger, and `--time` is then set above 1.40x the larger.

The extrapolation is also SHORT: 0.82x the fitted floor at L = 32, 0.83x at
L = 48, 0.83x at L = 64. Nothing here reaches into a regime the campaign has not
already measured within 20 %.

This file contains no scheduler call and cannot submit anything.
"""
import os, csv, math, collections

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
FROZEN = os.path.join(TASK, "frozen_inputs",
                      "predecessor_nc1024_populations.csv")

DTAU_MULT = 6.0          # CERTIFIED production value. Never the corpus 12.
NC = 1024
PESSIMISTIC = 1.40       # inherited; ARM B's measured max/median was 1.077 and
                         # this campaign's three arms gave 1.165 / 1.205 / 1.199
PACKING = 1.15           # calibrated on ARM B: 288 tasks, %64, 157.8 core-h,
                         # 2.47 h throughput-bound vs 2.76 h observed -> 1.118

# The affine fits above, kept as literals so this module needs no data file to
# answer a cost question, and CHECKED against a refit by fit_from_frozen().
AFFINE = {32: (0.815551, 68.43), 48: (1.588743, 286.09), 64: (2.723572, 850.23)}

# Highest per-clone-window rate any completed run of that arm actually showed.
# Used only as the second branch of the max() below.
RATE_MAX_MS = {32: 1.888, 48: 3.354, 64: 5.527}

# The n_steps span each fit was estimated over. Quoted by the preflight so a
# reader can see how far out of sample a request is.
FIT_RANGE = {32: (78, 117), 48: (176, 266), 64: (314, 475)}


def n_steps(L, T, lam, dtau_mult=DTAU_MULT):
    """Exactly the production discretisation (support/instrumented.py), with
    alpha == lam on this cut. Verified against n_steps recorded in the
    predecessor's own returned JSONs at all 13 old lambdas and all three L."""
    dtau = dtau_mult / max(2.0 * lam * (L - 1), 1e-12)
    return max(1, int(math.ceil(T / dtau)))


def wall_s_affine(L, T, lam, dtau_mult=DTAU_MULT):
    a, b = AFFINE[L]
    return a * n_steps(L, T, lam, dtau_mult) + b


def wall_s_maxrate(L, T, lam, N_c=NC, dtau_mult=DTAU_MULT):
    return RATE_MAX_MS[L] * 1e-3 * N_c * n_steps(L, T, lam, dtau_mult)


def wall_s(L, T, lam, N_c=NC, dtau_mult=DTAU_MULT):
    """The adopted prediction: the LARGER of the two measured-data models."""
    return max(wall_s_affine(L, T, lam, dtau_mult),
               wall_s_maxrate(L, T, lam, N_c, dtau_mult))


def mem_mb(L, N_c=NC):
    """Unchanged from the predecessor, where it was validated against ARM2's
    real footprint. Population size and L are identical here, so the footprint
    is identical; nothing about lambda touches memory."""
    per_clone = (2 * L) ** 2 * 8 + (2 * L) * L * 16
    return 128.0 + 2.0 * N_c * per_clone / 1e6


def elapsed_h(n_tasks, core_h, slowest_h, concurrency):
    """Elapsed time for one --array=0-(n-1)%C job.

    The predecessor used max(core_h/C * PACKING, slowest_h). That is right for
    a 240-624 task array, where many waves average out. THESE arrays are 96
    tasks at %64 -- exactly two waves -- so the wave floor, not throughput,
    is what binds: two sequential slowest-tasks. Both bounds are computed and
    the LARGER is adopted, which for every arm here is the two-wave floor.

    Queue wait is NOT included in any figure and is expected to dominate all of
    them; see ../COST_MODEL.md.
    """
    waves = math.ceil(n_tasks / concurrency)
    return max(core_h / concurrency * PACKING, waves * slowest_h)


def fit_from_frozen():
    """Refit the affine model from the frozen predecessor snapshot.

    Returns {L: (a, b, resid_sd, n, rate_max_ms, (n_lo, n_hi))}. The preflight
    calls this and FAILS if the literals above have drifted from the data, so
    the constants cannot silently rot away from their own provenance.
    """
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(FROZEN)):
        if r["status"] != "ok" or int(r["N_c"]) != NC:
            continue
        by[int(r["L"])].append((int(r["n_steps"]), float(r["wall_s"])))
    out = {}
    for L, v in by.items():
        n = [x[0] for x in v]
        w = [x[1] for x in v]
        m = len(v)
        mn = sum(n) / m
        mw = sum(w) / m
        sxx = sum((x - mn) ** 2 for x in n)
        sxy = sum((x - mn) * (y - mw) for x, y in v)
        a = sxy / sxx
        b = mw - a * mn
        rss = sum((y - (a * x + b)) ** 2 for x, y in v)
        sd = math.sqrt(rss / (m - 2))
        rmax = max(y * 1000.0 / (NC * x) for x, y in v)
        out[L] = (a, b, sd, m, rmax, (min(n), max(n)))
    return out


if __name__ == "__main__":
    print("refit from frozen_inputs/predecessor_nc1024_populations.csv")
    print(f"  {'L':>3} {'a (s/step)':>12} {'b (s)':>10} {'resid sd':>10} "
          f"{'n':>5} {'max rate ms':>12} {'n_steps span':>16}")
    for L, (a, b, sd, m, rmax, span) in sorted(fit_from_frozen().items()):
        print(f"  {L:>3} {a:12.6f} {b:10.2f} {sd:10.2f} {m:5d} {rmax:12.3f} "
              f"{str(span):>16}")
    print("\nliterals in this file")
    for L in sorted(AFFINE):
        print(f"  {L:>3} {AFFINE[L][0]:12.6f} {AFFINE[L][1]:10.2f} "
              f"{'':10} {'':5} {RATE_MAX_MS[L]:12.3f} {str(FIT_RANGE[L]):>16}")
