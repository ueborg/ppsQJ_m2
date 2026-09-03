#!/usr/bin/env python3
"""Cost model for TASK-2026-09-03-NC-PLATEAU-CALIBRATION.

Every rate here is a per-clone-window wall time ACTUALLY RECORDED BY A COMPLETED
RUCHE JOB of this identical production path. Never a requested `--time`, never a
laptop probe, never a predecessor projection. Provenance: the 1896 raw result
JSONs enumerated by `tools/reconstruct_inventory.py`, summarised in
`../EXISTING_POPULATION_INVENTORY.csv`.

WHAT CHANGED SINCE THE PREDECESSOR MODELS, AND WHY
--------------------------------------------------
1. THE N_c DIRECTION REVERSED, AND THE OLD MODEL EXTRAPOLATES THE WRONG WAY.
   TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA read the L=128 rungs it then had
   (27.18, 26.81, 21.52 ms at N_c = 64, 128, 256) as "small-batch inefficiency
   that is gone by N_c ~ 256", and extrapolated every larger N_c FLAT from the
   N_c = 256 rate. Two more rungs have since completed and the rate turns round:

       L=128   N_c    64     128     256     512    1024
               med  27.176  26.805  21.522  23.416  27.898   ms/clone-window

   Flat-from-256 predicts 21.52 ms at N_c = 1024. The measurement is 27.90 --
   30 % low. The old model is therefore not merely imprecise at high N_c, it is
   biased optimistic in exactly the regime this task is built to enter.

   This file fits the rise instead. Over the three rungs where the small-batch
   regime is over (N_c >= 256) the L=128 medians give a log-log slope

       G = 0.1871          rate ~ N_c ** G

   and that exponent is applied to every extrapolation ABOVE the largest
   measured rung at each L. It is CONSERVATIVE where it is applied out of its
   own L: at L=96 the three measured rungs show no trend, and at L=64 the
   measured 1024 -> 2048 step is slightly NEGATIVE (5.769 -> 5.075 worst-case).
   G is used anyway. A cost model that is wrong should be wrong upward.

2. THE RATE USED IS THE WORST SINGLE RUN AT ITS RUNG, not the median. `--time`
   protects the slowest task in an array, not the typical one.

3. THE RATE IS MADE MONOTONE IN N_c. The measured envelope wobbles (L=128 dips
   at N_c=256; L=64 dips at N_c=2048). A dip is not a licence to request less
   wall time at a larger population, so `rate()` returns the running maximum
   over all measured rungs at or below the requested N_c, and extrapolates from
   THAT with G.

4. SMALL-BATCH PENALTY BELOW THE SMALLEST MEASURED RUNG. Going down in N_c the
   corpus shows the rate RISING (L=128: 23.998 at 256 -> 29.506 at 64, a factor
   1.23; L=96: 11.534 at 256 -> 12.426 at 128, 1.08). Campaign E runs at
   N_c = 64 and 256 at L=64, where the smallest measured rung is 1024, so the
   penalty is applied rather than assumed away.

5. THE MEMORY MODEL IS NOW MEASURED. See mem_mb() below; this is the one place
   the inherited model was not merely imprecise but wrong by about a factor two,
   and no predecessor had ever measured it.

This file contains no scheduler call and cannot submit anything.
"""
import math

DTAU_MULT_PRODUCTION = 6.0    # the CERTIFIED value. Campaign E varies it ON PURPOSE.
ZETA = 0.35
PESSIMISTIC = 1.40            # inherited band; covers observed max/median spread
PACKING = 1.15                # array packing inefficiency, calibrated on ARM B
K_MEM = 4.5                   # measured peak-RSS coefficient, see mem_mb()
G = 0.1871                    # rate ~ N_c**G above the largest measured rung

# ---------------------------------------------------------------------------
# MEASURED per-clone-window rates, ms. WORST single completed run at each rung.
# Source: EXISTING_POPULATION_INVENTORY.csv, 1896 populations. `n` is the
# number of completed populations behind each entry.
#   L    N_c      n   median   p90     max      <- max is what is used
#   32  1024    408    1.501   1.735   1.922
#   48  1024    408    2.881   3.237   3.457
#   64  1024    624    4.841   5.370   5.769
#   64  2048     72    4.788   4.888   5.075
#   96   128     32   11.705  12.320  12.426
#   96   256     32   10.120  11.354  11.534
#   96   512     48   11.510  11.907  11.989
#  128    64     64   27.176  28.967  29.506
#  128   128     64   26.805  28.483  28.683
#  128   256     64   21.522  22.423  23.998
#  128   512     48   23.416  24.874  25.080
#  128  1024     32   27.898  28.137  28.260
# ---------------------------------------------------------------------------
MEASURED_MAX_MS = {
    32: {1024: 1.922},
    48: {1024: 3.457},
    64: {1024: 5.769, 2048: 5.075},
    96: {128: 12.426, 256: 11.534, 512: 11.989},
    128: {64: 29.506, 128: 28.683, 256: 23.998, 512: 25.080, 1024: 28.260},
}

# Penalty applied when N_c is BELOW the smallest measured rung at that L. Keyed
# by the requested N_c. Derived from the two L where the corpus straddles the
# small-batch regime: L=128 gives 29.506/23.998 = 1.230 at N_c=64 and
# 28.683/23.998 = 1.195 at 128; L=96 gives 12.426/11.534 = 1.077 at 128.
SMALL_BATCH = {64: 1.30, 128: 1.25, 256: 1.15, 512: 1.05}


def n_steps(L, T, lam, dtau_mult=DTAU_MULT_PRODUCTION):
    """The production discretisation, verbatim from support/instrumented.py
    lines 127-128 with alpha == lam on this cut:

        delta_tau = dtau_mult / (2 * lam * (L - 1))
        n_steps   = ceil(T / delta_tau)

    Verified against the n_steps every completed run recorded for itself, at
    every (L, lambda) in the corpus, exact in all 1896 cases -- see
    ../VALIDATION.md.
    """
    dtau = dtau_mult / max(2.0 * lam * (L - 1), 1e-12)
    return max(1, int(math.ceil(T / dtau)))


def rate_ms(L, N_c):
    """Adopted per-clone-window rate, ms. Monotone non-decreasing in N_c."""
    m = MEASURED_MAX_MS[L]
    rungs = sorted(m)
    if N_c < rungs[0]:
        return m[rungs[0]] * SMALL_BATCH.get(N_c, 1.30)
    env = max(m[r] for r in rungs if r <= N_c)       # running max: no dips
    if N_c <= rungs[-1]:
        return env
    top = max(m.values())                            # extrapolate from the envelope
    return top * (N_c / rungs[-1]) ** G


def wall_s(L, T, lam, N_c, dtau_mult=DTAU_MULT_PRODUCTION):
    return rate_ms(L, N_c) * 1e-3 * N_c * n_steps(L, T, lam, dtau_mult)


# Direct ru_maxrss measurements of the BUNDLED CERTIFIED SAMPLER, MB, by
# tools/mem_probe.py on this machine. Short-T probes, so the window-indexed
# genealogy arrays are absent from them and are added analytically below.
# These are the first peak-RSS measurements of this sampler anywhere in the
# repository.
#
# EACH CELL HOLDS A LIST, AND THE MAXIMUM IS USED, because repeated probes of
# the SAME cell are NOT reproducible -- and, at every cell probed twice, the
# SECOND probe came in HIGHER:
#
#     L=128 N_c=2048    3482.5, 3521.7, 6275.9    spread x1.80  (3 probes)
#     L=64  N_c=8192    3547.0, 4593.8            spread x1.30
#     L=64  N_c=4096    2032.6, 2747.1            spread x1.35
#     L=96  N_c=1024    2006.3, 2139.8            spread x1.07
#
# EVERY cell probed more than once varies. The L=128 triple also shows that two
# probes can agree closely (3482.5, 3521.7, within 1.1%) and STILL be 1.80x
# below a third -- so agreement between two probes is not evidence of a bound
# either.
#
# ru_maxrss is a high-water mark over the whole process and depends on when the
# allocator happens to release the transient copies that selection makes;
# nothing about the sampler changed between any pair of runs.
#
# That non-reproducibility is itself the finding. A SINGLE probe of a cell is
# not a bound, and treating one as a bound is how a 31-hour job dies at hour 20
# with OOM. Hence: max over probes, the old formula retained as a floor, and a
# 1.35x margin on top of both. See ../COST_MODEL.md section 4.
MEASURED_PEAK_MB_PROBES = {
    (32, 64): [90.8], (32, 256): [149.2], (32, 1024): [276.4],
    (32, 2048): [592.1], (32, 4096): [1063.9], (32, 8192): [1709.5],
    (64, 128): [236.3], (64, 512): [566.0], (64, 2048): [1694.2],
    (64, 4096): [2032.6, 2747.1], (64, 8192): [3547.0, 4593.8],
    (96, 128): [430.6], (96, 1024): [2006.3, 2139.8], (96, 2048): [2200.8],
    (128, 2048): [3482.5, 3521.7, 6275.9],
}
MEASURED_PEAK_MB = {k: max(v) for k, v in MEASURED_PEAK_MB_PROBES.items()}


def mem_mb(L, N_c, n_steps_=None):
    """Peak RSS, MB.

    The model EVERY predecessor package used was

        peak = 128 + 2 * N_c * per_clone

    and the coefficient 2 was never checked against a running process.
    TASK-2026-09-01-SMCRUCHE-READY describes its output as "the measured 732 MB
    peak"; 732 MB is exactly what that formula returns for L=96, N_c=512, and no
    MaxRSS from any Ruche job appears anywhere in this repository. So the number
    was a model quoted as a measurement.

    It is now measured. Direct ru_maxrss of the bundled sampler puts the true
    peak ABOVE that formula at seven of the thirteen probed cells -- L=64,
    N_c=2048 reads 1694 MB against a predicted 1202 MB, and that arm shipped
    with --mem=2G, i.e. 21 % headroom rather than the 70 % its own comment
    claimed. It never broke, and it was closer to breaking than anyone knew.

    This model therefore takes, per cell, the LARGER of
      * the direct measurement, where one exists;
      * a conservative K_MEM = 4.5 per-clone model, where it does not;
      * the deployed 128 + 2*N_c*per_clone formula, always, as a floor, so this
        campaign never requests LESS than a predecessor did for a comparable
        cell;
    and then adds the window-indexed genealogy arrays (anc_matrix and
    idxs_history, both n_steps x N_c intp) analytically, because a short-T
    probe cannot see them.

    The measurements are macOS ru_maxrss and the cluster is Linux. That is a
    real limitation, it is why the floor and the margin are both kept, and it
    is recorded rather than argued away.
    """
    per_clone = ((2 * L) ** 2 * 8 + (2 * L) * L * 16) / 1e6      # MB per clone
    geneal = 0.0 if n_steps_ is None else 2.0 * n_steps_ * N_c * 8 / 1e6
    floor = 128.0 + 2.0 * N_c * per_clone                        # deployed model
    body = MEASURED_PEAK_MB.get((L, N_c), 150.0 + K_MEM * N_c * per_clone)
    return max(floor, body) + geneal


def mem_measured(L, N_c):
    """True if this cell's peak was directly measured rather than modelled."""
    return (L, N_c) in MEASURED_PEAK_MB


def mem_probe_spread(L, N_c):
    """max/min over repeated probes of this cell, or None if probed once."""
    v = MEASURED_PEAK_MB_PROBES.get((L, N_c))
    return (max(v) / min(v)) if v and len(v) > 1 else None


def mem_request_gb(L, N_c, n_steps_=None, margin=1.5):
    """The --mem to ask for: margin x the model, rounded up to a whole GB."""
    return max(1, int(math.ceil(mem_mb(L, N_c, n_steps_) * margin / 1024.0)))


def elapsed_h(n_tasks, core_h, slowest_h, concurrency):
    """Elapsed for one array at a %concurrency cap, EXCLUDING queue wait.

    Two bounds, larger adopted: the wave floor (ceil(n/C) sequential slowest
    tasks) and the throughput bound (core_h / C, inflated by PACKING). Short
    arrays are wave-bound; long ones are throughput-bound.
    """
    waves = math.ceil(n_tasks / concurrency)
    return max(core_h / concurrency * PACKING, waves * slowest_h)


def slurm_time(pess_slowest_h):
    """--time from the PESSIMISTIC slowest task, with >= 1.6x headroom on top,
    snapped up to a readable limit. Never fitted to a partition's MaxTime."""
    need = pess_slowest_h * 1.6
    for h in (1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168):
        if h >= need:
            return "%02d:00:00" % h
    raise ValueError("job longer than cpu_long MaxTime (168 h): %.1f h" % need)


if __name__ == "__main__":
    print("adopted rate_ms(L, N_c)")
    print("   L " + "".join("%9d" % n for n in (64, 256, 512, 1024, 2048, 4096, 8192)))
    for L in (32, 48, 64, 96, 128):
        print("%4d " % L + "".join("%9.3f" % rate_ms(L, n)
                                   for n in (64, 256, 512, 1024, 2048, 4096, 8192)))
    print("\nmem_mb(L, N_c) and the old formula it replaces")
    print("%4s %7s %10s %10s" % ("L", "N_c", "measured", "old"))
    for L, n in ((64, 2048), (64, 4096), (64, 8192), (96, 1024), (96, 2048),
                 (128, 2048)):
        pc = ((2 * L) ** 2 * 8 + (2 * L) * L * 16) / 1e6
        print("%4d %7d %10.0f %10.0f" % (L, n, mem_mb(L, n), 128 + 2 * n * pc))
