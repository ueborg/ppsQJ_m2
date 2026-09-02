#!/usr/bin/env python3
"""Shared cost model for TASK-2026-09-02-MOCK-PRODUCTION.

Every number here is anchored on wall_s RECORDED BY COMPLETED RUCHE JOBS of this
identical production path. Never on a requested Slurm --time. Never on the
predecessor's pre-run projection. Full arithmetic and provenance: ../COST_MODEL.md.

Structure:

    rate_ms(L, N_c) = BASE_MS[L] * NC_FACTOR[N_c]

BASE_MS is the per-clone-window rate at the REFERENCE population N_c = 1024, and
NC_FACTOR carries the (measured, non-monotone) dependence on N_c at fixed L. The
two are separated because the predecessor conflated them and was wrong by 30 %:
it read the L=128 rungs 64/128/256 (27.18/26.81/21.52 ms) as a small-batch
inefficiency that flattens by N_c=256, and extrapolated FLAT to 512 and 1024.
ARM A's returned JSONs say the rate turns around and RISES again:

    L=128  N_c= 256   21.522 ms   (the predecessor's anchor, ARM2)
    L=128  N_c= 512   23.416 ms   ARM A512,  48 completed runs   +8.8 %
    L=128  N_c=1024   27.898 ms   ARM A1024, 32 completed runs  +29.6 %

so the rate is U-shaped in N_c: per-window loop overhead dominates below
N_c ~ 256, memory traffic dominates above it. Both branches matter here, because
this campaign runs N_c = 128, 1024 and 2048.
"""
import math

# --- BASE_MS: ms per clone-window at the reference N_c = 1024 -------------
#
# L = 64   MEASURED.  ARM B, 288 completed runs, N_c=1024, three lambdas:
#          4.680 / 4.806 / 4.674 ms.  Adopt 4.850, above all three.
#          (The predecessor had to DERIVE this and adopted 5.000. It was 6 %
#          conservative -- right direction, no longer needed.)
#
# L = 128  MEASURED.  ARM A1024, 32 completed runs: 27.898 ms.
#
# L = 32, 48  DERIVED by downward L-scaling from the measured L=64 anchor at
#          the SAME N_c, so the whole N_c dependence cancels identically. Three
#          candidate exponents were computed (COST_MODEL.md):
#             p = 2.563  same-N_c=1024 Ruche pair  L=64 vs L=128
#             p = 2.469  same-N_c= 512 Ruche pair  L=96 vs L=128
#             p = 2.339  same-N_c= 256 Mac pair    L=32 vs L=64  (SMCSTAT A-MV/A-HV)
#          A LARGER exponent predicts a SMALLER rate at low L, i.e. it is the
#          OPTIMISTIC direction. The adopted figures use p = 2.0 -- below every
#          measured exponent, hence conservative -- and are then rounded up:
#             L=48: 4.850*(48/64)^2 = 2.728  ->  adopt 3.000  (+10 %)
#             L=32: 4.850*(32/64)^2 = 1.213  ->  adopt 1.400  (+15 %)
#          Independent Mac->Ruche transfer gives 2.34 (L=48) and 0.96 (L=32);
#          the adopted values are above those too.
#
# L = 80   INTERPOLATION between two measured same-N_c=1024 points, not an
#          extrapolation: 4.850*(80/64)^2.563 = 8.548. Adopt 8.550. Used ONLY
#          to REJECT L=80 in ../L80_RUNTIME_GATE.md, where the rejection is
#          additionally shown to hold at the OPTIMISTIC p=2.0 rate of 7.578.
BASE_MS = {32: 1.400, 48: 3.000, 64: 4.850, 80: 8.550, 128: 27.898}

# --- NC_FACTOR: rate at N_c relative to the N_c = 1024 reference ----------
#
# N_c = 1024  reference, by construction.
#
# N_c = 2048  ONE doubling above a measured point. The only measured doubling of
#             this kind is L=128, N_c=512 -> 1024: 23.416 -> 27.898, i.e. +19.1 %.
#             That doubling also crossed a larger working set (537 -> 1074 MB)
#             than L=64's 2048-clone run will (268 -> 537 MB), and the smaller
#             doubling 256->512 at L=128 cost only +8.8 %. The true factor for
#             L=64 is therefore expected between 1.09 and 1.19; adopt 1.20,
#             above both. This is the single largest modelling judgement in the
#             package and it is flagged again in COST_MODEL.md.
#
# N_c = 128   SMALL-BATCH branch, where per-window overhead is amortised over
#             few clones. Measured penalties against the same L's N_c=256 rate:
#                L=128 Ruche  26.805 / 21.522 = 1.245
#                L= 32 Mac     0.682 /  0.587 = 1.162   (SMCSTAT A-MV)
#             Against the N_c=1024 reference the penalty is larger still because
#             the reference itself sits on the memory branch. Adopt 1.35, above
#             every measured comparison. The N_c=128 arms are ~2 % of the
#             campaign's core-hours, so over-provisioning them costs nothing.
NC_FACTOR = {128: 1.35, 1024: 1.00, 2048: 1.20}

# Multiplicative band applied when quoting the pessimistic figure. Inherited
# unchanged from the predecessor, and now known to be ample: ARM B's observed
# max/median wall spread was 1.077 and its 288-task array finished in 2.76 h
# against a 2.84 h central prediction.
PESSIMISTIC = 1.40

# Empirical packing factor for an --array=...%C job. Elapsed time is
# throughput-bound at core_h / C, times this. CALIBRATED on ARM B, the only
# completed multi-wave array in the programme: 288 tasks, %64, 157.8 core-hours
# actually consumed, 157.8/64 = 2.47 h, observed span 2.76 h -> 1.118.
# Adopt 1.15. Reconstructed from the array's own .out timestamps, which also
# confirm the allocation granted the full 64 concurrent slots PER ARRAY while
# two other arrays were running (max_concurrent was 64, 48 and 32 at once).
PACKING = 1.15

DTAU_MULT = 6.0          # CERTIFIED production value. Never the corpus 12.


def rate_ms(L, N_c):
    return BASE_MS[L] * NC_FACTOR[N_c]


def n_steps(L, T, lam, dtau_mult=DTAU_MULT):
    """Exactly the production discretisation: support/instrumented.py lines
    127-128, with alpha == lam on this cut. Verified against measured n_steps
    (L=64, lam=0.2932/0.3032/0.3132 -> 395/408/421; L=96 -> 922; L=128 -> 1643)."""
    dtau = dtau_mult / max(2.0 * lam * (L - 1), 1e-12)
    return max(1, int(math.ceil(T / dtau)))


def wall_s(L, T, N_c, lam, dtau_mult=DTAU_MULT):
    return rate_ms(L, N_c) * 1e-3 * N_c * n_steps(L, T, lam, dtau_mult)


def mem_mb(L, N_c):
    """Same formula the predecessor preflight used and that was validated
    against ARM2's real footprint."""
    per_clone = (2 * L) ** 2 * 8 + (2 * L) * L * 16
    return 128.0 + 2.0 * N_c * per_clone / 1e6


def elapsed_h(core_h, slowest_h, concurrency):
    """Throughput-bound elapsed time for one --array=%concurrency job,
    floored at the single slowest task. See PACKING."""
    return max(core_h / concurrency * PACKING, slowest_h)
