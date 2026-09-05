#!/usr/bin/env python3
"""Cost model for TASK-2026-09-06-NC-ZETA-MOCKPROD.

Every rate below is a per-clone-window wall time ACTUALLY RECORDED by a
completed Ruche job of this identical code path, or a ratio taken from the only
zeta-resolved timing in the repository. Never a requested --time, never a
predecessor's projection.

Three inputs and their status:

  K(L, lambda, dtau_mult)  EXACT      ceil(2*lambda*(L-1)*T/dtau_mult), T = L
  rate35(L, N_c)           MEASURED   Ruche, zeta = 0.35, dtau_mult = 6
  rho(zeta)                MEASURED RATIO, local  (ALGRD b0_L*.json)
  f_lam(lambda, zeta)      INFERRED   corrects rho off the 0.51*sqrt(zeta) line

`refit()` re-derives rate35 and rho from raw data and fails if the literals in
this file have drifted. The preflight calls it on every run.
"""
from __future__ import annotations
import collections, glob, json, math, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *([os.pardir] * 5)))

# ---------------------------------------------------------------------------
# 1. rate35(L, N_c) -- Ruche-MEASURED, ms per clone-window, at zeta = 0.35.
#
# The adopted value at each L is the MAXIMUM per-clone-window rate any completed
# run at that L showed, over EVERY measured N_c rung. --time protects the
# slowest task in an array, not the typical one.
#
# WHY NOT N_c^0.1871. TASK-2026-09-03-NC-PLATEAU-CALIBRATION fitted rate ~
# N_c^G with G = 0.1871 on three L=128 rungs (21.52, 23.42, 27.90 at N_c =
# 256, 512, 1024) and applied it above the largest measured rung everywhere.
# Its own campaign has since returned the rungs that test it:
#     L = 128, N_c = 2048  ->  22.15 ms  (G predicts 31.7)
#     L =  64, N_c = 4096  ->   4.60 ms  (G predicts  6.6)
#     L =  64, N_c = 8192  ->   4.75 ms  (G predicts  7.5)
# The law is REFUTED by the data that was collected to test it. Over
# N_c = 512..8192 at L in {32, 48, 64} the measured rate is FLAT to within
# +-9 %, with no monotone trend. A flat per-L constant is what this task adopts.
# ---------------------------------------------------------------------------
RATE35_MAX = {32: 1.951, 48: 3.457, 64: 5.769}      # ms per clone-window

# Small-batch penalty below the smallest MEASURED rung at that L. The corpus
# shows the rate rising as N_c falls at large L: L=128 gives 29.506/23.998 =
# 1.230 at N_c = 64 and 28.683/23.998 = 1.195 at 128; L=96 gives 11.705/10.120
# = 1.157 at 128. At L = 64 the rungs N_c = 64 and 256 ARE measured (5.111 and
# 5.011 worst-case) and are BELOW the adopted L=64 constant, so no penalty is
# needed there empirically -- it is applied anyway, upward.
SMALL_BATCH = {128: 1.25, 256: 1.15, 512: 1.0, 1024: 1.0, 2048: 1.0}

# ---------------------------------------------------------------------------
# 2. rho(zeta) -- ratio of per-clone-window rate to its value at zeta = 0.35,
# at FIXED L, from research/tasks/active/TASK-2026-08-11-ALGRD/results/b0_L*.json
# (sec_per_clone_window at zeta in {0.05, 0.15, 0.30, 0.70}), log-log
# interpolated in zeta and MAXIMISED over the four L. Local measurement, ratios
# only; the absolute level always comes from RATE35_MAX above.
# ---------------------------------------------------------------------------
RHO = {0.10: 0.4486, 0.20: 0.6391, 0.35: 1.0000, 0.70: 2.3720}

# ---------------------------------------------------------------------------
# 3. f_lam -- the confound correction, and the WEAKEST INPUT IN THIS PACKAGE.
#
# In the ALGRD data lambda co-varies with zeta as lambda = 0.51*sqrt(zeta), so
# RHO above is a rate ratio measured ALONG THAT LINE, not at fixed lambda. This
# task's grids are not on that line. At FIXED zeta = 0.35 the Ruche corpus
# spans lambda = 0.1932..0.3532 at three L and three N_c, and the per-clone-
# window rate DECREASES with lambda there:
#     L=32 N_c=1024  slope -0.434     L=48 N_c=1024  slope -0.367
#     L=64 N_c=1024  slope -0.357     L=64 N_c=2048  slope -0.360
# and, over the two ENDPOINTS of the same 21-point spans rather than a least-
# squares fit through all of them, the numerics investigator independently got
#     L=32 N_c=1024  -0.48            L=64 N_c=1024  -0.53
#
# WHICH DIRECTION IS CONSERVATIVE. This task's lambda grids sit BELOW the line
# RHO was measured on, so (lambda / LAM_REF)^A_LAM is > 1 and gets LARGER as
# A_LAM gets MORE negative. A draft of this file adopted -0.35 as "the least
# negative of the well-sampled fits, which makes the correction largest" -- that
# reasoning is backwards, and -0.35 is the LEAST conservative of the estimates,
# not the most. Caught by agent_reports/numerics.md Q2. A_LAM = -0.50 is adopted:
# it is the investigator's independent estimate and it errs upward.
A_LAM = -0.50
LAM_REF = lambda z: 0.51 * math.sqrt(z)      # the line RHO was measured along
# The cap exists so the correction cannot run away where it is extrapolated
# furthest. It is set ABOVE the largest value the design actually reaches
# (2.008, at zeta = 0.10, lambda = 0.040) so that it does not silently truncate
# this campaign's own worst cell -- a cap that binds is a cap that hides.
F_LAM_CAP = 2.5

# 1.40 pessimistic band; 1.15 packing overhead, inherited and calibrated on ARM
# B of TASK-2026-09-02-MOCK-PRODUCTION (157.8 core-h consumed, 2.47 h
# throughput-bound, 2.76 h observed span -> 1.118, adopted 1.15).
PESSIMISTIC, PACKING = 1.40, 1.15


def K(L: int, lam: float, dtau_mult: float = 6.0, T: float | None = None) -> int:
    """n_steps. EXACT, not estimated: instrumented.py lines 127-128."""
    T = float(L) if T is None else T
    return math.ceil(2.0 * lam * (L - 1) * T / dtau_mult)


def f_lam(lam: float, zeta: float) -> float:
    return min(F_LAM_CAP, (lam / LAM_REF(zeta)) ** A_LAM)


def rate(L: int, N_c: int, lam: float, zeta: float) -> float:
    """ms per clone-window."""
    return RATE35_MAX[L] * SMALL_BATCH[N_c] * RHO[zeta] * f_lam(lam, zeta)


def row_seconds(L, N_c, lam, zeta, dtau_mult=6.0, T=None) -> float:
    return K(L, lam, dtau_mult, T) * N_c * rate(L, N_c, lam, zeta) / 1000.0


def mem_mb(L: int, N_c: int) -> float:
    """1.45 x the inherited formula. The 1.45 exists because the formula alone
    under-predicts real peak RSS by up to 1.6x, measured at 15 cells by
    TASK-2026-09-03-NC-PLATEAU-CALIBRATION. zeta does not enter memory."""
    return 1.45 * (128 + 2 * N_c * ((2 * L) ** 2 * 8 + (2 * L) * L * 16) / 1e6)


def gib_request(L: int, N_c: int) -> str:
    need = mem_mb(L, N_c) / 1024.0
    for g in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        if g >= need * 1.35:
            return "%dG" % g
    raise ValueError("no request covers %.1f MB" % mem_mb(L, N_c))


# ---------------------------------------------------------------------------
# refit -- re-derive the literals from raw data. Called by the preflight.
# ---------------------------------------------------------------------------
def _load_ruche():
    out = collections.defaultdict(list)
    for p in glob.glob(os.path.join(ROOT, "research/tasks/**/results/*.json"),
                       recursive=True):
        if os.sep + "scratch" + os.sep in p:
            continue      # synthetic / throwaway; never a measured runtime
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("status") != "ok":
            continue
        if float(d.get("dtau_mult", 0)) != 6.0 or float(d["zeta"]) != 0.35:
            continue
        n, N = int(d["n_steps"]), int(d["N_c"])
        out[(int(d["L"]), N)].append(float(d["wall_s"]) / (N * n) * 1000.0)
    return out


def _load_algrd():
    D = {}
    for p in sorted(glob.glob(os.path.join(
            ROOT, "research/tasks/active/TASK-2026-08-11-ALGRD/results/b0_L*.json"))):
        for d in json.load(open(p)):
            D.setdefault(d["L"], {})[d["zeta"]] = d["sec_per_clone_window"]
    return D


def refit(tol=0.005, verbose=True):
    """Refit RATE35_MAX and RHO from raw data. Returns list of failures."""
    fails = []
    ru = _load_ruche()
    for L, adopted in sorted(RATE35_MAX.items()):
        rungs = {nc: max(v) for (LL, nc), v in ru.items() if LL == L}
        if not rungs:
            fails.append("no Ruche data at L=%d" % L); continue
        got = max(rungs.values())
        if verbose:
            print("  rate35 L=%-3d measured rungs %s -> max %.3f  (literal %.3f)"
                  % (L, {k: round(v, 3) for k, v in sorted(rungs.items())}, got, adopted))
        if abs(got - adopted) / adopted > tol:
            fails.append("RATE35_MAX[%d] literal %.3f but data gives %.3f"
                         % (L, adopted, got))
        # and the refutation of the N_c^0.1871 law, re-checked every run
        ks = sorted(rungs)
        if len(ks) >= 3:
            g_pred = rungs[ks[0]] * (ks[-1] / ks[0]) ** 0.1871
            if verbose:
                print("     N_c^0.1871 from N_c=%d predicts %.3f at N_c=%d; measured %.3f"
                      % (ks[0], g_pred, ks[-1], rungs[ks[-1]]))

    D = _load_algrd()

    def loglin(L, z):
        zs = sorted(D[L])
        if z in D[L]:
            return D[L][z]
        lo = max(x for x in zs if x < z); hi = min(x for x in zs if x > z)
        t = (math.log(z) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return math.exp(math.log(D[L][lo]) + t * (math.log(D[L][hi]) - math.log(D[L][lo])))

    for z, adopted in sorted(RHO.items()):
        got = max(loglin(L, z) / loglin(L, 0.35) for L in D)
        if verbose:
            print("  rho(%.2f) per L %s -> max %.4f  (literal %.4f)"
                  % (z, [round(loglin(L, z) / loglin(L, 0.35), 4) for L in sorted(D)],
                     got, adopted))
        if abs(got - adopted) / adopted > tol:
            fails.append("RHO[%.2f] literal %.4f but data gives %.4f" % (z, adopted, got))
    return fails


if __name__ == "__main__":
    import sys
    print("refitting cost-model literals from raw data")
    f = refit()
    print("FAIL: " + "; ".join(f) if f else "all literals within 0.5 % of the data")
    sys.exit(1 if f else 0)
