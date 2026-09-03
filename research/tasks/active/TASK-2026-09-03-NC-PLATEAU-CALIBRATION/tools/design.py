#!/usr/bin/env python3
"""THE FROZEN DESIGN of TASK-2026-09-03-NC-PLATEAU-CALIBRATION.

One module, imported by the builder, the preflight, the cost report, the dedup
scan and the analysis, so there is exactly one place a design constant lives.
Editing a manifest by hand is an error; regenerate from here.

Nothing in this file is a scientific conclusion. Every lambda comes from an
OBSERVED locator region in already-measured curves, never from a hypothesised
boundary law. In particular nothing here is centred on sqrt(zeta), zeta**(1/3)
or any other candidate exponent -- doing so would import the exponent the
programme exists to measure.

Contains no scheduler call and cannot submit.
"""
ZETA = 0.35
SCHEME = "systematic"
DTAU_PRODUCTION = 6.0

# --- Campaign A: the deep central ladder -----------------------------------
A_L, A_LAM = 64, 0.3032

# --- Campaign B: the transition-region grid --------------------------------
# Seven points, delta_lambda = 0.005, spanning 0.2182-0.2482.
# PROVENANCE, and its limit: the two low-L pairs whose cross-L difference
# changes sign in the INTERIOR of the measured 17-point grid do so at
#     L32-L64   between 0.2232 and 0.2332
#     L48-L64   between 0.2332 and 0.2432
# (TASK-2026-09-02-MOCK-PRODUCTION + TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION,
# recomputed from raw files by tools/reconstruct_inventory.py). The window is
# those two brackets plus one grid step of guard on each side. It is a LOCATOR
# region in L <= 64 curves at N_c = 1024. It is NOT lambda_c(zeta) and this
# task may not call it one.
B_GRID = [0.2182, 0.2232, 0.2282, 0.2332, 0.2382, 0.2432, 0.2482]
B_L, B_NCS, B_R = 64, [512, 1024, 2048], 48

# --- Campaign B2: the matched low-L reference ladders ----------------------
# B on its own can only move N_c on ONE side of a cross-L difference: the
# L = 32 and L = 48 reference curves exist at N_c = 1024 and nowhere else, so
# the locator test of section 4B would be one-sided. B2 puts L = 32 and L = 48
# on the SAME 7-point grid at the SAME three N_c with the same matched R.
#
# WHY THE FULL 7-POINT GRID AND NOT THE 3 SHARED LAMBDAS.
# The first version of this arm used only the three lambdas campaign B shares
# with the already-measured 0.010 grid -- 0.2232, 0.2332, 0.2432 -- for about a
# sixth of the cost. Running the frozen crossing protocol on that grid showed
# the design is unusable: BOTH interior crossings (L32-L64 near 0.2315,
# L48-L64 near 0.2369) fall in the FIRST or LAST interval of a three-point
# grid, so the protocol flags every one of them ENDPOINT_INDUCED by
# construction, whatever the data say. That is exactly the defect
# TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION was created to repair, and
# knowingly rebuilding it to save ~200 core-hours would have made the
# load-bearing question of section 4B unanswerable. On the 7-point grid both
# crossings sit in interior intervals with a guard point on each side.
B2_LS, B2_NCS, B2_R = [32, 48], [512, 1024, 2048], 48
B2_GRID = list(B_GRID)                      # identical to campaign B's grid
# the subset that already exists at N_c = 1024 and is topped up, not recomputed
B2_EXISTING = [0.2232, 0.2332, 0.2432]

# --- Campaign C / D: the higher-L central ladders ---------------------------
C_L, C_LAM, C_NCS, C_R = 96, 0.3032, [1024, 2048], 24
D_L, D_LAM, D_NC, D_R = 128, 0.3032, 2048, 16

# --- Campaign E: discretisation / continuous-time particle-limit test -------
# dtau_mult is a DISCRETISATION CONTROL, not a physical parameter. The
# Feynman-Kac weight is exact at any window size, so the target measure is
# exactly unchanged across the three arms; only where selection is applied
# moves. K = 816 / 408 / 204 at this cell.
E_L, E_LAM, E_DTAUS, E_NCS, E_R = 64, 0.3032, [3.0, 6.0, 12.0], [64, 256], 48

# --- Conditional / gated designs (prepared, blocked) ------------------------
COND_D_NC, COND_D_R = 4096, 8
# Section 9A's 9-point candidate grid. Centre at delta_lambda = 0.005 with two
# outer guards, so a locator cannot be manufactured by a scan endpoint.
MOCK9_GRID = [0.2032, 0.2182, 0.2232, 0.2282, 0.2332, 0.2382, 0.2432, 0.2482,
              0.2632]
MOCK96_R_STAGE1, MOCK128_R_STAGE1 = 12, 8
# Design 2 of TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING. "Matched lambda" is
# read as THE SAME lambda, not the same distance from a putative lambda_c:
# matching on a critical-law offset would import the law under test.
LOWZ_ZETA, LOWZ_L, LOWZ_LAM, LOWZ_NCS, LOWZ_R = 0.10, 64, 0.3032, [64, 256], 48

# --- Seeds ------------------------------------------------------------------
# Every seed allocated anywhere in the repository -- including the manifests of
# arms that were never run -- is <= 32,203,023 (tools/dedup_scan.py verifies
# this against every manifest.csv and every result JSON on disk). The floor
# below is 796,977 above that ceiling, so disjointness is STRUCTURAL.
SEED_FLOOR = 33_000_000
SEED_CEIL = 34_000_000

# --- Scheduler --------------------------------------------------------------
# cpu_med where --time fits its 4 h MaxTime, cpu_long otherwise. cpu_short is
# never used: TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION/SCHEDULER_DECISION.md
# records that it is effectively serialised for this account by
# QOSMaxJobsPerUserLimit, so its concurrency cap is not real.
PARTITION_MAXTIME_H = {"cpu_short": 1.0, "cpu_med": 4.0, "cpu_long": 7 * 24.0}
CONCURRENCY = 64

# --- Frozen tolerances (SUCCESS_CRITERIA.yaml is the authority) -------------
# Both were fixed before any new datum exists. tau_lambda is primary and
# tau_I is derived from it, not the other way round: the programme needs a
# transition LOCATION, and an absolute CMI tolerance that is not tied to one is
# a number with no decision attached.
TAU_LAMBDA = 0.004      # crossing-location tolerance
DDDLAM_MIN = 2.965      # smallest |dD/dlambda| at an interior crossing, measured
TAU_D = 0.0118          # = TAU_LAMBDA * DDDLAM_MIN, tolerance on a cross-L difference
TAU_I = 0.006           # = TAU_D / 2, per-curve tolerance in the worst case
                        #   where the two curves' displacements do not cancel
