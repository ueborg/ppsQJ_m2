#!/usr/bin/env python3
"""Shared cost model for TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA.

Every rate in RATE_RUCHE is MEASURED on Ruche from a completed run of the
IDENTICAL production path, except L=64 which is DERIVED by two independent
routes that agree to 2%. Nothing here is inherited from the Mac-only figure
that TASK-2026-09-01-SMCRUCHE-READY had to warn about at +/-50%.

Provenance of every number is in COST_MODEL.md.
"""
import math

# --- seconds per clone-window, measured on Ruche --------------------------
# L=96  : median over the 48 completed ARM1 N_c=512 rows      -> 11.510 ms
#         (the N_c=128/256 rows give 11.705 / 10.120; the ladder is flat)
# L=128 : median over the 64 completed ARM2 N_c=256 rows      -> 21.522 ms
#         (N_c=64/128 give 27.176 / 26.805; the rate FALLS with N_c and
#          flattens by 256, so the N_c=256 value is the right one to
#          extrapolate to 512/1024 and the smaller-N_c values are not.)
# L=64  : DERIVED, two independent routes:
#           (a) within-Ruche L-scaling  11.510*(64/96)**2.174 = 4.773 ms
#               where 2.174 = ln(21.522/11.510)/ln(128/96)
#           (b) Mac->Ruche transfer     2.969 * (11.510/7.253) = 4.712 ms
#               (A-HV L=64 N_c=256 Mac rate; A-BUD L=96 N_c=64 Mac rate)
#         They agree to 1.3%. We adopt 5.000 ms, above both.
RATE_RUCHE_MS = {64: 5.000, 96: 11.510, 128: 21.522}

# Multiplicative band applied to every prediction when quoting the pessimistic
# figure. 1.40 covers (i) the observed max/median wall spread within a completed
# ARM2 rung (1.115) and (ii) rate uncertainty from extrapolating in N_c.
PESSIMISTIC = 1.40

DTAU_MULT = 6.0          # CERTIFIED production value. Never 12.


def n_steps(L, T, lam, dtau_mult=DTAU_MULT):
    """Exactly the production discretisation: instrumented.py line 127-128,
    with alpha == lam on this cut. Verified against measured n_steps
    (L=96,lam=0.3032 -> 922; L=128,lam=0.3032 -> 1643)."""
    dtau = dtau_mult / max(2.0 * lam * (L - 1), 1e-12)
    return max(1, int(math.ceil(T / dtau)))


def wall_s(L, T, N_c, lam, dtau_mult=DTAU_MULT):
    return RATE_RUCHE_MS[L] * 1e-3 * N_c * n_steps(L, T, lam, dtau_mult)


def mem_mb(L, N_c):
    """Same formula the predecessor preflight used and that was validated
    against ARM2's observed footprint."""
    per_clone = (2 * L) ** 2 * 8 + (2 * L) * L * 16
    return 128.0 + 2.0 * N_c * per_clone / 1e6
