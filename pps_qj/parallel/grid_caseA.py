from __future__ import annotations

"""Case A production grid (two competing measurements, H = 0).

Task layout (lexicographic, L outer / lambda inner within each block):

  Block 1  ZETA < 1 core:   L in {32, 64, 128},  zeta in {0.10, 0.30, 0.50}
  Block 2  ZETA = 1 anchor:  L in {16, 32, 64, 128, 160},  zeta = 1.0

lambda_A mesh is uniform on [0.35, 0.65] (step 0.025) PLUS central refinement
{0.49, 0.495, 0.505, 0.51} so the crossing-slope dB_L/dlambda ~ L^{1/nu} and the
two-branch FSS collapse can resolve nu at the largest L (where the scaling
window is +/- L^{-1/nu} ~ +/- 0.008 at L=128).

Design notes
------------
* lambda_A = alpha / (alpha + gamma), alpha + gamma = 1, so
  alpha_rate (bond) = lambda, gamma_rate (site) = 1 - lambda.
* ZETA = 1 is the bias-free anchor: no cloning, no resampling, no ESS
  collapse, trustworthy at any N_c. It carries the extra sizes {16, 160}.
* ZETA < 1 uses cloning and is subject to the N_c ESS-collapse bias near
  criticality. A single N_c is NOT trusted at L=128 until the 2-rung check
  (N_c vs 2*N_c at L=128, zeta=0.30, lambda=0.50) shows B_L stable; see
  scripts/caseA_nc_rung_check + analysis/extrapolate_nc.py (model-agnostic).
* time_horizon_caseA has NO 5/alpha term: unlike Case B there is no vanishing
  measurement rate (gamma + alpha = 1 always). The mixing time is ~ L (z = 1).
  *** The T cap is provisional and MUST be confirmed by the B_L(T) saturation
  run at L=128, lambda=0.5, zeta=0.10 before the production submit. ***
"""

from typing import List

import numpy as np


# Uniform mesh + central refinement, rounded to 1e-4 so _seed stays integral.
_LAMBDAS = sorted(set(
    [round(float(x), 4) for x in np.linspace(0.35, 0.65, 13)]
    + [0.49, 0.495, 0.505, 0.51]
))

_ZETA_LT1 = [0.10, 0.30, 0.50]
_L_CORE = [32, 64, 128]          # zeta < 1 FSS core (Friedel-clean sizes)
_L_ZETA1 = [16, 32, 64, 128, 160]  # zeta = 1 anchor (bias-free, extra sizes)
# Backfill (2026-06-11): intermediate FSS sizes added at BOTH zeta<1 and the
# zeta=1 anchor. Appended as ids >= 238 so the already-run ids 0..237 (and the
# L=128 job in flight) keep their output files. See caseA_tier_ranges().
_L_BACKFILL = [48, 96]

N_REAL = 5


def alpha_gamma_from_lam(lam: float) -> tuple[float, float]:
    """Return (gamma_rate, alpha_rate) with alpha+gamma=1, lambda=alpha."""
    return float(1.0 - lam), float(lam)


def nc_for_L_caseA(L: int, zeta: float) -> int:
    """Clone / trajectory count per task.

    zeta = 1: independent Born trajectories (cheap, single-window) -> afford
    more. zeta < 1: cloning population; 300 base, gated by the 2-rung check.
    """
    if zeta >= 1.0:
        return {16: 1000, 32: 800, 48: 650, 64: 500, 96: 450, 128: 400, 160: 300}[L]
    return {32: 300, 48: 300, 64: 300, 96: 300, 128: 300}[L]


def time_horizon_caseA(L: int) -> float:
    """T(L) = max(30, 2L), capped at 160 for L >= 128.  PROVISIONAL CAP.

    No 5/alpha term (no vanishing rate in Case A). Cap to be confirmed by the
    L=128 saturation run before production.
    """
    T = max(30.0, 2.0 * L)
    if L >= 128:
        T = min(T, 160.0)
    return float(T)


def _seed(L: int, lam: float, zeta: float) -> int:
    # Offset +70e9 keeps Case A seeds disjoint from all Case B campaigns
    # (v1/v2/dense/rescue/ladder/slope use offsets up to +20e9).
    return int(70_000_000_000 + L * 10_000_000
               + round(lam * 1e4) * 1_000 + round(zeta * 1_000))


def _build_tasks() -> List[dict]:
    tasks: List[dict] = []
    for L in _L_CORE:                           # Block 1: zeta < 1 core, L outer
        for zeta in _ZETA_LT1:                  # so each L is a contiguous tier
            for lam in _LAMBDAS:
                gamma_rate, alpha_rate = alpha_gamma_from_lam(lam)
                tasks.append(dict(
                    L=int(L), lam=float(lam),
                    gamma_rate=gamma_rate, alpha_rate=alpha_rate,
                    zeta=float(zeta), T=time_horizon_caseA(L),
                    N_c=int(nc_for_L_caseA(L, zeta)),
                    seed=_seed(L, lam, zeta),
                ))
    for L in _L_ZETA1:                          # Block 2: zeta = 1 anchor
        for lam in _LAMBDAS:
            gamma_rate, alpha_rate = alpha_gamma_from_lam(lam)
            tasks.append(dict(
                L=int(L), lam=float(lam),
                gamma_rate=gamma_rate, alpha_rate=alpha_rate,
                zeta=1.0, T=time_horizon_caseA(L),
                N_c=int(nc_for_L_caseA(L, 1.0)),
                seed=_seed(L, lam, 1.0),
            ))
    # --- Backfill block (appended 2026-06-11). Ordered zeta<1 L=48, zeta<1
    # L=96, then zeta=1 {48,96} so each submit tier is a contiguous id range
    # (see caseA_tier_ranges()). Ids start at 238; ids 0..237 are unchanged.
    for L in _L_BACKFILL:                       # Block 3a: zeta < 1 backfill
        for zeta in _ZETA_LT1:
            for lam in _LAMBDAS:
                gamma_rate, alpha_rate = alpha_gamma_from_lam(lam)
                tasks.append(dict(
                    L=int(L), lam=float(lam),
                    gamma_rate=gamma_rate, alpha_rate=alpha_rate,
                    zeta=float(zeta), T=time_horizon_caseA(L),
                    N_c=int(nc_for_L_caseA(L, zeta)),
                    seed=_seed(L, lam, zeta),
                ))
    for L in _L_BACKFILL:                       # Block 3b: zeta = 1 backfill
        for lam in _LAMBDAS:
            gamma_rate, alpha_rate = alpha_gamma_from_lam(lam)
            tasks.append(dict(
                L=int(L), lam=float(lam),
                gamma_rate=gamma_rate, alpha_rate=alpha_rate,
                zeta=1.0, T=time_horizon_caseA(L),
                N_c=int(nc_for_L_caseA(L, 1.0)),
                seed=_seed(L, lam, 1.0),
            ))
    return tasks


_TASKS = _build_tasks()


def n_tasks_caseA() -> int:
    return len(_TASKS)


def caseA_tier_ranges() -> dict:
    """Contiguous (lo, hi) inclusive task-id ranges per submit tier.

    Tiers map to wall-time classes for the SLURM array:
      lt1_L32 / lt1_L64 / lt1_L128  -- zeta < 1 cloning at each core L
      zeta1                          -- the full zeta = 1 anchor (cheap)
    """
    nL = len(_LAMBDAS)
    per_L_lt1 = len(_ZETA_LT1) * nL
    ranges: dict = {}
    base = 0
    for L in _L_CORE:
        ranges[f"lt1_L{L}"] = (base, base + per_L_lt1 - 1)
        base += per_L_lt1
    ranges["zeta1"] = (base, base + len(_L_ZETA1) * nL - 1)
    base += len(_L_ZETA1) * nL                  # advance past zeta=1 anchor (-> 238)
    for L in _L_BACKFILL:                        # backfill zeta<1 tiers (lt1_L48, lt1_L96)
        ranges[f"lt1_L{L}"] = (base, base + per_L_lt1 - 1)
        base += per_L_lt1
    ranges["zeta1_bf"] = (base, base + len(_L_BACKFILL) * nL - 1)  # zeta=1 backfill {48,96}
    return ranges


def task_params_caseA(task_id: int) -> dict:
    if not (0 <= task_id < len(_TASKS)):
        raise IndexError(f"task_id {task_id} out of range [0, {len(_TASKS)})")
    return dict(_TASKS[task_id])


if __name__ == "__main__":
    print(f"Case A grid: {n_tasks_caseA()} tasks, N_REAL={N_REAL}")
    print(f"  lambdas ({len(_LAMBDAS)}): {_LAMBDAS}")
    print(f"  zeta<1 core L={_L_CORE} zeta={_ZETA_LT1}  "
          f"-> {len(_ZETA_LT1)*len(_L_CORE)*len(_LAMBDAS)} tasks")
    print(f"  zeta=1 anchor L={_L_ZETA1}  "
          f"-> {len(_L_ZETA1)*len(_LAMBDAS)} tasks")
    for tid in (0, len(_TASKS) // 2, len(_TASKS) - 1):
        print(f"  task {tid}: {task_params_caseA(tid)}")
