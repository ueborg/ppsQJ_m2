"""Parameter grid for the PPS phase-diagram scan.

Defines the Doob-WTMC and cloning grids used by the SLURM array workers, plus
task <-> parameter helpers. Task ids are assigned in the deterministic order
(L outer, lambda middle, zeta inner) so that all tasks at a given L form a
contiguous range of ids — this lets the SLURM submit scripts select L-groups
with simple ``--array=LO-HI`` ranges.
"""
from __future__ import annotations

from typing import List

import numpy as np


# ----------------------------------------------------------------------
# Doob WTMC grid
# ----------------------------------------------------------------------
L_DOOB: List[int] = [16, 24, 32, 48, 64, 96, 128, 192, 256]
LAMBDA_VALS: List[float] = np.linspace(0.1, 0.9, 17).tolist()  # 17 points
ZETA_VALS_DOOB: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# ----------------------------------------------------------------------
# Cloning cross-validation grid
# ----------------------------------------------------------------------
L_CLONE: List[int] = [8, 16, 32, 64, 128]

ZETA_VALS_CLONE: List[float] = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.75, 0.80, 0.85,
    0.90, 0.92, 0.95, 0.97, 1.00,
]  # 15 points; denser near ζ=1 where the transition lives for most λ

# Finer lambda resolution near the phase boundary (λ ∈ [0.30, 0.55]).
# Use LAMBDA_VALS_FINE instead of LAMBDA_VALS when constructing an L=128
# targeted scan — adds 5 intermediate points where the (32,64) crossings live.
LAMBDA_VALS_FINE: List[float] = sorted(set(
    np.linspace(0.1, 0.9, 17).tolist() +
    [0.325, 0.375, 0.425, 0.475, 0.525]
))  # 22 points total


def _alpha_w_from_lam(lam: float) -> tuple[float, float]:
    """alpha + w = 1, lambda = alpha / (alpha + w) ⇒ alpha = lam, w = 1-lam."""
    return float(lam), float(1.0 - lam)


def n_traj_for_L(L: int) -> int:
    if L <= 48:
        return 1000
    if L in (64, 96):
        return 500
    if L in (128, 192):
        return 200
    if L == 256:
        return 100
    raise ValueError(f"n_traj not defined for L={L}")


def nc_for_L(L: int) -> int:
    """Clone population size N_c per task.

    Balances statistical quality against wall time on Habrok.

    L=64 wall-time note
    -------------------
    Profiling shows ~43.8 ms/clone at L=64 vs 1.51 ms/clone at L=32 —
    effectively L^4.8 scaling rather than L^3, due to L2-cache thrashing
    on the 256 KB model matrices (V, V_inv, VhV) in the *serial* clone
    loop.  N_c=200, N_REAL=5 gives 78.5 h/task at alpha=0.4 — infeasible
    on a single CPU core.

    The fix is intra-task multiprocessing (see ``run_cloning(n_workers=…)``
    in cloning.py and ``--cpus-per-task`` in submit_clone_scan.sh): with
    8 cores per task the worst-case L=64 task drops from 78.5 h to ~12 h,
    keeping N_c=200 and N_REAL=5 across the full grid.
    """
    return {8: 2000, 16: 1000, 32: 500, 64: 200, 128: 200}[L]


def time_horizon(L: int, alpha: float) -> float:
    """T(L, alpha) = max(30, 2L, 5/alpha), capped at 128 for L>=128."""
    base = max(30.0, 5.0 / max(alpha, 1e-9))
    propagation = 2.0 * L
    T = float(max(base, propagation))
    # Cap at 128 for L>=128: saturation checks confirm T=128 is sufficient,
    # and T=256 would exceed the 24h Habrok walltime limit per task.
    if L >= 128:
        T = min(T, 128.0)
    return T


def _seed(L: int, lam: float, zeta: float) -> int:
    return int(L * 10_000_000 + round(lam * 1e4) * 1_000 + round(zeta * 1_000))


def make_doob_grid() -> List[dict]:
    """All (L, lambda, zeta) combos for the Doob scan.

    Task ids are assigned in (L, lambda, zeta) lexicographic order; all tasks
    for a given L occupy a contiguous id range.
    """
    grid: List[dict] = []
    task_id = 0
    for L in L_DOOB:
        for lam in LAMBDA_VALS:
            for zeta in ZETA_VALS_DOOB:
                alpha, w = _alpha_w_from_lam(lam)
                T = time_horizon(L, alpha)
                grid.append(dict(
                    task_id=task_id,
                    L=int(L),
                    lam=float(lam),
                    alpha=alpha,
                    w=w,
                    zeta=float(zeta),
                    T=float(T),
                    n_traj=int(n_traj_for_L(L)),
                    seed=_seed(L, lam, zeta),
                ))
                task_id += 1
    return grid


def make_clone_grid() -> List[dict]:
    grid: List[dict] = []
    task_id = 0
    for L in L_CLONE:
        for lam in LAMBDA_VALS:
            for zeta in ZETA_VALS_CLONE:
                alpha, w = _alpha_w_from_lam(lam)
                T = time_horizon(L, alpha)
                grid.append(dict(
                    task_id=task_id,
                    L=int(L),
                    lam=float(lam),
                    alpha=alpha,
                    w=w,
                    zeta=float(zeta),
                    T=float(T),
                    N_c=int(nc_for_L(L)),
                    seed=_seed(L, lam, zeta),
                ))
                task_id += 1
    return grid


def n_tasks_doob() -> int:
    return len(L_DOOB) * len(LAMBDA_VALS) * len(ZETA_VALS_DOOB)


def n_tasks_clone() -> int:
    return len(L_CLONE) * len(LAMBDA_VALS) * len(ZETA_VALS_CLONE)


def clone_task_id_ranges() -> dict[int, tuple[int, int]]:
    """L -> (first_task_id, last_task_id) inclusive. Useful for SLURM arrays."""
    per_L = len(LAMBDA_VALS) * len(ZETA_VALS_CLONE)
    return {L: (i * per_L, (i + 1) * per_L - 1)
            for i, L in enumerate(L_CLONE)}


def task_params_doob(task_id: int) -> dict:
    grid = make_doob_grid()
    if not (0 <= task_id < len(grid)):
        raise IndexError(f"Doob task_id {task_id} out of range [0, {len(grid)})")
    return grid[task_id]


def task_params_clone(task_id: int) -> dict:
    grid = make_clone_grid()
    if not (0 <= task_id < len(grid)):
        raise IndexError(f"Clone task_id {task_id} out of range [0, {len(grid)})")
    return grid[task_id]


def doob_task_id_ranges() -> dict[int, tuple[int, int]]:
    """L -> (first_task_id, last_task_id) inclusive. Useful for SLURM arrays."""
    per_L = len(LAMBDA_VALS) * len(ZETA_VALS_DOOB)
    out: dict[int, tuple[int, int]] = {}
    for i, L in enumerate(L_DOOB):
        out[L] = (i * per_L, (i + 1) * per_L - 1)
    return out


# ======================================================================
# Cloning v2 grid — full production scan
# ======================================================================
# Design rationale
# ----------------
# Lambda grid:  extends LAMBDA_VALS_FINE with two very-small-λ points
#   (0.02, 0.05) to resolve the small-ζ phase boundary, predicted to lie
#   at λ_c(ζ) ~ Cζ (linear onset).  24 points total.
#
# Zeta grid:    10 points — dense for ζ ∈ [0.02, 0.20] to probe the Ising
#   plateau, then sparser for ζ ∈ [0.30, 1.00].  ζ=0 (pure no-click) is
#   handled separately if needed; ζ=0.02 is already extremely close.
#
# L sizes:      adds L=24, 48, 96 to the original set, as recommended for
#   finite-size scaling with non-power-of-two steps.  L=8 is kept only as a
#   sanity check (it is excluded from FSS fits).
#
# Total tasks:  8 sizes × 24 λ × 10 ζ = 1920 tasks.
#
# Compute budget (3 Habrok nodes, 120 cores each):
#   Node 1 — tasks   0..719  (L=8,16,24):  serial, 1 core/task, ≤12h
#   Node 2 — tasks 720..1199 (L=32,48):    serial, 1 core/task, ≤24h
#   Node 3 — tasks 1200..1919 (L=64,96,128): 5 cores/task, 24 concurrent,
#                                            ≤48h (worst case L=128, α=0.02)
# ======================================================================

L_CLONE_V2: List[int] = [8, 16, 24, 32, 48, 64, 96, 128]

# 24 λ points: very-small extension + LAMBDA_VALS_FINE
LAMBDA_VALS_V2: List[float] = sorted(set(
    [0.02, 0.05]
    + np.linspace(0.1, 0.9, 17).tolist()
    + [0.325, 0.375, 0.425, 0.475, 0.525]
))  # 24 points

ZETA_VALS_V2: List[float] = [
    0.02, 0.05, 0.10, 0.15, 0.20,  # dense small-ζ: probe Ising plateau
    0.30, 0.50, 0.70, 0.85, 1.00,  # coarser across the crossover and Born-rule end
]  # 10 points

# Per-task grid dimensions
_N_LAM_V2  = len(LAMBDA_VALS_V2)   # 24
_N_ZETA_V2 = len(ZETA_VALS_V2)     # 10
_PER_L_V2  = _N_LAM_V2 * _N_ZETA_V2  # 240


def nc_for_L_v2(L: int) -> int:
    """Clone population size for the v2 grid.

    New intermediate sizes (L=24,48,96) are interpolated conservatively.
    L=128 keeps N_c=100 — wall-time limited even with 5-core intra-task MP.
    """
    return {
        8:   2000,
        16:  1000,
        24:   800,
        32:   500,
        48:   300,
        64:   200,
        96:   150,
        128:  100,
    }[L]


def time_horizon_v2(L: int, alpha: float) -> float:
    """T(L, α) for the v2 grid, with tighter caps for large L.

    The existing time_horizon() allows T up to 5/α, which can reach 250 for
    α=0.02.  At L≥64 this is unaffordable.  Caps here keep the worst-case task
    at ≲48 h:
      L≥96  → T ≤ 100
      L≥64  → T ≤ 150
      L≥32  → T ≤ 200
      L<32  → uncapped (max 250, affordable)

    Steady-state saturation has been verified at these cap values for the
    relevant parameter ranges in earlier production runs.
    """
    base = max(30.0, 5.0 / max(alpha, 1e-9))
    T = float(max(base, 2.0 * L))
    if L >= 96:
        T = min(T, 100.0)
    elif L >= 64:
        T = min(T, 150.0)
    elif L >= 32:
        T = min(T, 200.0)
    return T


def make_clone_v2_grid() -> List[dict]:
    """All (L, λ, ζ) parameter combinations for the v2 production scan.

    Task ids are assigned in (L outer, λ middle, ζ inner) order, so tasks
    for each L form a contiguous block of 240 ids.
    """
    grid: List[dict] = []
    task_id = 0
    for L in L_CLONE_V2:
        for lam in LAMBDA_VALS_V2:
            for zeta in ZETA_VALS_V2:
                alpha, w = _alpha_w_from_lam(lam)
                T = time_horizon_v2(L, alpha)
                grid.append(dict(
                    task_id=task_id,
                    L=int(L),
                    lam=float(lam),
                    alpha=alpha,
                    w=w,
                    zeta=float(zeta),
                    T=float(T),
                    N_c=int(nc_for_L_v2(L)),
                    seed=_seed(L, lam, zeta),
                ))
                task_id += 1
    return grid


def task_params_clone_v2(task_id: int) -> dict:
    grid = make_clone_v2_grid()
    if not (0 <= task_id < len(grid)):
        raise IndexError(
            f"Clone-v2 task_id {task_id} out of range [0, {len(grid)})"
        )
    return grid[task_id]


def n_tasks_clone_v2() -> int:
    return len(L_CLONE_V2) * _N_LAM_V2 * _N_ZETA_V2  # 1920


def clone_v2_task_id_ranges() -> dict[int, tuple[int, int]]:
    """L -> (first_task_id, last_task_id) inclusive for the v2 grid."""
    return {
        L: (i * _PER_L_V2, (i + 1) * _PER_L_V2 - 1)
        for i, L in enumerate(L_CLONE_V2)
    }


# ======================================================================
# ζ=0 no-click benchmark grid
# ======================================================================
# Purpose: test the postselected corner scaling λ_cross(L, ζ=0) ~ L^{-1/2}.
# The simulation is fully deterministic (no random jumps), so N_REAL=1 per
# task and the total cost is negligible.
#
# λ range: [0.01, 0.3] — only the lower half of the phase diagram is needed
#   since the transition at ζ=0 is near λ=0 for large L.
# T: capped at 500 (single-shot propagation under H_eff; cost is O(L^3)
#   independent of T via precomputed matrix exponential).
# Total tasks: 8 × 9 = 72.
# ======================================================================

L_ZETA0: List[int] = [8, 16, 24, 32, 48, 64, 96, 128]

LAMBDA_VALS_ZETA0: List[float] = [
    0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30
]  # 9 points, dense near 0 for λ_cross(L) ~ L^{-1/2} test


def time_horizon_zeta0(L: int, alpha: float) -> float:
    """T for the deterministic no-click evolution.

    The convergence time to the H_eff ground state scales as L/α
    (O(L) modes each damping at rate ~α).  The original cap of 500 was
    too short for small α and large L.  New formula:

        T = min(20000, max(15·L, 200/α))

    For (L=128, α=0.01): T = min(20000, max(1920, 20000)) = 20000  (~24s)
    For (L=128, α=0.03): T = min(20000, max(1920,  6667)) =  6667  (~8s)
    For (L=96,  α=0.02): T = min(20000, max(1440, 10000)) = 10000  (~3s)
    For (L=8,   α=0.01): T = min(20000, max( 120, 20000)) = 20000  (<1s)

    All tasks remain computationally trivial.
    """
    return float(min(20000.0, max(15.0 * L, 200.0 / max(alpha, 1e-9))))


def make_zeta0_grid() -> List[dict]:
    """All (L, λ) combinations for the ζ=0 no-click benchmark."""
    grid: List[dict] = []
    task_id = 0
    for L in L_ZETA0:
        for lam in LAMBDA_VALS_ZETA0:
            alpha, w = _alpha_w_from_lam(lam)
            T = time_horizon_zeta0(L, alpha)
            grid.append(dict(
                task_id=task_id, L=int(L), lam=float(lam),
                alpha=alpha, w=w, T=float(T),
                seed=_seed(L, lam, 0.0),
            ))
            task_id += 1
    return grid


def task_params_zeta0(task_id: int) -> dict:
    grid = make_zeta0_grid()
    if not (0 <= task_id < len(grid)):
        raise IndexError(f"ζ=0 task_id {task_id} out of range [0, {len(grid)})")
    return grid[task_id]


def n_tasks_zeta0() -> int:
    return len(L_ZETA0) * len(LAMBDA_VALS_ZETA0)  # 72


def zeta0_task_id_ranges() -> dict[int, tuple[int, int]]:
    per_L = len(LAMBDA_VALS_ZETA0)
    return {L: (i * per_L, (i + 1) * per_L - 1) for i, L in enumerate(L_ZETA0)}


# ======================================================================
# Clone v2 supplement grid (300 tasks)
# ======================================================================
# Two supplementary blocks:
#
#   Block A (tasks 0..59):   low-λ small-ζ — resolves the phase boundary
#     where λ_c(ζ) ~ Cζ may fall below the v2 grid's λ_min=0.02.
#     L=[32,48,64,96,128], λ=[0.005,0.01,0.03,0.075], ζ=[0.02,0.05,0.10]
#
#   Block B (tasks 60..299): large-ζ — adds ζ=0.90,0.95 for the near-
#     Born-rule linear-response test λ_c(ζ)=λ_c(1)-A(1-ζ)+O((1-ζ)^2).
#     L=[32,48,64,96,128], full 24-point λ grid, ζ=[0.90,0.95]
# ======================================================================

L_SUPP: List[int] = [32, 48, 64, 96, 128]

LAMBDA_VALS_SUPP_LOWLAM: List[float] = [0.005, 0.01, 0.03, 0.075]  # 4 pts
ZETA_VALS_SUPP_LOWLAM: List[float] = [0.02, 0.05, 0.10]             # 3 pts
ZETA_VALS_LARGE: List[float] = [0.90, 0.95]                          # 2 new pts

_N_SUPP_LOW      = len(L_SUPP) * len(LAMBDA_VALS_SUPP_LOWLAM) * len(ZETA_VALS_SUPP_LOWLAM)  # 60
_N_SUPP_LARGE_Z  = len(L_SUPP) * _N_LAM_V2 * len(ZETA_VALS_LARGE)                           # 240
_N_SUPP_TOTAL    = _N_SUPP_LOW + _N_SUPP_LARGE_Z                                             # 300


def make_clone_v2_supp_grid() -> List[dict]:
    """Supplement tasks: low-λ small-ζ block + large-ζ block.

    Uses the same N_c and time_horizon_v2 as the main v2 grid.
    """
    grid: List[dict] = []
    task_id = 0
    # --- Block A: low-λ, small ζ ---
    for L in L_SUPP:
        for lam in LAMBDA_VALS_SUPP_LOWLAM:
            for zeta in ZETA_VALS_SUPP_LOWLAM:
                alpha, w = _alpha_w_from_lam(lam)
                T = time_horizon_v2(L, alpha)
                grid.append(dict(
                    task_id=task_id, L=int(L), lam=float(lam),
                    alpha=alpha, w=w, zeta=float(zeta),
                    T=float(T), N_c=int(nc_for_L_v2(L)),
                    seed=_seed(L, lam, zeta),
                ))
                task_id += 1
    # --- Block B: large ζ ---
    for L in L_SUPP:
        for lam in LAMBDA_VALS_V2:
            for zeta in ZETA_VALS_LARGE:
                alpha, w = _alpha_w_from_lam(lam)
                T = time_horizon_v2(L, alpha)
                grid.append(dict(
                    task_id=task_id, L=int(L), lam=float(lam),
                    alpha=alpha, w=w, zeta=float(zeta),
                    T=float(T), N_c=int(nc_for_L_v2(L)),
                    seed=_seed(L, lam, zeta),
                ))
                task_id += 1
    return grid


def task_params_clone_v2_supp(task_id: int) -> dict:
    grid = make_clone_v2_supp_grid()
    if not (0 <= task_id < len(grid)):
        raise IndexError(
            f"Clone-v2-supp task_id {task_id} out of range [0, {len(grid)})"
        )
    return grid[task_id]


def n_tasks_clone_v2_supp() -> int:
    return _N_SUPP_TOTAL  # 300


def clone_v2_supp_task_id_ranges() -> dict[str, tuple[int, int]]:
    """Block-level ranges for the supplement grid."""
    return {
        "low_lam":  (0,                 _N_SUPP_LOW - 1),
        "large_z":  (_N_SUPP_LOW,       _N_SUPP_TOTAL - 1),
    }


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "doob"
    if which == "doob":
        g = make_doob_grid()
        print(f"n_tasks_doob = {n_tasks_doob()} (expected 1530)")
        print(f"first: {g[0]}")
        print(f"last : {g[-1]}")
        print("L task_id ranges:")
        for L, (lo, hi) in doob_task_id_ranges().items():
            print(f"  L={L:4d}  tasks {lo}..{hi}")
    elif which == "clone":
        g = make_clone_grid()
        print(f"n_tasks_clone = {n_tasks_clone()} (expected 1020)")
        print(f"first: {g[0]}")
        print(f"last : {g[-1]}")
        print(f"L=64 N_c: {nc_for_L(64)}")
    elif which == "clone_v2":
        g = make_clone_v2_grid()
        print(f"n_tasks_clone_v2 = {n_tasks_clone_v2()} (expected 1920)")
        print(f"first: {g[0]}")
        print(f"last : {g[-1]}")
        print(f"lambda grid ({_N_LAM_V2} pts): {LAMBDA_VALS_V2}")
        print(f"zeta  grid ({_N_ZETA_V2} pts): {ZETA_VALS_V2}")
        print("L task_id ranges:")
        for L, (lo, hi) in clone_v2_task_id_ranges().items():
            print(f"  L={L:4d}  tasks {lo:4d}..{hi:4d}  N_c={nc_for_L_v2(L)}")
    elif which == "clone_v2_supp":
        g = make_clone_v2_supp_grid()
        print(f"n_tasks_clone_v2_supp = {n_tasks_clone_v2_supp()} (expected 300)")
        print(f"first: {g[0]}")
        print(f"last : {g[-1]}")
        for name, (lo, hi) in clone_v2_supp_task_id_ranges().items():
            print(f"  {name}: tasks {lo}..{hi}")
    elif which == "zeta0":
        g = make_zeta0_grid()
        print(f"n_tasks_zeta0 = {n_tasks_zeta0()} (expected 72)")
        print(f"first: {g[0]}")
        print(f"last : {g[-1]}")
        print(f"lambda grid ({len(LAMBDA_VALS_ZETA0)} pts): {LAMBDA_VALS_ZETA0}")
        print("L task_id ranges:")
        for L, (lo, hi) in zeta0_task_id_ranges().items():
            print(f"  L={L:4d}  tasks {lo:2d}..{hi:2d}")
    elif which == "clone_dense":
        g = make_clone_dense_grid()
        print(f"n_tasks_clone_dense = {n_tasks_clone_dense()} (expected 4112)")
        print(f"first: {g[0]}")
        print(f"last : {g[-1]}")
        print(f"λ grid SMALL  ({len(LAMBDA_VALS_DENSE_SMALL)} pts): {[round(x,4) for x in LAMBDA_VALS_DENSE_SMALL]}")
        print(f"λ grid MEDIUM ({len(LAMBDA_VALS_DENSE_MEDIUM)} pts): {[round(x,4) for x in LAMBDA_VALS_DENSE_MEDIUM]}")
        print(f"λ grid LARGE  ({len(LAMBDA_VALS_DENSE_LARGE)} pts): {[round(x,4) for x in LAMBDA_VALS_DENSE_LARGE]}")
        print(f"ζ grid SMALL  ({len(ZETA_VALS_DENSE_SMALL)} pts): {ZETA_VALS_DENSE_SMALL}")
        print(f"ζ grid MEDIUM ({len(ZETA_VALS_DENSE_MEDIUM)} pts): {ZETA_VALS_DENSE_MEDIUM}")
        print(f"ζ grid LARGE  ({len(ZETA_VALS_DENSE_LARGE)} pts): {ZETA_VALS_DENSE_LARGE}")
        print(f"Per-L tasks: {_N_DENSE_PER_L} (= {len(LAMBDA_VALS_DENSE_SMALL)}×{len(ZETA_VALS_DENSE_SMALL)} "
              f"+ {len(LAMBDA_VALS_DENSE_MEDIUM)}×{len(ZETA_VALS_DENSE_MEDIUM)} "
              f"+ {len(LAMBDA_VALS_DENSE_LARGE)}×{len(ZETA_VALS_DENSE_LARGE)})")
        print("L task_id ranges:")
        for L, (lo, hi) in clone_dense_task_id_ranges().items():
            print(f"  L={L:4d}  tasks {lo:4d}..{hi:4d}  N_c={nc_for_L_dense(L)}")
    else:
        print(f"unknown grid type: {which}")
        sys.exit(1)


# ======================================================================
# Clone v2 finite-size test grid (84 tasks) — L=192, L=256 extension
# ======================================================================
# Purpose
# -------
# Distinguish two scenarios for the empirical separatrix at zeta ~ 0.143
# observed in the L<=128 aggregate:
#
#   Scenario A (Ising universality, lambda_c=0.5 robust for zeta>0):
#     the apparent separatrix is a non-universal correction-to-scaling
#     feature where the leading 1/L amplitude B(zeta) crosses zero.
#     At L=192,256 the drift in lambda_c should REVERSE at small zeta
#     and start pointing back toward lambda_c=0.5.
#
#   Scenario B (genuine 2D phase diagram):
#     the separatrix at zeta_c~0.143 is a true RG fixed point.
#     The drift sign at small zeta stays negative; lambda_c flows
#     toward 0 in the thermodynamic limit.
#
# Grid
# ----
#   L:    [192, 256]                                          (2 sizes)
#   lam:  [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]          (7 points)
#         focused on the empirical lambda_c range.
#   zeta: [0.05, 0.10, 0.14, 0.18, 0.50, 1.00]                (6 points)
#         four densely cover the separatrix region 0.10..0.18,
#         plus zeta=0.05 (deep into the apparently-area-law region) and
#         zeta=0.50, 1.00 as Ising anchors.
#
# Total: 2 x 7 x 6 = 84 tasks.  Task ids 0..83.
#
# N_c and time horizons
# ---------------------
# Aggressive reductions vs the v2 main grid; the goal here is to extract
# c_eff at a fixed handful of (lam, zeta) points and compare against the
# (32,64) and (64,128) drift-sign pattern, not a fully-converged scan.
#
#   L=192: N_c=80,  T_cap=80
#   L=256: N_c=40,  T_cap=50
#
# Worst-case wall time at L=256, alpha=0.05 (slowest task):
#   T=50, dtau ~ 0.039, K ~ 1280 steps
#   per step at L=256 (L^4.8 cache-thrashing estimate): ~11 s
#   per task ~ 1280 * 11 * 40 / 5 = 113 000 s = 31 h
#   Wall time 36:00:00 covers this with margin.
# ======================================================================

L_CLONE_FST: List[int] = [192, 256]

LAMBDA_VALS_FST: List[float] = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
]  # 7 points focused around the empirical lambda_c range

ZETA_VALS_FST: List[float] = [
    0.05, 0.10, 0.14, 0.18,  # dense across the separatrix
    0.50, 1.00,              # Ising anchors
]  # 6 points

_N_LAM_FST  = len(LAMBDA_VALS_FST)   # 7
_N_ZETA_FST = len(ZETA_VALS_FST)     # 6
_PER_L_FST  = _N_LAM_FST * _N_ZETA_FST  # 42


def nc_for_L_fst(L: int) -> int:
    return {192: 80, 256: 40}[L]


def time_horizon_fst(L: int, alpha: float) -> float:
    """Tighter cap than time_horizon_v2: T<=80 for L=192, T<=50 for L=256."""
    base = max(30.0, 5.0 / max(alpha, 1e-9))
    T = float(max(base, 2.0 * L))
    if L >= 256:
        T = min(T, 50.0)
    elif L >= 192:
        T = min(T, 80.0)
    return T


def make_clone_fst_grid() -> List[dict]:
    """All (L, lambda, zeta) parameter combinations for the FST grid.

    Task ids are assigned in (L outer, lambda middle, zeta inner) order, so
    tasks for each L form a contiguous block of 42 ids.
    """
    grid: List[dict] = []
    task_id = 0
    for L in L_CLONE_FST:
        for lam in LAMBDA_VALS_FST:
            for zeta in ZETA_VALS_FST:
                alpha, w = _alpha_w_from_lam(lam)
                T = time_horizon_fst(L, alpha)
                grid.append(dict(
                    task_id=task_id,
                    L=int(L),
                    lam=float(lam),
                    alpha=alpha,
                    w=w,
                    zeta=float(zeta),
                    T=float(T),
                    N_c=int(nc_for_L_fst(L)),
                    seed=_seed(L, lam, zeta),
                ))
                task_id += 1
    return grid


def task_params_clone_fst(task_id: int) -> dict:
    grid = make_clone_fst_grid()
    if not (0 <= task_id < len(grid)):
        raise IndexError(
            f"Clone-FST task_id {task_id} out of range [0, {len(grid)})"
        )
    return grid[task_id]


def n_tasks_clone_fst() -> int:
    return len(L_CLONE_FST) * _N_LAM_FST * _N_ZETA_FST  # 84


def clone_fst_task_id_ranges() -> dict[int, tuple[int, int]]:
    """L -> (first_task_id, last_task_id) inclusive for the FST grid."""
    return {
        L: (i * _PER_L_FST, (i + 1) * _PER_L_FST - 1)
        for i, L in enumerate(L_CLONE_FST)
    }


# ======================================================================
# Slope-test grid (added 2026-05-26)
# ======================================================================
# Purpose: discriminate Möbius slope 1/8 from naive NLSM slope 1/4 at
# ζ → 1, by providing precise λ_c(ζ) at ζ ∈ {0.70,0.80,0.90}.
#
# Design:
#   Part A: ζ ∈ {0.80, 0.90} at L ∈ {8..128}     — fills gaps in v2 grid
#   Part B: ζ ∈ {0.70, 0.80, 0.90} at L ∈ {192,256} — precision step
#
# Task layout (L outer, λ middle, ζ inner):
#   Part A: 8 L-values × 24 λ × 2 ζ  = 384 tasks  (IDs   0..383)
#   Part B: 2 L-values × 24 λ × 3 ζ  = 144 tasks  (IDs 384..527)
#   Total: 528 tasks
#
# λ-points: reuse LAMBDA_VALS_V2 (24 points, includes fine grid near λ_c)
# N_c / time_horizon: reuse nc_for_L_v2 for L≤128, nc_for_L_fst for L≥192
# Seeds: unique — ζ=0.80,0.90 were absent from all prior grids.
# ======================================================================

_ZETA_SLOPE_SMALL: List[float] = [0.80, 0.90]
_ZETA_SLOPE_LARGE: List[float] = [0.70, 0.80, 0.90]
_L_SLOPE_SMALL: List[int]      = L_CLONE_V2           # [8,16,24,32,48,64,96,128]
_L_SLOPE_LARGE: List[int]      = [192, 256]

_N_SLOPE_SMALL = len(_L_SLOPE_SMALL) * len(LAMBDA_VALS_V2) * len(_ZETA_SLOPE_SMALL)  # 384
_N_SLOPE_LARGE = len(_L_SLOPE_LARGE) * len(LAMBDA_VALS_V2) * len(_ZETA_SLOPE_LARGE)  # 144
_N_SLOPE_TOTAL = _N_SLOPE_SMALL + _N_SLOPE_LARGE  # 528


def _nc_slope(L: int) -> int:
    """N_c for the slope-test grid, consistent with existing schedules."""
    if L in {8, 16, 24, 32, 48, 64, 96, 128}:
        return nc_for_L_v2(L)
    return {192: 80, 256: 40}[L]


def _time_horizon_slope(L: int, alpha: float) -> float:
    """Time horizon for slope-test grid."""
    if L <= 128:
        return time_horizon_v2(L, alpha)
    return time_horizon_fst(L, alpha)


def make_clone_slope_grid() -> List[dict]:
    """All (L, λ, ζ) parameter combinations for the slope-test scan.

    Task ids are assigned in (L outer, λ middle, ζ inner) order.
    Part A fills task ids 0..383; Part B fills 384..527.
    """
    grid: List[dict] = []
    task_id = 0

    # Part A: new ζ values at existing L≤128 sizes
    for L in _L_SLOPE_SMALL:
        for lam in LAMBDA_VALS_V2:
            for zeta in _ZETA_SLOPE_SMALL:
                alpha, w = _alpha_w_from_lam(lam)
                T = _time_horizon_slope(L, alpha)
                grid.append(dict(
                    task_id=task_id,
                    L=int(L),
                    lam=float(lam),
                    alpha=alpha,
                    w=w,
                    zeta=float(zeta),
                    T=float(T),
                    N_c=int(_nc_slope(L)),
                    seed=_seed(L, lam, zeta),
                ))
                task_id += 1

    # Part B: large-L precision runs
    for L in _L_SLOPE_LARGE:
        for lam in LAMBDA_VALS_V2:
            for zeta in _ZETA_SLOPE_LARGE:
                alpha, w = _alpha_w_from_lam(lam)
                T = _time_horizon_slope(L, alpha)
                grid.append(dict(
                    task_id=task_id,
                    L=int(L),
                    lam=float(lam),
                    alpha=alpha,
                    w=w,
                    zeta=float(zeta),
                    T=float(T),
                    N_c=int(_nc_slope(L)),
                    seed=_seed(L, lam, zeta),
                ))
                task_id += 1

    assert len(grid) == _N_SLOPE_TOTAL, (
        f"Slope grid size mismatch: {len(grid)} != {_N_SLOPE_TOTAL}"
    )
    return grid


def task_params_clone_slope(task_id: int) -> dict:
    grid = make_clone_slope_grid()
    if not (0 <= task_id < len(grid)):
        raise IndexError(
            f"Slope task_id {task_id} out of range [0, {len(grid)})"
        )
    return grid[task_id]


def n_tasks_clone_slope() -> int:
    return _N_SLOPE_TOTAL  # 528


def clone_slope_task_id_ranges() -> dict[int, tuple[int, int]]:
    """L -> (first_task_id, last_task_id) inclusive for the slope grid.

    Returns a dict for all L values in the grid (both Part A and Part B).
    """
    out: dict[int, tuple[int, int]] = {}
    grid = make_clone_slope_grid()
    from collections import defaultdict
    by_L: dict[int, list[int]] = defaultdict(list)
    for p in grid:
        by_L[p["L"]].append(p["task_id"])
    for L, ids in sorted(by_L.items()):
        out[L] = (min(ids), max(ids))
    return out



# ======================================================================
# Clone-dense grid (added 2026-05-27)
# ======================================================================
# Purpose: high-statistics fine-grid scan for λ_c(ζ) phase-boundary
# refinement and clean CMI / Renyi crossing analysis.  Builds on v2 data
# but with:
#   • 21 ζ values (5 small, 8 medium, 8 large) — vs. v2's 10
#   • Dense λ grid in the predicted crossing band per ζ region
#   • 2×/2×/3× N_c bump (asymmetric, weighted to larger L)
#   • Full CMI tripartition components saved per (L, λ, ζ)
#   • Renyi-2 and Renyi-3 entropies (PPS_RECORD_RENYI=1 required at submit)
#
# Task ordering (L outer, region middle, ζ inner) — see
# make_clone_dense_grid() for the exact layout.  Each L contributes
# 130 + 192 + 192 = 514 tasks, giving 4112 total.
#
# L task_id ranges:
#   L=8:    0..  513     L=48: 2056..2569
#   L=16: 514..1027     L=64: 2570..3083
#   L=24:1028..1541     L=96: 3084..3597
#   L=32:1542..2055     L=128:3598..4111
#
# Per-L total = 130 (small) + 192 (medium) + 192 (large) = 514 tasks
# ======================================================================

L_CLONE_DENSE: List[int] = [8, 16, 24, 32, 48, 64, 96, 128]

# --- λ grids per ζ region ----------------------------------------------
# Small ζ (predicted λ_c ≈ 0.06–0.18): dense low-λ coverage
LAMBDA_VALS_DENSE_SMALL: List[float] = sorted(set(
    np.linspace(0.01, 0.06, 6).tolist()      # 6 dense low-λ points
    + np.linspace(0.07, 0.30, 16).tolist()    # 16 in the crossing band
    + np.linspace(0.35, 0.60, 4).tolist()     # 4 in area phase for crossing
))  # 26 λ-points (after dedup/sort)

# Medium ζ (predicted λ_c ≈ 0.25–0.45): crossing band recentred
LAMBDA_VALS_DENSE_MEDIUM: List[float] = sorted(set(
    np.linspace(0.10, 0.20, 4).tolist()
    + np.linspace(0.22, 0.45, 16).tolist()    # crossing band
    + np.linspace(0.48, 0.65, 4).tolist()
))  # 24 λ-points

# Large ζ (predicted λ_c ≈ 0.45–0.50, approaching Born endpoint):
LAMBDA_VALS_DENSE_LARGE: List[float] = sorted(set(
    np.linspace(0.20, 0.35, 4).tolist()
    + np.linspace(0.38, 0.55, 16).tolist()    # crossing band around λ_c~0.45–0.50
    + np.linspace(0.58, 0.70, 4).tolist()
))  # 24 λ-points

# --- ζ grids -----------------------------------------------------------
ZETA_VALS_DENSE_SMALL:  List[float] = [0.02, 0.05, 0.08, 0.10, 0.15]                                 # 5
ZETA_VALS_DENSE_MEDIUM: List[float] = [0.18, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]               # 8
ZETA_VALS_DENSE_LARGE:  List[float] = [0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.92, 1.00]               # 8

_N_DENSE_PER_L = (
    len(LAMBDA_VALS_DENSE_SMALL)  * len(ZETA_VALS_DENSE_SMALL)
    + len(LAMBDA_VALS_DENSE_MEDIUM) * len(ZETA_VALS_DENSE_MEDIUM)
    + len(LAMBDA_VALS_DENSE_LARGE)  * len(ZETA_VALS_DENSE_LARGE)
)  # = 26*5 + 24*8 + 24*8 = 130 + 192 + 192 = 514

# Seed offset to keep dense seeds disjoint from v2/FST/slope seeds.
# Base _seed(L, lam, zeta) values reach ~1.3e9 at L=128; realisation seeds
# in worker extend by +r*999_983 for r ∈ [0, N_REAL=5).  Adding 7e9 puts
# all dense (base + realisation) seeds in [7e9, 8.4e9], safely disjoint
# from v2/FST/slope (≤ ~1.3e9).
_DENSE_SEED_OFFSET = 7_000_000_000


def _seed_dense(L: int, lam: float, zeta: float) -> int:
    """Seed for the dense grid, offset to guarantee independence from v2."""
    return _seed(L, lam, zeta) + _DENSE_SEED_OFFSET


def nc_for_L_dense(L: int) -> int:
    """N_c schedule for the dense grid.

    Asymmetric 2×/2×/3× bump weighted toward larger L (vs. v2 schedule):
      L=8:  4000  (2× of v2 2000)        L=48:  600  (2×)
      L=16: 2000  (2×)                    L=64:  400  (2×)
      L=24: 1600  (2×)                    L=96:  450  (3×)
      L=32: 1000  (2×)                    L=128: 300  (3×)

    Statistical errors on Binder cumulants scale as 1/sqrt(N_c), so
    √2 ≈ 1.41× tighter at small/medium L and √3 ≈ 1.73× at L=96, 128
    where v2 was statistics-limited.
    """
    return {
        8:   4000,
        16:  2000,
        24:  1600,
        32:  1000,
        48:   600,
        64:   400,
        96:   450,
        128:  300,
    }[L]


def time_horizon_dense(L: int, alpha: float) -> float:
    """T(L, α) for the dense grid — same caps as v2.

    The v2 caps were derived from steady-state convergence tests and have
    not been re-derived for higher N_c; same caps apply since saturation
    time is independent of N_c.
    """
    return time_horizon_v2(L, alpha)


def make_clone_dense_grid() -> List[dict]:
    """Unified dense grid: 8 L × (5+8+8) ζ × (26 or 24) λ = 4112 tasks.

    Task ordering: L outer, region middle (small→medium→large), ζ inner.
    Within each region, λ middle / ζ inner.

    L task-ID layout:
      L_i contributes tasks [i*514, (i+1)*514).  Within an L-block:
        - tasks [0, 130)   : small-ζ region (26 λ × 5 ζ)
        - tasks [130, 322) : medium-ζ region (24 λ × 8 ζ)
        - tasks [322, 514) : large-ζ region (24 λ × 8 ζ)
    """
    grid: List[dict] = []
    task_id = 0
    for L in L_CLONE_DENSE:
        N_c = int(nc_for_L_dense(L))

        # Small-ζ block
        for lam in LAMBDA_VALS_DENSE_SMALL:
            for zeta in ZETA_VALS_DENSE_SMALL:
                alpha, w = _alpha_w_from_lam(lam)
                T = time_horizon_dense(L, alpha)
                grid.append(dict(
                    task_id=task_id, L=int(L), lam=float(lam),
                    alpha=alpha, w=w, zeta=float(zeta),
                    T=float(T), N_c=N_c,
                    seed=_seed_dense(L, lam, zeta),
                ))
                task_id += 1

        # Medium-ζ block
        for lam in LAMBDA_VALS_DENSE_MEDIUM:
            for zeta in ZETA_VALS_DENSE_MEDIUM:
                alpha, w = _alpha_w_from_lam(lam)
                T = time_horizon_dense(L, alpha)
                grid.append(dict(
                    task_id=task_id, L=int(L), lam=float(lam),
                    alpha=alpha, w=w, zeta=float(zeta),
                    T=float(T), N_c=N_c,
                    seed=_seed_dense(L, lam, zeta),
                ))
                task_id += 1

        # Large-ζ block
        for lam in LAMBDA_VALS_DENSE_LARGE:
            for zeta in ZETA_VALS_DENSE_LARGE:
                alpha, w = _alpha_w_from_lam(lam)
                T = time_horizon_dense(L, alpha)
                grid.append(dict(
                    task_id=task_id, L=int(L), lam=float(lam),
                    alpha=alpha, w=w, zeta=float(zeta),
                    T=float(T), N_c=N_c,
                    seed=_seed_dense(L, lam, zeta),
                ))
                task_id += 1

    expected = len(L_CLONE_DENSE) * _N_DENSE_PER_L
    assert len(grid) == expected, (
        f"Dense grid size mismatch: {len(grid)} != {expected}"
    )
    return grid


def task_params_clone_dense(task_id: int) -> dict:
    grid = make_clone_dense_grid()
    if not (0 <= task_id < len(grid)):
        raise IndexError(
            f"Dense task_id {task_id} out of range [0, {len(grid)})"
        )
    return grid[task_id]


def n_tasks_clone_dense() -> int:
    return len(L_CLONE_DENSE) * _N_DENSE_PER_L  # = 4112


def clone_dense_task_id_ranges() -> dict[int, tuple[int, int]]:
    """L -> (first_task_id, last_task_id) inclusive for the dense grid."""
    return {
        L: (i * _N_DENSE_PER_L, (i + 1) * _N_DENSE_PER_L - 1)
        for i, L in enumerate(L_CLONE_DENSE)
    }



# ======================================================================
# Clone Phase-2 supplement grid (added 2026-05-29)
# ======================================================================
# Purpose: narrow, deep, high-L scan to break the L<=128 finite-size
# ceiling and make the phi=1/2 vs phi=1 question conclusive.
#
# Physics rationale (see collaborator spec):
#   The crossing g_c(L,2L) is biased while L < xi_nc ~ g_c^{-2}.  To put
#   the crossing in the asymptotic regime we need L well above xi_nc.  In
#   the decisive window zeta in [0.10, 0.30] the two hypotheses differ by
#   ~3x in g_c and xi_nc ~ 10-100, so L = 160, 192, 256 actually exceeds
#   xi_nc there (unlike at smaller zeta, where a fake sqrt-zeta signal can
#   appear).  Three new L values (160, 192, 256) paired with the dense
#   L=128 give a 4-point 1/L extrapolation that can FIT the correction-to-
#   scaling exponent omega rather than assume it.
#
#   N_c is set high (400/300/250) so the B_L relative error in the
#   critical band is ~5%, giving crossing precision delta-lambda_c <~ 0.02
#   -- enough to separate g_c=0.20 (linear) from g_c=0.32 (sqrt) at
#   zeta=0.10.
#
# CRITICAL DEPENDENCY -- READ THIS:
#   The lambda-mesh is NARROW (13 points, +/-0.06) and must be CENTERED on
#   the dense L<=128 crossing lambda_c^{(128)}(zeta).  Those values are NOT
#   known until the dense campaign (submit_clone_dense_*.sh) finishes and
#   is analysed.  The table _PHASE2_LAMBDA_C_128 below ships with
#   PLACEHOLDER values from the sqrt-zeta fit.  Submitting with placeholders
#   risks centering the expensive scan on the wrong lambda and wasting the
#   run (the retraction table has lambda_c(0.20) ~ 0.43, far from the fit's
#   0.29 -- a +/-0.06 window centered wrong would MISS the crossing).
#
#   WORKFLOW:
#     1. Let dense L<=128 jobs finish.
#     2. Extract lambda_c^{(128)}(zeta) from B_L / CMI crossings for
#        zeta in {0.10, 0.15, 0.20, 0.25, 0.30}.
#     3. Edit _PHASE2_LAMBDA_C_128 with those values.
#     4. Set _PHASE2_CENTERS_VERIFIED = True.
#     5. (optional) widen _PHASE2_LAMBDA_HALFWIDTH if any center is
#        uncertain by more than ~0.04.
#     6. THEN submit.
#
# Task layout (L outer, zeta middle, lambda inner): 5 zeta x 13 lambda = 65
# per L; 3 L -> 195 tasks.
#   L=160: tasks   0.. 64    N_c=400
#   L=192: tasks  65..129    N_c=300
#   L=256: tasks 130..194    N_c=250
# ======================================================================

# ---- EDIT THIS BLOCK AFTER THE DENSE L<=128 ANALYSIS -------------------
# lambda_c^{(128)}(zeta) crossing locations.  PLACEHOLDER values below are
# from the C*sqrt(zeta)/(1+C*sqrt(zeta)) fit with C=0.91 -- replace with
# the measured dense crossings before submitting.
_PHASE2_LAMBDA_C_128: dict[float, float] = {
    0.10: 0.223,   # PLACEHOLDER (sqrt-zeta fit). Dense retraction gave ~0.232.
    0.15: 0.261,   # PLACEHOLDER (sqrt-zeta fit). No direct dense value yet.
    0.20: 0.289,   # PLACEHOLDER (sqrt-zeta fit). Retraction gave ~0.43 (!) -- CHECK.
    0.25: 0.313,   # PLACEHOLDER (sqrt-zeta fit). No direct dense value yet.
    0.30: 0.333,   # PLACEHOLDER (sqrt-zeta fit). Retraction gave ~0.368.
}
# Flip to True ONLY after the centers above are replaced with measured
# dense crossings.  make_clone_phase2_grid() warns loudly while this is
# False.
_PHASE2_CENTERS_VERIFIED: bool = False
# ------------------------------------------------------------------------

ZETA_VALS_PHASE2: List[float] = [0.10, 0.15, 0.20, 0.25, 0.30]
L_PHASE2: List[int] = [160, 192, 256]

_PHASE2_LAMBDA_HALFWIDTH: float = 0.06   # +/- window around lambda_c^{(128)}
_PHASE2_N_LAMBDA: int = 13               # points across the window
_PHASE2_PER_L = len(ZETA_VALS_PHASE2) * _PHASE2_N_LAMBDA   # = 65

# Seed offset disjoint from v2 (<=1.3e9), FST/slope (~2.5e9), and dense
# ([7e9, 8.3e9]).  Phase-2 base seeds land in [9.08e9, ~10.3e9].
_PHASE2_SEED_OFFSET = 9_000_000_000


def nc_for_L_phase2(L: int) -> int:
    """N_c schedule for the Phase-2 supplement.

    Higher than the dense schedule because crossing precision at large L
    is the binding constraint:  delta-lambda_c <~ 0.02 needs B_L relative
    error ~5%, i.e. N_c ~ a few hundred even at L=256.
    """
    return {160: 400, 192: 300, 256: 250}[L]


def time_horizon_phase2(L: int, alpha: float) -> float:
    """Same time-horizon cap as v2/dense (saturation time is N_c- and
    L-driven, set by the same convergence tests)."""
    return time_horizon_v2(L, alpha)


def _seed_phase2(L: int, lam: float, zeta: float) -> int:
    return _seed(L, lam, zeta) + _PHASE2_SEED_OFFSET


def lambda_mesh_phase2(
    zeta: float,
    halfwidth: float = _PHASE2_LAMBDA_HALFWIDTH,
    n: int = _PHASE2_N_LAMBDA,
) -> np.ndarray:
    """13-point lambda mesh centered on lambda_c^{(128)}(zeta).

    Clipped to (0.01, 0.95) for safety; at the intended centers (0.22-0.33)
    with halfwidth 0.06 no clipping occurs.
    """
    if zeta not in _PHASE2_LAMBDA_C_128:
        raise KeyError(
            f"zeta={zeta} has no lambda_c^(128) center in _PHASE2_LAMBDA_C_128"
        )
    lc = _PHASE2_LAMBDA_C_128[zeta]
    lo = max(0.01, lc - halfwidth)
    hi = min(0.95, lc + halfwidth)
    return np.linspace(lo, hi, n)


def make_clone_phase2_grid() -> List[dict]:
    """Phase-2 supplement: 3 L x 5 zeta x 13 lambda = 195 tasks.

    Task ordering: L outer, zeta middle, lambda inner.
      L=160: tasks   0.. 64
      L=192: tasks  65..129
      L=256: tasks 130..194
    """
    if not _PHASE2_CENTERS_VERIFIED:
        import warnings
        warnings.warn(
            "Phase-2 lambda centers are PLACEHOLDERS (_PHASE2_CENTERS_VERIFIED"
            " is False). The narrow +/-0.06 lambda mesh may miss the true"
            " crossing. Update _PHASE2_LAMBDA_C_128 from the dense L<=128"
            " analysis and set _PHASE2_CENTERS_VERIFIED = True before"
            " submitting.",
            stacklevel=2,
        )

    grid: List[dict] = []
    task_id = 0
    for L in L_PHASE2:
        N_c = int(nc_for_L_phase2(L))
        for zeta in ZETA_VALS_PHASE2:
            for lam in lambda_mesh_phase2(zeta):
                alpha, w = _alpha_w_from_lam(float(lam))
                T = time_horizon_phase2(L, alpha)
                grid.append(dict(
                    task_id=task_id, L=int(L), lam=float(lam),
                    alpha=alpha, w=w, zeta=float(zeta),
                    T=float(T), N_c=N_c,
                    seed=_seed_phase2(L, float(lam), zeta),
                ))
                task_id += 1

    expected = len(L_PHASE2) * _PHASE2_PER_L
    assert len(grid) == expected, (
        f"Phase-2 grid size mismatch: {len(grid)} != {expected}"
    )
    return grid


def task_params_clone_phase2(task_id: int) -> dict:
    grid = make_clone_phase2_grid()
    if not (0 <= task_id < len(grid)):
        raise IndexError(
            f"Phase-2 task_id {task_id} out of range [0, {len(grid)})"
        )
    return grid[task_id]


def n_tasks_clone_phase2() -> int:
    return len(L_PHASE2) * _PHASE2_PER_L  # = 195


def clone_phase2_task_id_ranges() -> dict[int, tuple[int, int]]:
    """L -> (first_task_id, last_task_id) inclusive for the Phase-2 grid."""
    return {
        L: (i * _PHASE2_PER_L, (i + 1) * _PHASE2_PER_L - 1)
        for i, L in enumerate(L_PHASE2)
    }



# ======================================================================
# Clone "rescue" grid (added 2026-06-03) -- lean large-L resubmission
# ======================================================================
# Context: the dense large_L job (N_c=450 at L=96, full 24-26 pt lambda
# grid) was too heavy -- L=96 alone exceeded the 120h walltime (342/514
# done) and L=128 never started.  Two compounding causes, both measured
# from the partial data:
#   (1) compute scales as ~ N_c * T * L^4  (validated to L=96)
#   (2) cloning variance inflates with L: B_L rel-err grows 2.8% (L=32)
#       -> 13.5% (L=96) at comparable N_c, because ESS collapses near
#       criticality.  Beating it by N_c alone is hopeless (~5000 at L=128).
#
# This rescue grid gets L=128 (and optionally L=160) by:
#   - NARROW lambda window (13 pts, +/-0.07) centered on the MEASURED
#     dense L<=96 crossings (no need for the full grid -- lambda_c is known)
#   - reduced N_c (250 at L=128, 120 at L=160): ~20% B_L err, same ballpark
#     as the L=96 data we already extract usable crossings from
#   - decisive-but-broad zeta set (10 values, all present in the dense grid
#     so (64,128) crossings are computable against existing L=64 data)
#   - T kept at 100 (saturation is ballistic ~L/v; do NOT cut it at L=128)
#
# Centers below are MEASURED from clone_aggregate_dense_partial.pkl
# (mean of available (32,64)&(48,96) B_L crossings; (16,32)+drift for
# large zeta where L=96 didn't finish).
#
# Cost (validated model): L=128 N_c=250 ~13h/task; 130 tasks / 24 conc
# ~= 72h wall -> fits a 96h job.  L=160 N_c=120 ~16h/task; ~87h wall.
#
# Task layout (L outer, zeta middle, lambda inner): 10 zeta x 13 lambda
# = 130 per L.
#   L=128: tasks   0..129
#   L=160: tasks 130..259   (only if L_RESCUE includes 160)
# ======================================================================

# Measured lambda_c(zeta) from the dense L<=96 Binder crossings.
_RESCUE_LAMBDA_C: dict[float, float] = {
    0.10: 0.19,
    0.15: 0.22,
    0.22: 0.24,
    0.25: 0.25,
    0.30: 0.26,
    0.40: 0.32,
    0.50: 0.36,
    0.65: 0.43,
    0.80: 0.51,
    1.00: 0.53,
}
ZETA_VALS_RESCUE: List[float] = sorted(_RESCUE_LAMBDA_C.keys())

# L sizes for the rescue run.  Edit to [128] only, or [128, 160].
L_RESCUE: List[int] = [128, 160]

_RESCUE_LAMBDA_HALFWIDTH: float = 0.07   # +/- window; brackets cross-L drift
_RESCUE_N_LAMBDA: int = 13
_RESCUE_PER_L = len(ZETA_VALS_RESCUE) * _RESCUE_N_LAMBDA   # = 130

# Seed offset disjoint from v2(<=1.3e9), FST/slope(~2.5e9),
# dense([7e9,8.3e9]), phase2([9e9,11.5e9]).  Rescue lands in [12e9,13.3e9].
_RESCUE_SEED_OFFSET = 12_000_000_000


def nc_for_L_rescue(L: int) -> int:
    """Reduced N_c schedule for the rescue run.

    ~20% B_L rel-err expected (same ballpark as the L=96 dense data, from
    which usable crossings were extracted).  Going higher is better but
    each step in N_c costs linearly in wall-time; these values keep
    130 tasks inside ~72-96h at 24-way concurrency.
    """
    return {128: 250, 160: 120, 192: 80}.get(L, 150)


def _seed_rescue(L: int, lam: float, zeta: float) -> int:
    return _seed(L, lam, zeta) + _RESCUE_SEED_OFFSET


def lambda_mesh_rescue(zeta: float) -> np.ndarray:
    lc = _RESCUE_LAMBDA_C[zeta]
    lo = max(0.01, lc - _RESCUE_LAMBDA_HALFWIDTH)
    hi = min(0.95, lc + _RESCUE_LAMBDA_HALFWIDTH)
    return np.linspace(lo, hi, _RESCUE_N_LAMBDA)


def make_clone_rescue_grid() -> List[dict]:
    """Lean large-L grid: len(L_RESCUE) x 10 zeta x 13 lambda.

    Task ordering: L outer, zeta middle, lambda inner.
      L=128: tasks   0..129
      L=160: tasks 130..259  (if present in L_RESCUE)
    """
    grid: List[dict] = []
    task_id = 0
    for L in L_RESCUE:
        N_c = int(nc_for_L_rescue(L))
        for zeta in ZETA_VALS_RESCUE:
            for lam in lambda_mesh_rescue(zeta):
                alpha, w = _alpha_w_from_lam(float(lam))
                T = time_horizon_v2(L, alpha)
                grid.append(dict(
                    task_id=task_id, L=int(L), lam=float(lam),
                    alpha=alpha, w=w, zeta=float(zeta),
                    T=float(T), N_c=N_c,
                    seed=_seed_rescue(L, float(lam), zeta),
                ))
                task_id += 1
    expected = len(L_RESCUE) * _RESCUE_PER_L
    assert len(grid) == expected, f"Rescue grid size mismatch: {len(grid)} != {expected}"
    return grid


def task_params_clone_rescue(task_id: int) -> dict:
    grid = make_clone_rescue_grid()
    if not (0 <= task_id < len(grid)):
        raise IndexError(f"Rescue task_id {task_id} out of range [0, {len(grid)})")
    return grid[task_id]


def n_tasks_clone_rescue() -> int:
    return len(L_RESCUE) * _RESCUE_PER_L


def clone_rescue_task_id_ranges() -> dict[int, tuple[int, int]]:
    return {
        L: (i * _RESCUE_PER_L, (i + 1) * _RESCUE_PER_L - 1)
        for i, L in enumerate(L_RESCUE)
    }


# ======================================================================
# Clone N_c-ladder campaign (added 2026-06-05) -- the "2-day decisive" run
# ======================================================================
# Goal: a CLEAN lambda_c(zeta) in the discriminating small-zeta window by
# fixing the two things the dense/rescue data could not:
#   (1) finite-N_c bias on B_L (~45% at L=128, N_c=250) -> removed with a
#       1/N_c ladder {250,500,800} + extrapolation to N_c->inf;
#   (2) per-point statistical error -> beaten with seed BLOCKS (B independent
#       5-realisation tasks per point), NOT by N_c alone (hopeless near
#       criticality: ESS collapses, would need N_c~5000).
# Pairs against existing CLEAN dense L=32,64 (same zeta) -> (32,64,128) FSS.
# Records full observable set (CMI comps + Renyi-2/3 + corr) so the cleaner
# estimators (Renyi-2 crossing, bipartite MI) and the washout test reuse it.
#
# 2-DAY FEASIBILITY: a task runs N_REAL=5 realisations on 5 cores IN PARALLEL,
# so wall = ONE realisation: ~13h(Nc250)/~26h(Nc500)/~42h(Nc800) at L=128,
# all <48h.  Wall is set by the largest-N_c task, not by #points or #seeds:
# more seeds = more blocks = more concurrent 5-core tasks = more cores, same
# wall.  ~53k core-h at B=3 -> ~10-12 nodes for <2 days (or 17 nodes ~42h).
#
# Each N_c rung writes to its OWN output dir (the aggregator keys by
# (L,lam,zeta) and would otherwise merge rungs).  Blocks pool via
# analysis/aggregate_ladder.py.  Task order: rung outer, block, zeta, lam.
# ======================================================================

LADDER_L: int = 128

# zeta subset of the dense grid (so (32,64,128) crossings reuse existing
# clean L=32,64 at the SAME zeta) -- the discriminating window.
LADDER_ZETA: List[float] = [0.08, 0.10, 0.15, 0.18, 0.22, 0.25, 0.30]

# lambda_c(zeta) window centers: blend of measured dense (32,64) crossings and
# rescue centers; halfwidth widened to bracket the (64,128) drift. Exact
# center is not critical -- the window only has to contain the L=128 crossing.
_LADDER_LAMBDA_C: dict[float, float] = {
    0.08: 0.15, 0.10: 0.17, 0.15: 0.21, 0.18: 0.23,
    0.22: 0.25, 0.25: 0.26, 0.30: 0.27,
}
_LADDER_HALFWIDTH: float = 0.08
_LADDER_N_LAMBDA: int = 13          # full-grid lambda count (N_c=500 rung)
_LADDER_N_LAMBDA_CALIB: int = 3     # central lambda count (calib rungs)

_LADDER_NC_FULL: int = 500          # workhorse full-grid rung
_LADDER_NC_CALIB: List[int] = [250, 800]   # calibration rungs (central lam only)

_LADDER_B_FULL: int = 3             # blocks on full grid  -> 15 seeds
_LADDER_B_CALIB: int = 3            # blocks on calib subset -> 15 seeds

# Seeds land in [20e9, ~20.5e9], disjoint from v2(<=1.3e9), FST/slope(~2.5e9),
# dense([7e9,8.3e9]), phase2([9e9,11.5e9]), rescue([12e9,13.3e9]).
_LADDER_SEED_OFFSET: int = 20_000_000_000
_LADDER_RUNG_OFFSET: dict[int, int] = {500: 0, 250: 100_000_000, 800: 200_000_000}
_LADDER_BLOCK_STRIDE: int = 10_000_000


def _ladder_lambda_mesh(zeta: float, n: int) -> np.ndarray:
    lc = _LADDER_LAMBDA_C[zeta]
    lo = max(0.01, lc - _LADDER_HALFWIDTH)
    hi = min(0.95, lc + _LADDER_HALFWIDTH)
    full = np.linspace(lo, hi, _LADDER_N_LAMBDA)
    if n >= _LADDER_N_LAMBDA:
        return full
    i0 = (_LADDER_N_LAMBDA - n) // 2     # central n points (the crossing band)
    return full[i0:i0 + n]


def _seed_ladder(L: int, lam: float, zeta: float, N_c: int, block: int) -> int:
    return (_seed(L, lam, zeta) + _LADDER_SEED_OFFSET
            + _LADDER_RUNG_OFFSET[N_c] + block * _LADDER_BLOCK_STRIDE)


def _ladder_rungs() -> list[tuple[int, int, int]]:
    """(N_c, n_lambda, n_block) per rung; full rung first, then calib rungs."""
    rungs = [(_LADDER_NC_FULL, _LADDER_N_LAMBDA, _LADDER_B_FULL)]
    rungs += [(nc, _LADDER_N_LAMBDA_CALIB, _LADDER_B_CALIB) for nc in _LADDER_NC_CALIB]
    return rungs


def make_clone_ladder_grid() -> List[dict]:
    """N_c-ladder campaign grid at L=128.

    Task order (contiguous): rung outer, block, zeta, lambda.
    Rung order: full(N_c=500) then calib(250, 800).
    """
    grid: List[dict] = []
    task_id = 0
    for N_c, n_lam, n_block in _ladder_rungs():
        for block in range(n_block):
            for zeta in LADDER_ZETA:
                for lam in _ladder_lambda_mesh(zeta, n_lam):
                    alpha, w = _alpha_w_from_lam(float(lam))
                    T = time_horizon_v2(LADDER_L, alpha)   # T=100 cap at L=128
                    grid.append(dict(
                        task_id=task_id, L=int(LADDER_L), lam=float(lam),
                        alpha=alpha, w=w, zeta=float(zeta),
                        T=float(T), N_c=int(N_c), block=int(block),
                        seed=_seed_ladder(LADDER_L, float(lam), zeta, int(N_c), int(block)),
                    ))
                    task_id += 1
    return grid


def task_params_clone_ladder(task_id: int) -> dict:
    grid = make_clone_ladder_grid()
    if not (0 <= task_id < len(grid)):
        raise IndexError(f"Ladder task_id {task_id} out of range [0, {len(grid)})")
    return grid[task_id]


def n_tasks_clone_ladder() -> int:
    return len(make_clone_ladder_grid())


def clone_ladder_rung_ranges() -> dict[int, tuple[int, int]]:
    """N_c -> (first_task_id, last_task_id) inclusive. Used by the submit
    script to run each rung into its own output dir."""
    from collections import defaultdict
    by_nc: dict[int, list[int]] = defaultdict(list)
    for p in make_clone_ladder_grid():
        by_nc[p["N_c"]].append(p["task_id"])
    return {nc: (min(ids), max(ids)) for nc, ids in sorted(by_nc.items())}
