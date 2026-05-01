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
L_CLONE: List[int] = [8, 16, 32, 64]

ZETA_VALS_CLONE: List[float] = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.75, 0.80, 0.85,
    0.90, 0.92, 0.95, 0.97, 1.00,
]  # 15 points; denser near ζ=1 where the transition lives for most λ


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
    return {8: 2000, 16: 1000, 32: 500, 64: 200}[L]


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
    else:
        print(f"unknown grid type: {which}")
        sys.exit(1)
