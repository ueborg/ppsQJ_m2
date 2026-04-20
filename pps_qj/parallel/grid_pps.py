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
L_CLONE: List[int] = [8, 12, 16]
ZETA_VALS_CLONE: List[float] = [0.3, 0.5, 0.7, 1.0]


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
    if L == 8:
        return 2000
    if L == 12:
        return 1000
    if L == 16:
        return 500
    raise ValueError(f"N_c not defined for L={L}")


def time_horizon(L: int, alpha: float) -> float:
    """T(L, alpha) = max(30, 2L, 5/alpha).

    Exception: at L=256 we drop the 2L contribution to L to keep wall time
    tractable (see the prompt's note). Documented here for provenance.
    """
    base = max(30.0, 5.0 / max(alpha, 1e-9))
    if L >= 256:
        propagation = float(L)          # reduced from 2*L for the largest L
    else:
        propagation = 2.0 * L
    return float(max(base, propagation))


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
        print(f"n_tasks_clone = {n_tasks_clone()} (expected 204)")
        print(f"first: {g[0]}")
        print(f"last : {g[-1]}")
    else:
        print(f"unknown grid type: {which}")
        sys.exit(1)
