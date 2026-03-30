from __future__ import annotations

from typing import Callable

import numpy as np

from pps_qj.types import Tolerances


def safe_normalize(vec: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= tol:
        raise ValueError("Cannot normalize near-zero vector")
    return vec / norm


def safe_probs(weights: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    arr = np.clip(arr, 0.0, np.inf)
    total = float(arr.sum())
    if total <= tol:
        raise ValueError("Probability weights are near zero")
    return arr / total


def bracket_and_bisect(
    fn: Callable[[float], float],
    target: float,
    x0: float = 0.0,
    x1: float = 1.0,
    tol: Tolerances = Tolerances(),
    max_expand: int = 80,
    max_iter: int = 200,
) -> float:
    f0 = fn(x0) - target
    if abs(f0) <= tol.atol:
        return x0

    lo, hi = x0, x1
    fhi = fn(hi) - target
    expands = 0
    while fhi > 0.0 and expands < max_expand:
        lo = hi
        hi *= 2.0
        fhi = fn(hi) - target
        expands += 1

    if fhi > 0.0:
        raise RuntimeError("Failed to bracket root for waiting time")

    left = lo
    right = hi
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        fmid = fn(mid) - target
        if abs(fmid) <= tol.atol or abs(right - left) <= tol.rtol * max(1.0, abs(mid)):
            return mid
        if fmid > 0.0:
            left = mid
        else:
            right = mid

    return 0.5 * (left + right)
