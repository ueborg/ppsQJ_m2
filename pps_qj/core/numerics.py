from __future__ import annotations

from typing import Callable

import numpy as np

from pps_qj.types import Tolerances


def heff_from(H: np.ndarray, jump_ops: list[np.ndarray]) -> np.ndarray:
    if H is None:
        raise ValueError("H must be provided for exact backend")
    acc = np.zeros_like(H, dtype=np.complex128)
    for op in jump_ops:
        acc += op.conj().T @ op
    return np.asarray(H, dtype=np.complex128) - 0.5j * acc


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


def matrix_exponential(A: np.ndarray, t: float) -> np.ndarray:
    # Eigendecomposition-based expm. Assumes A is diagonalizable.
    vals, vecs = np.linalg.eig(A)
    inv = np.linalg.inv(vecs)
    exp_diag = np.diag(np.exp(vals * t))
    return vecs @ exp_diag @ inv


def bracket_and_bisect(
    fn: Callable[[float], float],
    target: float,
    x0: float = 0.0,
    x1: float = 1.0,
    tol: Tolerances = Tolerances(),
    max_expand: int = 80,
    max_iter: int = 200,
) -> float:
    # Solve fn(x)=target for monotone decreasing fn.
    f0 = fn(x0) - target
    if abs(f0) <= tol.atol:
        return x0

    lo, hi = x0, x1
    flo = f0
    fhi = fn(hi) - target

    expands = 0
    while fhi > 0.0 and expands < max_expand:
        lo = hi
        flo = fhi
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
