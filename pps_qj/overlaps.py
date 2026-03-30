from __future__ import annotations

import numpy as np


def exact_operator_overlap(operator: np.ndarray, state: np.ndarray) -> float:
    """Return <psi|operator|psi> for a normalized pure state."""
    value = np.vdot(state, operator @ state)
    return float(np.real_if_close(value, tol=1_000.0).real)


def _sqrt_positive_determinant(matrix: np.ndarray, tol: float = 1e-10) -> float:
    sign, logabs = np.linalg.slogdet(np.asarray(matrix, dtype=np.complex128))
    value = sign * np.exp(logabs)
    value = np.real_if_close(value, tol=1_000.0)
    if np.iscomplexobj(value) and abs(value.imag) > tol:
        raise ValueError("Gaussian overlap determinant acquired a complex phase")
    real_value = float(np.real(value))
    if real_value < -tol:
        raise ValueError(f"Gaussian overlap determinant is negative: {real_value}")
    return float(np.sqrt(max(real_value, 0.0)))


def gaussian_overlap(
    operator_covariance: np.ndarray,
    state_covariance: np.ndarray,
    z_scalar: float = 1.0,
) -> float:
    """Evaluate Tr(G rho) for a positive Gaussian operator and a pure Gaussian state.

    `operator_covariance` is the covariance matrix of the normalized Gaussian
    operator G / Tr(G). `z_scalar` is defined by z = 2^{-L} Tr(G), so the full
    overlap is

        Tr(G rho) = z * sqrt(det(I - C Gamma)).
    """
    C = np.asarray(operator_covariance, dtype=np.float64)
    Gamma = np.asarray(state_covariance, dtype=np.float64)
    if C.shape != Gamma.shape:
        raise ValueError("operator and state covariances must have the same shape")
    det_sqrt = _sqrt_positive_determinant(np.eye(C.shape[0]) - C @ Gamma)
    return float(z_scalar * det_sqrt)


def gaussian_post_jump_overlap(
    operator_covariance: np.ndarray,
    state_covariance: np.ndarray,
    jump_pair: tuple[int, int],
    z_scalar: float = 1.0,
) -> tuple[float, np.ndarray, float]:
    """Return (q, Gamma_post, Tr(G P rho P))."""
    from pps_qj.gaussian_backend import apply_projective_jump

    q, gamma_post = apply_projective_jump(state_covariance, jump_pair)
    overlap = q * gaussian_overlap(operator_covariance, gamma_post, z_scalar=z_scalar)
    return q, gamma_post, overlap
