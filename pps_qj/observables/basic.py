from __future__ import annotations

import numpy as np


def entanglement_entropy_statevector(psi: np.ndarray, L: int, l_sub: int) -> float:
    if l_sub <= 0 or l_sub >= L:
        return 0.0
    dim_a = 2**l_sub
    dim_b = 2 ** (L - l_sub)
    x = psi.reshape((dim_a, dim_b))
    singular_values = np.linalg.svd(x, compute_uv=False)
    probabilities = np.clip((singular_values**2).real, 1e-15, 1.0)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def entanglement_entropy_gamma(Gamma: np.ndarray, l_sub: int) -> float:
    if l_sub <= 0:
        return 0.0
    n = 2 * l_sub
    submatrix = np.asarray(Gamma[:n, :n], dtype=np.float64)
    eigenvalues = np.linalg.eigvals(1j * submatrix)
    nus = np.sort(np.abs(np.real(eigenvalues)))[::2]
    x_plus = np.clip((1.0 + nus) * 0.5, 1e-15, 1.0)
    x_minus = np.clip((1.0 - nus) * 0.5, 1e-15, 1.0)
    return float(-np.sum(x_plus * np.log2(x_plus) + x_minus * np.log2(x_minus)))
