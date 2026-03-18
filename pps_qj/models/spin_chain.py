from __future__ import annotations

import numpy as np

from pps_qj.types import ModelSpec

I2 = np.eye(2, dtype=np.complex128)
SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
SP = np.array([[0, 1], [0, 0]], dtype=np.complex128)
SM = np.array([[0, 0], [1, 0]], dtype=np.complex128)


def kron_many(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def jw_c_operators(L: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    c_ops = []
    cd_ops = []
    for j in range(L):
        ops_c = []
        ops_cd = []
        for k in range(L):
            if k < j:
                ops_c.append(SZ)
                ops_cd.append(SZ)
            elif k == j:
                ops_c.append(SM)
                ops_cd.append(SP)
            else:
                ops_c.append(I2)
                ops_cd.append(I2)
        c_ops.append(kron_many(ops_c))
        cd_ops.append(kron_many(ops_cd))
    return c_ops, cd_ops


def product_state(L: int, pattern: str = "alternating") -> np.ndarray:
    up = np.array([1.0, 0.0], dtype=np.complex128)
    dn = np.array([0.0, 1.0], dtype=np.complex128)

    if pattern == "up":
        local = [up for _ in range(L)]
    elif pattern == "down":
        local = [dn for _ in range(L)]
    else:
        local = [up if (i % 2 == 0) else dn for i in range(L)]

    psi = local[0]
    for s in local[1:]:
        psi = np.kron(psi, s)
    return psi / np.linalg.norm(psi)


def spin_chain_model(L: int, w: float, gamma: float, initial_pattern: str = "alternating") -> ModelSpec:
    c_ops, cd_ops = jw_c_operators(L)
    dim = 2**L

    H = np.zeros((dim, dim), dtype=np.complex128)
    jump_ops: list[np.ndarray] = []

    # Open boundary conditions: L-1 bonds (eq 184)
    for j in range(L - 1):
        jp1 = j + 1
        H += w * (cd_ops[j] @ c_ops[jp1] + cd_ops[jp1] @ c_ops[j])

        # d_j = 1/2(c†_j + c_j + c†_{j+1} - c_{j+1})  (eq 186)
        # Jump operator L_j = sqrt(gamma) * d†_j d_j   (eq 187)
        d = 0.5 * (cd_ops[j] + c_ops[j] + cd_ops[jp1] - c_ops[jp1])
        n_d = d.conj().T @ d
        jump_ops.append(np.sqrt(gamma) * n_d)

    return ModelSpec(
        L=L,
        H=H,
        jump_ops=jump_ops,
        gamma=gamma,
        w=w,
        boundary="open",
        model_id="spin_chain_model_i",
        initial_state=product_state(L, initial_pattern),
    )
