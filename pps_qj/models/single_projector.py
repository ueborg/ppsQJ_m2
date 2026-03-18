from __future__ import annotations

import numpy as np

from pps_qj.types import ModelSpec


def single_projector_model(
    gamma: float = 1.0,
    hamiltonian: np.ndarray | None = None,
    initial_state: np.ndarray | None = None,
    dim: int = 2,
    projector_index: int = 1,
) -> ModelSpec:
    if hamiltonian is None:
        hamiltonian = np.zeros((dim, dim), dtype=np.complex128)

    P = np.zeros((dim, dim), dtype=np.complex128)
    P[projector_index, projector_index] = 1.0
    L = np.sqrt(gamma) * P

    if initial_state is None:
        initial_state = np.zeros(dim, dtype=np.complex128)
        initial_state[0] = 1.0

    return ModelSpec(
        L=1,
        H=hamiltonian,
        jump_ops=[L],
        gamma=gamma,
        model_id="single_projector",
        initial_state=np.asarray(initial_state, dtype=np.complex128),
    )
