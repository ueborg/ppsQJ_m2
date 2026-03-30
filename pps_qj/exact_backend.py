from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply

from pps_qj.core.numerics import bracket_and_bisect, safe_probs, safe_normalize
from pps_qj.types import JumpTrajectory, Tolerances

I2 = sp.csr_matrix(np.eye(2, dtype=np.complex128))
SZ = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))
SP = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=np.complex128))
SM = sp.csr_matrix(np.array([[0, 0], [1, 0]], dtype=np.complex128))


def _kron_many_sparse(operators: list[sp.spmatrix]) -> sp.csr_matrix:
    out = operators[0]
    for op in operators[1:]:
        out = sp.kron(out, op, format="csr")
    return out.tocsr()


def build_jordan_wigner_operators(L: int) -> tuple[tuple[sp.csr_matrix, ...], tuple[sp.csr_matrix, ...]]:
    annihilation_ops: list[sp.csr_matrix] = []
    creation_ops: list[sp.csr_matrix] = []
    for site in range(L):
        ops_c: list[sp.spmatrix] = []
        ops_cd: list[sp.spmatrix] = []
        for idx in range(L):
            if idx < site:
                ops_c.append(SZ)
                ops_cd.append(SZ)
            elif idx == site:
                ops_c.append(SM)
                ops_cd.append(SP)
            else:
                ops_c.append(I2)
                ops_cd.append(I2)
        annihilation_ops.append(_kron_many_sparse(ops_c))
        creation_ops.append(_kron_many_sparse(ops_cd))
    return tuple(annihilation_ops), tuple(creation_ops)


def neel_state(L: int) -> np.ndarray:
    up = np.array([1.0, 0.0], dtype=np.complex128)
    down = np.array([0.0, 1.0], dtype=np.complex128)
    factors = [up if site % 2 == 0 else down for site in range(L)]
    psi = factors[0]
    for factor in factors[1:]:
        psi = np.kron(psi, factor)
    return safe_normalize(psi)


@dataclass(frozen=True)
class ExactSpinChainModel:
    L: int
    w: float
    gamma_m: float
    c_ops: tuple[sp.csr_matrix, ...]
    cd_ops: tuple[sp.csr_matrix, ...]
    hamiltonian: sp.csr_matrix
    jump_projectors: tuple[sp.csr_matrix, ...]
    h_effective: sp.csr_matrix
    initial_state: np.ndarray

    @property
    def dim(self) -> int:
        return 2**self.L


def build_exact_spin_chain_model(L: int, w: float, gamma_m: float) -> ExactSpinChainModel:
    c_ops, cd_ops = build_jordan_wigner_operators(L)
    dim = 2**L
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)
    projectors: list[sp.csr_matrix] = []

    for bond in range(L - 1):
        left = bond
        right = bond + 1
        H = H + w * (cd_ops[left] @ c_ops[right] + cd_ops[right] @ c_ops[left])
        d_op = 0.5 * (cd_ops[left] + c_ops[left] + cd_ops[right] - c_ops[right])
        P = (d_op.getH() @ d_op).tocsr()
        projectors.append(P)

    jump_sum = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for projector in projectors:
        jump_sum = jump_sum + projector
    h_eff = (H - 0.5j * gamma_m * jump_sum).tocsr()

    return ExactSpinChainModel(
        L=L,
        w=w,
        gamma_m=gamma_m,
        c_ops=c_ops,
        cd_ops=cd_ops,
        hamiltonian=H.tocsr(),
        jump_projectors=tuple(projectors),
        h_effective=h_eff,
        initial_state=neel_state(L),
    )


def exact_model_consistency(model: ExactSpinChainModel) -> dict[str, float]:
    H = model.hamiltonian.toarray()
    projectors = [P.toarray() for P in model.jump_projectors]
    jump_sum = sum(projectors)
    eigenvalues = np.linalg.eigvalsh(jump_sum)
    projector_errors = [
        float(np.linalg.norm(P @ P - P) + np.linalg.norm(P - P.conj().T))
        for P in projectors
    ]
    commutator_norms = []
    for j, Pj in enumerate(projectors):
        for k, Pk in enumerate(projectors):
            if j >= k:
                continue
            commutator_norms.append(float(np.linalg.norm(Pj @ Pk - Pk @ Pj)))
    return {
        "hamiltonian_hermiticity_error": float(np.linalg.norm(H - H.conj().T)),
        "max_projector_error": max(projector_errors, default=0.0),
        "min_jump_sum_eig": float(np.min(eigenvalues).real) if eigenvalues.size else 0.0,
        "max_jump_sum_eig": float(np.max(eigenvalues).real) if eigenvalues.size else 0.0,
        "max_projector_commutator": max(commutator_norms, default=0.0),
        "initial_state_norm_error": abs(np.linalg.norm(model.initial_state) - 1.0),
    }


def _propagate_unnormalized(model: ExactSpinChainModel, state: np.ndarray, dt: float) -> np.ndarray:
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if dt == 0.0:
        return np.asarray(state, dtype=np.complex128).copy()
    return np.asarray(expm_multiply((-1j * model.h_effective) * dt, state), dtype=np.complex128)


def _survival_probability(model: ExactSpinChainModel, state: np.ndarray, dt: float) -> float:
    psi_tilde = _propagate_unnormalized(model, state, dt)
    return float(np.real(np.vdot(psi_tilde, psi_tilde)))


def _sample_waiting_time(
    model: ExactSpinChainModel,
    state: np.ndarray,
    rng: np.random.Generator,
    tol: Tolerances | None = None,
) -> float:
    tolerances = tol or Tolerances()
    target = float(rng.uniform(0.0, 1.0))

    if len(model.jump_projectors) == 1 and model.hamiltonian.nnz == 0:
        q = float(np.real(np.vdot(state, model.jump_projectors[0] @ state)))
        q = float(np.clip(q, 0.0, 1.0))
        s_inf = 1.0 - q
        if q <= 1e-15 or target < s_inf + 1e-15:
            return float("inf")
        argument = (target - s_inf) / q
        argument = float(np.clip(argument, 1e-15, 1.0))
        return float(-(1.0 / model.gamma_m) * np.log(argument))

    survival = lambda dt: _survival_probability(model, state, dt)
    try:
        return float(
            bracket_and_bisect(
                fn=survival,
                target=target,
                x0=0.0,
                x1=1.0,
                tol=tolerances,
            )
        )
    except RuntimeError:
        return float("inf")


def _channel_probabilities(model: ExactSpinChainModel, state: np.ndarray) -> np.ndarray:
    weights = np.array(
        [np.real(np.vdot(state, projector @ state)) for projector in model.jump_projectors],
        dtype=np.float64,
    )
    return safe_probs(weights)


def _sample_channel(model: ExactSpinChainModel, state: np.ndarray, rng: np.random.Generator) -> int:
    probs = _channel_probabilities(model, state)
    return int(rng.choice(len(probs), p=probs))


def ordinary_quantum_jump_trajectory(
    model: ExactSpinChainModel,
    T: float,
    rng: np.random.Generator,
    tol: Tolerances | None = None,
) -> JumpTrajectory:
    state = np.asarray(model.initial_state, dtype=np.complex128).copy()
    t = 0.0
    jump_times: list[float] = []
    channels: list[int] = []

    while t < T:
        dt = _sample_waiting_time(model, state, rng, tol=tol)
        if not np.isfinite(dt) or t + dt >= T:
            state = safe_normalize(_propagate_unnormalized(model, state, T - t))
            t = T
            break

        psi_tilde = _propagate_unnormalized(model, state, dt)
        state = safe_normalize(psi_tilde)
        t += dt
        channel = _sample_channel(model, state, rng)
        state = safe_normalize(model.jump_projectors[channel] @ state)
        jump_times.append(t)
        channels.append(channel)

    return JumpTrajectory(
        jump_times=jump_times,
        channels=channels,
        final_time=t,
        final_state=state,
        candidate_jump_times=jump_times.copy(),
    )


def procedure_a_trajectory(
    model: ExactSpinChainModel,
    T: float,
    zeta: float,
    rng: np.random.Generator,
    tol: Tolerances | None = None,
) -> JumpTrajectory:
    while True:
        trajectory = ordinary_quantum_jump_trajectory(model, T, rng, tol=tol)
        if trajectory.n_jumps == 0:
            trajectory.accepted = True
            trajectory.diagnostics["acceptance_probability"] = 1.0
            return trajectory

        acceptance_probability = float(zeta ** trajectory.n_jumps)
        if float(rng.uniform(0.0, 1.0)) <= acceptance_probability:
            trajectory.accepted = True
            trajectory.diagnostics["acceptance_probability"] = acceptance_probability
            return trajectory


def procedure_b_trajectory(
    model: ExactSpinChainModel,
    T: float,
    zeta: float,
    rng: np.random.Generator,
    tol: Tolerances | None = None,
) -> JumpTrajectory:
    attempts = 0
    while True:
        attempts += 1
        state = np.asarray(model.initial_state, dtype=np.complex128).copy()
        t = 0.0
        jump_times: list[float] = []
        channels: list[int] = []
        candidate_jump_times: list[float] = []
        accepted = True

        while t < T:
            dt = _sample_waiting_time(model, state, rng, tol=tol)
            if not np.isfinite(dt) or t + dt >= T:
                state = safe_normalize(_propagate_unnormalized(model, state, T - t))
                t = T
                break

            psi_tilde = _propagate_unnormalized(model, state, dt)
            state = safe_normalize(psi_tilde)
            t += dt
            candidate_jump_times.append(t)
            if float(rng.uniform(0.0, 1.0)) > zeta:
                accepted = False
                break

            channel = _sample_channel(model, state, rng)
            state = safe_normalize(model.jump_projectors[channel] @ state)
            jump_times.append(t)
            channels.append(channel)

        if accepted and t >= T:
            return JumpTrajectory(
                jump_times=jump_times,
                channels=channels,
                final_time=t,
                final_state=state,
                accepted=True,
                candidate_jump_times=candidate_jump_times,
                diagnostics={"attempts": attempts},
            )


def procedure_c_local_trajectory(
    model: ExactSpinChainModel,
    T: float,
    zeta: float,
    rng: np.random.Generator,
    tol: Tolerances | None = None,
) -> JumpTrajectory:
    state = np.asarray(model.initial_state, dtype=np.complex128).copy()
    t = 0.0
    jump_times: list[float] = []
    channels: list[int] = []
    candidate_jump_times: list[float] = []

    while t < T:
        dt = _sample_waiting_time(model, state, rng, tol=tol)
        if not np.isfinite(dt) or t + dt >= T:
            state = safe_normalize(_propagate_unnormalized(model, state, T - t))
            t = T
            break

        psi_tilde = _propagate_unnormalized(model, state, dt)
        state = safe_normalize(psi_tilde)
        t += dt
        candidate_jump_times.append(t)
        if float(rng.uniform(0.0, 1.0)) > zeta:
            continue

        channel = _sample_channel(model, state, rng)
        state = safe_normalize(model.jump_projectors[channel] @ state)
        jump_times.append(t)
        channels.append(channel)

    return JumpTrajectory(
        jump_times=jump_times,
        channels=channels,
        final_time=t,
        final_state=state,
        candidate_jump_times=candidate_jump_times,
    )


def lindbladian_superoperator(
    model: ExactSpinChainModel,
    zeta: float = 1.0,
    adjoint: bool = False,
) -> sp.csr_matrix:
    dim = model.dim
    identity = sp.identity(dim, dtype=np.complex128, format="csr")
    H = model.hamiltonian.tocsr()
    if adjoint:
        superop = 1j * (sp.kron(identity, H, format="csr") - sp.kron(H.T, identity, format="csr"))
    else:
        superop = -1j * (sp.kron(identity, H, format="csr") - sp.kron(H.T, identity, format="csr"))

    for projector in model.jump_projectors:
        P = projector.tocsr()
        superop = superop + model.gamma_m * (
            zeta * sp.kron(P.T, P, format="csr")
            - 0.5 * sp.kron(identity, P, format="csr")
            - 0.5 * sp.kron(P.T, identity, format="csr")
        )
    return superop.tocsr()


def integrate_lindblad(
    model: ExactSpinChainModel,
    T: float,
    t_eval: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dim = model.dim
    rho0 = np.outer(model.initial_state, model.initial_state.conj())
    y0 = np.asarray(rho0.reshape(dim * dim, order="F"), dtype=np.complex128)
    superop = lindbladian_superoperator(model, zeta=1.0, adjoint=False)

    if t_eval is None:
        t_eval = np.linspace(0.0, T, 101)

    solution = solve_ivp(
        fun=lambda _t, y: superop @ y,
        t_span=(0.0, T),
        y0=y0,
        t_eval=np.asarray(t_eval, dtype=np.float64),
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Lindblad integration failed: {solution.message}")

    rhos = np.empty((solution.t.size, dim, dim), dtype=np.complex128)
    for idx, flat in enumerate(solution.y.T):
        rhos[idx] = flat.reshape((dim, dim), order="F")
    return solution.t, rhos
