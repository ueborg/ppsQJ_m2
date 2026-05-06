from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq


def bond_jump_pair(bond: int) -> tuple[int, int]:
    """Return the 0-indexed Majorana pair for bond `bond`."""
    return 2 * bond, 2 * bond + 3


def neel_covariance(L: int) -> np.ndarray:
    """Majorana covariance of the |up,down,up,...> product state."""
    gamma = np.zeros((2 * L, 2 * L), dtype=np.float64)
    for site in range(L):
        a = 2 * site
        b = a + 1
        sign = 1.0 if site % 2 == 0 else -1.0
        gamma[a, b] = sign
        gamma[b, a] = -sign
    return gamma


def covariance_from_orbitals(orbitals: np.ndarray) -> np.ndarray:
    n = orbitals.shape[0]
    gamma = 1j * (2.0 * (orbitals @ orbitals.conj().T) - np.eye(n))
    gamma = np.real_if_close(gamma, tol=1_000.0)
    gamma = np.asarray(gamma.real, dtype=np.float64)
    return 0.5 * (gamma - gamma.T)


def orbitals_from_covariance(covariance: np.ndarray) -> np.ndarray:
    Gamma = np.asarray(covariance, dtype=np.float64)
    n = Gamma.shape[0]
    L = n // 2
    values, vectors = np.linalg.eigh(1j * Gamma)
    order = np.argsort(values.real)
    orbitals = vectors[:, order[:L]]
    orbitals, _ = np.linalg.qr(orbitals, mode="reduced")
    return np.asarray(orbitals, dtype=np.complex128)


def project_to_physical_covariance(covariance: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    C = 0.5 * (np.asarray(covariance, dtype=np.float64) - np.asarray(covariance, dtype=np.float64).T)
    spectrum, vectors = np.linalg.eigh(1j * C)
    clipped = np.clip(spectrum.real, -1.0 + eps, 1.0 - eps)
    hermitian = vectors @ np.diag(clipped) @ vectors.conj().T
    projected = np.real_if_close(-1j * hermitian, tol=1_000.0).real
    return 0.5 * (projected - projected.T)


def majorana_hamiltonian_generator(L: int, w: float) -> np.ndarray:
    h = np.zeros((2 * L, 2 * L), dtype=np.float64)
    for bond in range(L - 1):
        a, b = bond_jump_pair(bond)
        c = 2 * bond + 1
        d = 2 * bond + 2
        h[a, b] = w
        h[b, a] = -w
        h[c, d] = -w
        h[d, c] = w
    return h


def effective_generator(L: int, w: float, alpha: float) -> np.ndarray:
    h_eff = np.asarray(majorana_hamiltonian_generator(L, w), dtype=np.complex128)
    for bond in range(L - 1):
        a, b = bond_jump_pair(bond)
        h_eff[a, b] -= 1j * alpha
        h_eff[b, a] += 1j * alpha
    return h_eff


@dataclass(frozen=True)
class GaussianChainModel:
    L: int
    w: float
    # alpha: measurement rate in Kells et al. (2023), Eq. (1); jump rate = 2*alpha
    alpha: float
    h_hamiltonian: np.ndarray
    h_effective: np.ndarray
    jump_pairs: tuple[tuple[int, int], ...]
    gamma0: np.ndarray
    orbitals0: np.ndarray
    # Cached eigendecomposition of h_effective — computed once at construction
    # and reused by every gaussian_born_rule_trajectory call, eliminating the
    # dominant per-call overhead (np.linalg.eig on 2L×2L matrix, O(L^3)).
    # V @ diag(h_eff_evals) @ V_inv == h_effective (to machine precision).
    h_eff_evals: np.ndarray
    h_eff_V:     np.ndarray
    h_eff_V_inv: np.ndarray
    h_eff_VhV:   np.ndarray  # V†V, needed for Gram-matrix branch-norm


def build_gaussian_chain_model(L: int, w: float, alpha: float) -> GaussianChainModel:
    gamma0  = neel_covariance(L)
    h_eff   = effective_generator(L, w, alpha)
    h_eff_c = np.asarray(h_eff, dtype=np.complex128)
    evals, V = np.linalg.eig(h_eff_c)
    V_inv    = np.linalg.inv(V)
    VhV      = V.conj().T @ V
    return GaussianChainModel(
        L=L,
        w=w,
        alpha=alpha,
        h_hamiltonian=majorana_hamiltonian_generator(L, w),
        h_effective=h_eff,
        jump_pairs=tuple(bond_jump_pair(bond) for bond in range(L - 1)),
        gamma0=gamma0,
        orbitals0=orbitals_from_covariance(gamma0),
        h_eff_evals=evals,
        h_eff_V=V,
        h_eff_V_inv=V_inv,
        h_eff_VhV=VhV,
    )


def jump_probability(covariance: np.ndarray, jump_pair: tuple[int, int]) -> float:
    a, b = jump_pair
    sigma = float(np.asarray(covariance, dtype=np.float64)[a, b])
    return float(np.clip(0.5 * (1.0 - sigma), 0.0, 1.0))


@dataclass(frozen=True)
class GaussianNoClickEvolution:
    orbitals_unnormalized: np.ndarray
    orbitals_normalized: np.ndarray
    covariance: np.ndarray
    branch_norm: float


def propagate_no_click_orbitals(
    orbitals: np.ndarray,
    h_effective: np.ndarray,
    dt: float,
    *,
    alpha: float = 0.0,
    n_monitored: int = 0,
) -> GaussianNoClickEvolution:
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if dt == 0.0:
        gamma = covariance_from_orbitals(orbitals)
        return GaussianNoClickEvolution(
            orbitals_unnormalized=np.asarray(orbitals, dtype=np.complex128).copy(),
            orbitals_normalized=np.asarray(orbitals, dtype=np.complex128).copy(),
            covariance=gamma,
            branch_norm=1.0,
        )

    M = expm(np.asarray(h_effective, dtype=np.complex128) * dt)
    orbitals_tilde = M @ np.asarray(orbitals, dtype=np.complex128)
    q_matrix, r_matrix = np.linalg.qr(orbitals_tilde, mode="reduced")
    diag_abs = np.abs(np.diag(r_matrix))
    if np.any(diag_abs <= 1e-300):
        branch_norm = 0.0
    else:
        log_abs_det_r = float(np.sum(np.log(diag_abs)))
        branch_norm = float(np.exp(log_abs_det_r - alpha * n_monitored * dt))
    gamma = covariance_from_orbitals(q_matrix)
    return GaussianNoClickEvolution(
        orbitals_unnormalized=orbitals_tilde,
        orbitals_normalized=q_matrix,
        covariance=gamma,
        branch_norm=branch_norm,
    )


def apply_projective_jump(
    covariance: np.ndarray,
    jump_pair: tuple[int, int],
) -> tuple[float, np.ndarray]:
    Gamma = np.asarray(covariance, dtype=np.float64)
    a, b = jump_pair
    sigma = float(Gamma[a, b])
    denom = 1.0 - sigma
    q = 0.5 * denom
    if denom <= 1e-14:
        raise ValueError("Jump probability is numerically zero")

    u = Gamma[:, a].copy()
    v = Gamma[:, b].copy()
    keep_mask = np.ones(Gamma.shape[0], dtype=bool)
    keep_mask[a] = False
    keep_mask[b] = False
    keep = np.flatnonzero(keep_mask)

    gamma_new = Gamma.copy()
    gamma_new[np.ix_(keep, keep)] += (
        np.outer(u[keep], v[keep]) - np.outer(v[keep], u[keep])
    ) / denom
    gamma_new[a, :] = 0.0
    gamma_new[:, a] = 0.0
    gamma_new[b, :] = 0.0
    gamma_new[:, b] = 0.0
    gamma_new[a, b] = -1.0
    gamma_new[b, a] = 1.0
    gamma_new = 0.5 * (gamma_new - gamma_new.T)
    return float(np.clip(q, 0.0, 1.0)), gamma_new


def entanglement_entropy(covariance: np.ndarray, ell: int, base: float = 2.0) -> float:
    Gamma = np.asarray(covariance, dtype=np.float64)
    if ell < 0 or 2 * ell > Gamma.shape[0]:
        raise ValueError("ell must define a valid subsystem")
    if ell == 0:
        return 0.0
    sub = Gamma[: 2 * ell, : 2 * ell]
    eigenvalues = np.linalg.eigvals(sub)
    nus = np.sort(np.abs(np.imag(eigenvalues)))[::2]
    entropy = 0.0
    log_fn = np.log2 if base == 2.0 else lambda x: np.log(x) / np.log(base)
    for nu in nus:
        nu = float(np.clip(nu, 0.0, 1.0))
        p_plus = 0.5 * (1.0 + nu)
        p_minus = 0.5 * (1.0 - nu)
        term = 0.0
        if p_plus > 1e-15:
            term -= p_plus * log_fn(p_plus)
        if p_minus > 1e-15:
            term -= p_minus * log_fn(p_minus)
        entropy += term
    return float(entropy)


@dataclass(frozen=True)
class GaussianTrajectoryResult:
    """Result of a Born-rule quantum-jump trajectory in the Gaussian backend."""
    final_covariance: np.ndarray
    final_orbitals: np.ndarray   # consistent with final_covariance; avoids
                                 # re-calling orbitals_from_covariance in cloning
    n_jumps: int
    jump_times: list[float]
    jump_channels: list[int]


def gaussian_born_rule_trajectory(
    model: GaussianChainModel,
    T: float,
    rng: np.random.Generator,
    bisection_tol: float = 1e-6,
    *,
    gamma0_override: Optional[np.ndarray] = None,
    orbitals0_override: Optional[np.ndarray] = None,
    ja_cached: Optional[np.ndarray] = None,
    jb_cached: Optional[np.ndarray] = None,
    initial_uniform_override: Optional[float] = None,
) -> GaussianTrajectoryResult:
    """Exact Born-rule quantum-jump trajectory.

    Waiting times are sampled by inverting the survival probability
    ``branch_norm(dt) = U`` using Brent's method (brentq), which converges
    in ~5–8 function evaluations vs ~40 for pure bisection.

    Per-jump no-click propagation uses the cached eigendecomposition of
    ``h_effective`` (two matrix multiplies) rather than scipy's Padé expm,
    and reuses ``coeffs = V⁻¹ @ orbitals`` already computed for the branch-norm
    evaluation, saving one extra matrix multiply per jump.

    Optional override parameters
    ----------------------------
    gamma0_override, orbitals0_override
        If provided, the trajectory starts from this (covariance, orbitals)
        pair instead of model.gamma0/orbitals0.  This avoids creating a new
        frozen GaussianChainModel via ``dataclasses.replace`` per-clone in
        the cloning loop.
    ja_cached, jb_cached
        Precomputed jump-pair index arrays.  If not supplied, they are built
        on each call.  In cloning, these depend only on jump_pairs, so they
        are computed once outside the step loop and passed in.
    """
    n_monitored = len(model.jump_pairs)

    # Read cached eigendecomposition — computed once at model construction,
    # shared across all N_c clones and all resampling steps. This eliminates
    # the dominant per-call overhead (eig on 2L×2L, ~0.6ms at L=16, ~3.6ms
    # at L=32) which previously accounted for 40–60% of cloning wall time.
    evals = model.h_eff_evals
    V     = model.h_eff_V
    V_inv = model.h_eff_V_inv
    VhV   = model.h_eff_VhV

    # Precompute jump-pair index arrays for vectorised probability extraction
    # — fall back to building them if the caller didn't supply cached versions.
    if ja_cached is None or jb_cached is None:
        _jp = model.jump_pairs
        _ja = np.array([p[0] for p in _jp], dtype=np.intp)
        _jb = np.array([p[1] for p in _jp], dtype=np.intp)
    else:
        _ja = ja_cached
        _jb = jb_cached

    # Use override state if provided (avoids replace(model, ...) overhead)
    if orbitals0_override is None:
        orbitals = np.asarray(model.orbitals0, dtype=np.complex128).copy()
    else:
        orbitals = np.asarray(orbitals0_override, dtype=np.complex128).copy()
    if gamma0_override is None:
        cov = np.asarray(model.gamma0, dtype=np.float64).copy()
    else:
        cov = np.asarray(gamma0_override, dtype=np.float64).copy()
    t = 0.0
    jump_times: list[float] = []
    jump_channels: list[int] = []
    _first_iter_override = initial_uniform_override is not None

    while t < T:
        if _first_iter_override:
            U = float(initial_uniform_override)
            _first_iter_override = False
        else:
            U = float(rng.uniform(0.0, 1.0))
        T_rem = T - t

        # V⁻¹ @ orbitals — reused both in branch-norm and in propagation.
        coeffs = V_inv @ orbitals  # (2L, L)

        def _fast_branch_norm(dt: float) -> float:
            """Survival probability via L×L Gram-matrix Cholesky logdet.

            Gram = A^† (V^†V) A is PSD Hermitian by construction (= ||V^{1/2}A||²),
            so np.linalg.cholesky is ~2x faster than np.linalg.slogdet on PSD
            matrices (cholesky skips the LU pivoting that slogdet needs for
            general matrices).  Falls back to slogdet on LinAlgError, which
            happens only when the Gram is numerically singular — i.e. when the
            survival probability is exponentially small anyway and the caller
            will route the clone into the jumping branch regardless.
            """
            if dt <= 0.0:
                return 1.0
            exp_d = np.exp(evals * dt)
            A = exp_d[:, None] * coeffs
            gram = A.conj().T @ (VhV @ A)
            try:
                L_chol = np.linalg.cholesky(gram)
                # log sqrt(det(gram)) = sum log |diag(L_chol)|  for L lower-triangular
                log_half_logdet = float(np.sum(np.log(np.abs(np.diag(L_chol)))))
                return float(np.exp(log_half_logdet - model.alpha * n_monitored * dt))
            except np.linalg.LinAlgError:
                sign, logdet = np.linalg.slogdet(gram)
                if sign <= 0 or not np.isfinite(logdet):
                    return 0.0
                return float(np.exp(0.5 * logdet - model.alpha * n_monitored * dt))

        bn_end = _fast_branch_norm(T_rem)
        if bn_end >= U:
            # No jump in remaining time — propagate to T using cached eig.
            exp_d = np.exp(evals * T_rem)
            orbs_tilde = V @ (exp_d[:, None] * coeffs)
            q_mat, _ = np.linalg.qr(orbs_tilde, mode="reduced")
            orbitals = q_mat   # keep consistent with cov below
            cov = covariance_from_orbitals(q_mat)
            break

        # Find jump time via Brent's method: ~5–8 evals vs ~40 for bisection.
        try:
            dt_star = brentq(
                lambda dt: _fast_branch_norm(dt) - U,
                0.0, T_rem,
                xtol=bisection_tol,
                maxiter=50,
                full_output=False,
            )
        except ValueError:
            # Bracketing failure (numerical edge case) — fall back to midpoint.
            dt_star = 0.5 * T_rem

        # Propagate to jump time using cached eigendecomposition.
        # Reuses coeffs already computed above — no extra V_inv multiply.
        exp_d_star = np.exp(evals * dt_star)
        orbs_tilde = V @ (exp_d_star[:, None] * coeffs)
        q_mat, r_mat = np.linalg.qr(orbs_tilde, mode="reduced")
        orbitals = q_mat
        cov = covariance_from_orbitals(orbitals)
        t += dt_star

        # Select jump channel proportional to q_j (vectorised over all bonds).
        probs = np.clip(0.5 * (1.0 - cov[_ja, _jb]), 0.0, 1.0)
        total = probs.sum()
        if total < 1e-15:
            break
        probs /= total
        channel = int(rng.choice(n_monitored, p=probs))

        # Apply jump and recover orthonormal orbitals.
        _, cov = apply_projective_jump(cov, model.jump_pairs[channel])
        orbitals = orbitals_from_covariance(cov)

        jump_times.append(t)
        jump_channels.append(channel)

    return GaussianTrajectoryResult(
        final_covariance=cov,
        final_orbitals=orbitals,
        n_jumps=len(jump_times),
        jump_times=jump_times,
        jump_channels=jump_channels,
    )


def topological_entanglement_entropy(covariance: np.ndarray) -> float:
    """Topological entanglement entropy from the ABDC partition.

    Accepts a 2L x 2L real antisymmetric Majorana covariance matrix and returns
    S_top = S_AB + S_BC - S_B - S_D, where the ABDC partition divides L sites
    into four equal quarters: A=[0,L/4), B=[L/4,L/2), D=[3L/4,L), C=[L/2,3L/4).

    In the large-alpha, small-w limit (topological phase) the steady-state
    average approaches 1; in the small-alpha, large-w limit (critical phase)
    it remains O(1) but the product B_L = S_top * S_L diverges with L.

    Parameters
    ----------
    covariance : (2L, 2L) ndarray
        Majorana covariance matrix.

    Returns
    -------
    float
        The topological entanglement entropy.

    Raises
    ------
    ValueError
        If L is not divisible by 4.
    """
    n = covariance.shape[0]
    if n % 8 != 0:
        raise ValueError(
            f"topological_entanglement_entropy requires L divisible by 4, "
            f"got covariance shape {n}x{n} implying L={n//2}"
        )
    L = n // 2
    q = L // 4
    Gamma = np.asarray(covariance, dtype=np.float64)

    # Convert Majorana covariance -> (G, F) correlation matrices via unitary
    # c_j = (gamma_{2j} + i*gamma_{2j+1})/2, so <c^dag_j c_k> and <c_j c_k>
    # are obtained from the Majorana two-point function <gamma_a gamma_b> = delta_{ab} + i*Gamma_{ab}.
    U_mat = np.zeros((L, n), dtype=np.complex128)
    for j in range(L):
        U_mat[j, 2 * j] = 0.5
        U_mat[j, 2 * j + 1] = 0.5j
    M = np.eye(n) + 1j * Gamma
    G = U_mat @ M @ U_mat.conj().T   # G_jk = <c_j^dag c_k>
    F = U_mat @ M @ U_mat.T          # F_jk = <c_j c_k>

    def _region_entropy(G, F, idx):
        """Von Neumann entropy of a region from the (G, F) correlation matrix (log2)."""
        idx = np.asarray(idx)
        G_X = G[np.ix_(idx, idx)]
        F_X = F[np.ix_(idx, idx)]
        n_x = len(idx)
        C_X = np.block([
            [G_X, F_X],
            [-F_X.conj(), np.eye(n_x) - G_X.T]
        ])
        eigvals = np.linalg.eigvalsh(C_X)
        eps = 1e-14
        eigvals = np.clip(eigvals.real, eps, 1.0 - eps)
        # BdG eigenvalues come in (lambda, 1-lambda) pairs; divide by 2 to avoid double-counting.
        return 0.5 * float(-np.sum(eigvals * np.log2(eigvals) + (1.0 - eigvals) * np.log2(1.0 - eigvals)))

    # ABDC partition: A=[0,q), B=[q,2q), C=[2q,3q), D=[3q,4q)
    A = np.arange(0, q)
    B = np.arange(q, 2 * q)
    C = np.arange(2 * q, 3 * q)
    D = np.arange(3 * q, 4 * q)

    AB = np.concatenate([A, B])
    BC = np.concatenate([B, C])

    S_AB = _region_entropy(G, F, AB)
    S_BC = _region_entropy(G, F, BC)
    S_B = _region_entropy(G, F, B)
    S_D = _region_entropy(G, F, D)

    return S_AB + S_BC - S_B - S_D



# ---------------------------------------------------------------------------
# Batched Born-rule trajectory advance (single cloning interval, N_c clones)
# ---------------------------------------------------------------------------

def gaussian_born_rule_trajectory_batched(
    model: GaussianChainModel,
    T: float,
    rngs,                              # Sequence[np.random.Generator], len = N_c
    cov_stack: np.ndarray,             # (N_c, 2L, 2L), float64
    orbs_stack: np.ndarray,            # (N_c, 2L, L),  complex128
    bisection_tol: float = 1e-6,
    *,
    ja_cached: Optional[np.ndarray] = None,
    jb_cached: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance N_c independent Born-rule trajectories over [0, T] in batch.

    Hybrid strategy:
      1. Batched single-shot survival check at t = T for all clones.
      2. Clones whose draw indicates "no jump in [0, T]" go through a
         fully batched no-jump branch (one matmul, one batched QR, one
         batched outer-product for the covariances).
      3. Clones that jump fall back to the scalar gaussian_born_rule_trajectory
         for the remainder of T, reusing the U_i already drawn (so no RNG
         draws are wasted).

    Statistical (not bit-exact) equivalent to running the scalar function
    on each clone independently — small differences arise because batched
    LAPACK calls accumulate floating-point in a different order than serial
    calls.  Validation is via Monte Carlo agreement on S_mean over many seeds.

    Parameters
    ----------
    model : GaussianChainModel
        Source of cached eigendecomposition and jump pairs.
    T : float
        Duration of the interval (typically delta_tau in cloning).
    rngs : Sequence[np.random.Generator]
        Length-N_c list of independent RNGs, one per clone.
    cov_stack, orbs_stack : ndarray
        Stacked initial states.  Both are modified in place is NOT done — new
        arrays are returned to keep the function pure.
    ja_cached, jb_cached : ndarray, optional
        Precomputed jump-pair index arrays.

    Returns
    -------
    new_cov_stack : (N_c, 2L, 2L) float64
    new_orbs_stack : (N_c, 2L, L) complex128
    n_jumps : (N_c,) int64
    """
    N_c = cov_stack.shape[0]
    n_dim = cov_stack.shape[1]
    L = model.L
    n_monitored = len(model.jump_pairs)

    if orbs_stack.shape != (N_c, n_dim, L):
        raise ValueError(
            f"orbs_stack shape {orbs_stack.shape} inconsistent with "
            f"cov_stack {cov_stack.shape}"
        )
    if len(rngs) != N_c:
        raise ValueError(f"len(rngs)={len(rngs)} != N_c={N_c}")

    evals = model.h_eff_evals
    V     = model.h_eff_V
    V_inv = model.h_eff_V_inv
    VhV   = model.h_eff_VhV

    if ja_cached is None or jb_cached is None:
        _jp = model.jump_pairs
        ja = np.array([p[0] for p in _jp], dtype=np.intp)
        jb = np.array([p[1] for p in _jp], dtype=np.intp)
    else:
        ja = ja_cached
        jb = jb_cached

    # Cast inputs to expected dtypes (cheap if already correct).
    orbs_in = np.ascontiguousarray(orbs_stack, dtype=np.complex128)
    cov_in  = np.ascontiguousarray(cov_stack,  dtype=np.float64)

    # --- Step 1: batched coefficients V⁻¹ @ orbs_i for all clones -----------
    # einsum('ij,njk->nik', V_inv, orbs) → (N_c, 2L, L)
    coeffs_stack = np.einsum('ij,njk->nik', V_inv, orbs_in, optimize=True)

    # --- Step 2: batched survival probability at t = T ----------------------
    exp_d_T = np.exp(evals * T)                                # (2L,) complex
    A_stack = exp_d_T[None, :, None] * coeffs_stack            # (N_c, 2L, L)

    # gram_i = A_i^† (V^†V) A_i  (shape L×L per clone, PSD Hermitian)
    VhV_A = np.einsum('ij,njk->nik', VhV, A_stack, optimize=True)        # (N_c, 2L, L)
    gram_stack = np.einsum('nij,nik->njk', A_stack.conj(), VhV_A, optimize=True)  # (N_c, L, L)

    sign_stack, logdet_stack = np.linalg.slogdet(gram_stack)
    log_S_stack = 0.5 * np.asarray(logdet_stack, dtype=np.complex128).real \
                  - model.alpha * n_monitored * T
    # Degenerate (sign=0) → -inf so the clone is forced into the jumping branch.
    log_S_stack = np.where(np.abs(sign_stack) > 0.0, log_S_stack, -np.inf)

    # --- Step 3: draw U_i for all clones, decide jump vs no-jump -----------
    U_stack = np.empty(N_c, dtype=np.float64)
    for i in range(N_c):
        U_stack[i] = float(rngs[i].uniform(0.0, 1.0))
    log_U_stack = np.log(np.clip(U_stack, 1e-300, 1.0))

    no_jump_mask = log_S_stack >= log_U_stack
    no_jump_idx  = np.flatnonzero(no_jump_mask)
    jump_idx     = np.flatnonzero(~no_jump_mask)

    # --- Output buffers ----------------------------------------------------
    new_orbs_stack = np.empty_like(orbs_in)
    new_cov_stack  = np.empty_like(cov_in)
    n_jumps        = np.zeros(N_c, dtype=np.int64)

    # --- Step 4: batched no-jump branch ------------------------------------
    if no_jump_idx.size > 0:
        # M(T) V_i = V @ A_i (we already have A_stack)
        orbs_tilde_nj = np.einsum(
            'ij,njk->nik', V, A_stack[no_jump_idx], optimize=True,
        )                                                       # (n_nj, 2L, L)

        # Batched thin QR.  numpy.linalg.qr broadcasts over leading axes.
        q_stack, _ = np.linalg.qr(orbs_tilde_nj, mode='reduced')
        new_orbs_stack[no_jump_idx] = q_stack

        # Batched covariance: cov_i = i (2 Q_i Q_i^† − I), take real part,
        # antisymmetrise.  For valid orthonormal Q the imaginary part is
        # ~machine epsilon; we drop it without an explicit real_if_close
        # check because the cost would be O(n_nj·L^2) per clone and the
        # downstream consumers (apply_projective_jump, channel selection)
        # already coerce/clip values.
        QQt = np.einsum('nij,nkj->nik', q_stack, q_stack.conj(), optimize=True)
        Eye = np.eye(n_dim)
        cov_complex = 1j * (2.0 * QQt - Eye[None, :, :])
        cov_real = np.ascontiguousarray(cov_complex.real, dtype=np.float64)
        cov_real = 0.5 * (cov_real - cov_real.transpose(0, 2, 1))
        new_cov_stack[no_jump_idx] = cov_real
        # n_jumps already 0 for these clones.

    # --- Step 5: serial fallback for jumping clones ------------------------
    # We pass U_stack[i] as initial_uniform_override so the scalar function
    # uses the same uniform draw we already consumed for the survival check
    # — preserving the conditional distribution of jump times.  Subsequent
    # within-interval draws (for any second jump, channel selection) come
    # from rngs[i] as usual.
    for i in jump_idx:
        result = gaussian_born_rule_trajectory(
            model,
            T=T,
            rng=rngs[i],
            bisection_tol=bisection_tol,
            gamma0_override=cov_in[i],
            orbitals0_override=orbs_in[i],
            ja_cached=ja,
            jb_cached=jb,
            initial_uniform_override=float(U_stack[i]),
        )
        new_cov_stack[i]  = result.final_covariance
        new_orbs_stack[i] = result.final_orbitals
        n_jumps[i]        = result.n_jumps

    return new_cov_stack, new_orbs_stack, n_jumps
