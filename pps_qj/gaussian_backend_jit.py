"""Numba-JIT trajectory driver for the Gaussian free-fermion backend.

This is a numerically-equivalent reimplementation of
``gaussian_backend.gaussian_born_rule_trajectory`` compiled with Numba's
``@njit`` decorator.  It eliminates Python interpreter overhead in the
inner WTMC loop, where:

  - ``_fast_branch_norm`` is called multiple times per Brent root-finding
    invocation, each costing one matmul + slogdet on an L×L matrix.
  - The per-jump propagation calls QR on a 2L×L matrix and then a full
    eigh on a 2L×2L matrix (via ``orbitals_from_covariance``).

The compiled version inlines all of the above and uses a numerically
stable bisection root finder in place of ``scipy.optimize.brentq``.

Implementation notes
--------------------

* **RNG**: numba supports ``np.random.uniform`` etc. inside ``@njit`` but
  not the modern ``np.random.Generator`` API. To preserve reproducibility
  while keeping the JIT function thread-safe (no global state surprises)
  the caller pre-draws a buffer of uniform variates outside the JIT and
  passes it in. The function consumes uniforms sequentially from the
  buffer; if it runs out, it raises a ``RuntimeError`` and the caller
  retries with a larger buffer.

* **Jump record arrays**: Numba's ``typed.List`` works but is slow. We
  preallocate fixed-size arrays for ``jump_times`` and ``jump_channels``
  with size ``MAX_JUMPS_PER_TRAJ``, then return them sliced to length
  ``n_jumps``.

* **No closure**: ``_fast_branch_norm`` is inlined in the trajectory loop
  rather than written as a closure (which Numba can sometimes handle but
  is more brittle and harder to vectorise).

Use
---

>>> from pps_qj.gaussian_backend_jit import run_trajectory_jit
>>> # warm up JIT compilation once at the start of a run:
>>> run_trajectory_jit(...)  # first call is slow (~5–15 s compilation)
>>> # subsequent calls run at compiled speed.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: define njit as identity decorator so the module imports
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(f):
            return f
        return deco


# Maximum jumps per single trajectory call. At delta_tau ~ 1/(2*alpha*L)
# the expected jump count is ~1; tail probability of >50 is astronomically
# small. 256 gives a comfortable safety margin without wasting memory.
MAX_JUMPS_PER_TRAJ = 256

# Buffer size for random uniforms per trajectory call. Each iteration of
# the outer while loop consumes 2 uniforms (1 for survival check, 1 for
# channel selection if a jump occurs). 2 * MAX_JUMPS_PER_TRAJ + 16 is
# a safe upper bound.
UNIFORM_BUFFER_SIZE = 2 * MAX_JUMPS_PER_TRAJ + 16


# ---------------------------------------------------------------------------
# JIT-compiled core
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _branch_norm_jit(
    coeffs: np.ndarray,        # (2L, L) complex128
    evals: np.ndarray,         # (2L,)   complex128
    VhV: np.ndarray,           # (2L, 2L) complex128
    alpha_n_monitored: float,
    dt: float,
) -> float:
    """Compute survival probability at time dt.

    Implements ``|det(M(dt) V_0)| * exp(-alpha * n_monitored * dt)``
    via the L×L Gram-matrix slogdet trick.
    """
    if dt <= 0.0:
        return 1.0
    exp_d = np.exp(evals * dt)              # (2L,) complex128
    A = exp_d.reshape(-1, 1) * coeffs        # (2L, L)
    AhV = VhV @ A                            # (2L, L)
    gram = A.conj().T @ AhV                  # (L, L)
    sign, logdet = np.linalg.slogdet(gram)
    if sign.real <= 0.0:
        return 0.0
    return np.exp(0.5 * logdet.real - alpha_n_monitored * dt)


@njit(cache=True, fastmath=False)
def _bisect_jump_time(
    coeffs: np.ndarray, evals: np.ndarray, VhV: np.ndarray,
    alpha_n_monitored: float, U: float, T_rem: float,
    tol: float, max_iter: int,
) -> float:
    """Find dt* such that branch_norm(dt*) = U via bisection.

    branch_norm is monotone-decreasing in dt, so:
      f(dt) = branch_norm(dt) - U
    has f(0)=1-U > 0 (assuming U<1) and we know f(T_rem) < 0 by the time
    we reach this code (caller checked it).
    """
    lo = 0.0
    hi = T_rem
    f_lo = 1.0 - U
    f_hi = _branch_norm_jit(coeffs, evals, VhV, alpha_n_monitored, hi) - U
    # Defensive: shouldn't happen in normal flow but handle it.
    if f_lo * f_hi > 0.0:
        return 0.5 * T_rem
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _branch_norm_jit(coeffs, evals, VhV, alpha_n_monitored, mid) - U
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


@njit(cache=True, fastmath=False)
def _covariance_from_orbitals_jit(orbitals: np.ndarray) -> np.ndarray:
    """gamma = i * (2 * orbitals @ orbitals† - I), made real antisymmetric."""
    n = orbitals.shape[0]
    M = orbitals @ orbitals.conj().T          # (2L, 2L) complex
    M_complex = 1j * (2.0 * M - np.eye(n).astype(np.complex128))
    G = M_complex.real
    # Antisymmetrise to clean up roundoff.
    return 0.5 * (G - G.T)


@njit(cache=True, fastmath=False)
def _orbitals_from_covariance_jit(gamma: np.ndarray) -> np.ndarray:
    """Recover an L-dimensional orthonormal orbital basis from the covariance.

    Equivalent to gaussian_backend.orbitals_from_covariance.
    """
    n = gamma.shape[0]
    L = n // 2
    iC = (1j * gamma).astype(np.complex128)
    eigvals, eigvecs = np.linalg.eigh(iC)
    # Take the L eigenvectors with smallest (most negative) eigenvalues.
    order = np.argsort(eigvals.real)
    orbs = np.empty((n, L), dtype=np.complex128)
    for j in range(L):
        orbs[:, j] = eigvecs[:, order[j]]
    Q, _ = np.linalg.qr(orbs)
    return Q


@njit(cache=True, fastmath=False)
def _apply_jump_jit(
    gamma: np.ndarray, a: int, b: int,
) -> np.ndarray:
    """In-place projective jump on Majorana pair (a, b).

    Mirrors gaussian_backend.apply_projective_jump. Returns the updated
    covariance (a fresh array — no aliasing with input).
    """
    n = gamma.shape[0]
    sigma = gamma[a, b]
    denom = 1.0 - sigma
    if denom <= 1e-14:
        return gamma.copy()  # silent fallback: leave unchanged

    u = gamma[:, a].copy()
    v = gamma[:, b].copy()
    g_new = gamma.copy()

    inv_denom = 1.0 / denom
    for i in range(n):
        if i == a or i == b:
            continue
        for j in range(n):
            if j == a or j == b:
                continue
            g_new[i, j] += (u[i] * v[j] - v[i] * u[j]) * inv_denom

    for k in range(n):
        g_new[a, k] = 0.0
        g_new[k, a] = 0.0
        g_new[b, k] = 0.0
        g_new[k, b] = 0.0
    g_new[a, b] = -1.0
    g_new[b, a] = 1.0
    # Antisymmetrise.
    return 0.5 * (g_new - g_new.T)


@njit(cache=True, fastmath=False)
def _trajectory_jit(
    gamma0: np.ndarray,            # (2L, 2L) float64
    orbitals0: np.ndarray,         # (2L, L)  complex128
    evals: np.ndarray,             # (2L,)    complex128
    V: np.ndarray,                 # (2L, 2L) complex128
    V_inv: np.ndarray,             # (2L, 2L) complex128
    VhV: np.ndarray,               # (2L, 2L) complex128
    ja: np.ndarray,                # (n_pairs,) intp
    jb: np.ndarray,                # (n_pairs,) intp
    alpha: float,
    n_monitored: int,
    T: float,
    uniforms: np.ndarray,          # (UNIFORM_BUFFER_SIZE,) float64
    bisection_tol: float,
):
    """Compiled WTMC trajectory.

    Returns
    -------
    final_covariance : (2L, 2L) float64
    final_orbitals   : (2L, L)  complex128
    n_jumps          : int
    jump_times       : (MAX_JUMPS_PER_TRAJ,) float64 (only [:n_jumps] valid)
    jump_channels    : (MAX_JUMPS_PER_TRAJ,) int64 (only [:n_jumps] valid)
    rng_used         : int  (number of uniforms consumed)
    """
    L = gamma0.shape[0] // 2
    n_pairs = ja.shape[0]
    alpha_nm = alpha * float(n_monitored)

    cov = gamma0.copy()
    orbs = orbitals0.copy()
    t = 0.0

    jump_times    = np.zeros(MAX_JUMPS_PER_TRAJ, dtype=np.float64)
    jump_channels = np.zeros(MAX_JUMPS_PER_TRAJ, dtype=np.int64)
    n_jumps = 0
    rng_idx = 0
    n_unif = uniforms.shape[0]

    while t < T:
        # Survival uniform.
        if rng_idx >= n_unif:
            return cov, orbs, n_jumps, jump_times, jump_channels, -1
        U = uniforms[rng_idx]
        rng_idx += 1

        T_rem = T - t

        # Compute coeffs = V_inv @ orbitals once per outer iteration.
        coeffs = V_inv @ orbs

        bn_end = _branch_norm_jit(coeffs, evals, VhV, alpha_nm, T_rem)

        if bn_end >= U:
            # No jump in remaining time — propagate to T.
            exp_d = np.exp(evals * T_rem)
            orbs_tilde = V @ (exp_d.reshape(-1, 1) * coeffs)
            q_mat, _ = np.linalg.qr(orbs_tilde)
            orbs = q_mat
            cov = _covariance_from_orbitals_jit(q_mat)
            break

        # Find jump time via bisection.
        dt_star = _bisect_jump_time(
            coeffs, evals, VhV, alpha_nm, U, T_rem,
            bisection_tol, 60,
        )

        # Propagate to jump time.
        exp_d_star = np.exp(evals * dt_star)
        orbs_tilde = V @ (exp_d_star.reshape(-1, 1) * coeffs)
        q_mat, _ = np.linalg.qr(orbs_tilde)
        orbs = q_mat
        cov = _covariance_from_orbitals_jit(q_mat)
        t += dt_star

        # Compute jump probabilities.
        probs = np.empty(n_pairs, dtype=np.float64)
        total = 0.0
        for k in range(n_pairs):
            p = 0.5 * (1.0 - cov[ja[k], jb[k]])
            if p < 0.0:
                p = 0.0
            elif p > 1.0:
                p = 1.0
            probs[k] = p
            total += p

        if total < 1e-15:
            break

        # Channel selection uniform.
        if rng_idx >= n_unif:
            return cov, orbs, n_jumps, jump_times, jump_channels, -1
        U_chan = uniforms[rng_idx] * total
        rng_idx += 1

        # Cumulative search.
        cum = 0.0
        channel = n_pairs - 1  # default to last (handles fp edge)
        for k in range(n_pairs):
            cum += probs[k]
            if cum >= U_chan:
                channel = k
                break

        # Apply jump.
        a_idx = int(ja[channel])
        b_idx = int(jb[channel])
        cov = _apply_jump_jit(cov, a_idx, b_idx)
        orbs = _orbitals_from_covariance_jit(cov)

        if n_jumps < MAX_JUMPS_PER_TRAJ:
            jump_times[n_jumps] = t
            jump_channels[n_jumps] = channel
        n_jumps += 1

    return cov, orbs, n_jumps, jump_times, jump_channels, rng_idx


# ---------------------------------------------------------------------------
# Python wrapper that mirrors gaussian_born_rule_trajectory's signature
# ---------------------------------------------------------------------------

class _JitTrajectoryResult:
    """Lightweight result object mirroring GaussianTrajectoryResult."""
    __slots__ = ("final_covariance", "final_orbitals", "n_jumps",
                 "jump_times", "jump_channels")

    def __init__(self, cov, orbs, n_jumps, j_times, j_chans):
        self.final_covariance = cov
        self.final_orbitals   = orbs
        self.n_jumps          = int(n_jumps)
        # Slice to actual length and convert to lists for compat.
        self.jump_times       = list(j_times[:n_jumps])
        self.jump_channels    = list(j_chans[:n_jumps])


def run_trajectory_jit(
    model,                          # GaussianChainModel
    T: float,
    rng: np.random.Generator,
    bisection_tol: float = 1e-8,
    *,
    gamma0_override=None,
    orbitals0_override=None,
    ja_cached=None,
    jb_cached=None,
):
    """JIT-compiled drop-in replacement for gaussian_born_rule_trajectory.

    Same signature as the original (including override kwargs) but routes
    the inner work through the Numba-compiled core.
    """
    if not NUMBA_AVAILABLE:
        raise ImportError(
            "Numba is not installed. Install it (`pip install numba`) or "
            "use gaussian_born_rule_trajectory from gaussian_backend instead."
        )

    # Resolve initial state.
    gamma0 = (np.asarray(gamma0_override, dtype=np.float64).copy()
              if gamma0_override is not None
              else np.asarray(model.gamma0, dtype=np.float64).copy())
    orbitals0 = (np.asarray(orbitals0_override, dtype=np.complex128).copy()
                 if orbitals0_override is not None
                 else np.asarray(model.orbitals0, dtype=np.complex128).copy())

    # Resolve jump-pair index arrays.
    if ja_cached is None or jb_cached is None:
        _jp = model.jump_pairs
        ja = np.array([p[0] for p in _jp], dtype=np.intp)
        jb = np.array([p[1] for p in _jp], dtype=np.intp)
    else:
        ja = np.asarray(ja_cached, dtype=np.intp)
        jb = np.asarray(jb_cached, dtype=np.intp)

    evals = np.asarray(model.h_eff_evals, dtype=np.complex128)
    V     = np.asarray(model.h_eff_V,     dtype=np.complex128)
    V_inv = np.asarray(model.h_eff_V_inv, dtype=np.complex128)
    VhV   = np.asarray(model.h_eff_VhV,   dtype=np.complex128)
    n_monitored = len(model.jump_pairs)

    # Pre-draw uniforms outside JIT — reproducibility comes from the
    # caller's rng. If the buffer is exhausted, retry with a larger one.
    buf_size = UNIFORM_BUFFER_SIZE
    for _attempt in range(3):
        uniforms = rng.uniform(0.0, 1.0, size=buf_size).astype(np.float64)
        cov, orbs, n_jumps, j_times, j_chans, rng_used = _trajectory_jit(
            gamma0, orbitals0, evals, V, V_inv, VhV,
            ja, jb, float(model.alpha), n_monitored,
            float(T), uniforms, float(bisection_tol),
        )
        if rng_used != -1:
            return _JitTrajectoryResult(cov, orbs, n_jumps, j_times, j_chans)
        buf_size *= 4
    # If still failing after 3 attempts, fall back gracefully.
    raise RuntimeError(
        f"JIT trajectory exhausted uniform buffer of size {buf_size} — "
        "this should not occur under normal parameters. Inspect for "
        "pathological input (very large T, very high jump rate)."
    )
