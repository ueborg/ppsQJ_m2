"""Population-dynamics (cloning) algorithm for the tilted path measure.

Implements P_zeta(Traj) \\propto zeta^{N_T} P_Born(Traj) by evolving a population
of N_c Gaussian (covariance-matrix) clones in parallel, tilting trajectory
weights by zeta^{n_jumps} per resampling window, and performing systematic
resampling between windows. Estimates the scaled-cumulant generating function
theta(zeta) = lim_{T -> inf} (1/T) log <zeta^{N_T}>_Born, together with the
ensemble-averaged half-chain entanglement S_zeta.

The Born-rule trajectory driver is ``gaussian_born_rule_trajectory`` from
``pps_qj.gaussian_backend`` (exact bisection-based QJ unraveling).

Jack-Sollich feedback control
------------------------------
When a pre-computed ``GaussianBackwardData`` (or ``LoadedBackwardPass``) object
is supplied via the ``backward_data`` keyword, the per-window clone weight is
augmented by the control ratio

    w_i^JS = zeta^{n_i} * u(Gamma_i(t), t) / u(Gamma_i(t - delta_tau), t - delta_tau)

where u(Gamma, t) = Tr(G_t rho_Gamma) is the Gaussian approximation to the
Doob backward operator, evaluated via ``_batched_log_u``. This is an exact
variance-reduction technique: the control potential cancels in expectation
(no bias) but reduces weight variance near the phase transition, allowing
the same statistical accuracy with a smaller clone population.

The Gaussian G_t approximation that failed catastrophically as a direct sampler
is adequate here because the cloning resampling step corrects for any
inaccuracy in u — the correction only needs to capture the rough shape of the
weight distribution, not its precise values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .gaussian_backend import (
    GaussianChainModel,
    entanglement_entropy,
    gaussian_born_rule_trajectory,
)
from .overlaps import log_gaussian_overlap_from_orbitals


# ---------------------------------------------------------------------------
# Multiprocessing worker (must be at module level so it can be pickled)
# ---------------------------------------------------------------------------

# Process-local model cache, populated by the pool initializer.
# Avoids re-pickling the ~5 MB model object on every step.
_WORKER_MODEL: Optional[GaussianChainModel] = None


def _init_clone_worker(model: GaussianChainModel) -> None:
    """Initializer for clone worker processes — caches the model in a global."""
    global _WORKER_MODEL
    _WORKER_MODEL = model


def _worker_evolve_one_clone(args):
    """Evolve a single clone in a worker process.

    The model is read from the process-local global ``_WORKER_MODEL`` (set
    once per worker by ``_init_clone_worker``), so only the small per-clone
    state (cov, orb, dt, rng) is transferred per call.
    """
    from dataclasses import replace
    cov, orb, dt, sub_rng = args
    sub_model = replace(_WORKER_MODEL, gamma0=cov, orbitals0=orb)
    r = gaussian_born_rule_trajectory(sub_model, T=dt, rng=sub_rng)
    return (
        np.asarray(r.final_covariance, dtype=np.float64),
        r.final_orbitals,
        int(r.n_jumps),
    )


def _batched_entanglement_entropy(
    covs: list[np.ndarray],
    ell: int,
    base: float = 2.0,
) -> np.ndarray:
    """Vectorised half-chain entanglement entropy for a population of clones.

    Replaces a Python loop over N_c calls to ``entanglement_entropy`` with a
    single batched eigvalsh on stacked 2ell×2ell submatrices. For a real
    antisymmetric covariance subblock C, the matrix iC is complex Hermitian, so
    ``eigvalsh`` is both faster and more numerically stable than the general
    ``eigvals`` path previously used. Falls back to the serial per-clone
    computation if eigvalsh fails (can occur at L=32+ when covariance matrices
    accumulate numerical drift).
    """
    two_ell = 2 * ell
    log_fn  = np.log2 if base == 2.0 else np.log

    # Try batched path first
    try:
        subs = np.stack([c[:two_ell, :two_ell] for c in covs], axis=0)  # (N_c, 2*ell, 2*ell)
        # iC is complex Hermitian (C real antisymmetric => (iC)† = -iC^T = iC).
        # eigvalsh exploits Hermitian structure: faster and more stable than eigvals.
        eigs = np.linalg.eigvalsh((1j * subs).astype(np.complex128))    # (N_c, 2*ell) ascending
        nus  = np.abs(eigs[:, ell:])   # positive half of ±ν spectrum, shape (N_c, ell)
        nus  = np.clip(nus, 0.0, 1.0)
        p_plus  = np.clip(0.5 * (1.0 + nus), 1e-15, 1.0 - 1e-15)
        p_minus = np.clip(0.5 * (1.0 - nus), 1e-15, 1.0 - 1e-15)
        S = -np.sum(p_plus * log_fn(p_plus) + p_minus * log_fn(p_minus), axis=-1)
        return S.astype(np.float64)
    except np.linalg.LinAlgError:
        # Batched eigvalsh failed (poorly conditioned matrix at large L).
        # Fall back to serial computation which uses eigh on the antisymmetric
        # form and is more numerically stable.
        return np.array(
            [entanglement_entropy(c, ell) for c in covs], dtype=np.float64
        )


def _batched_log_u(
    backward_orbitals: np.ndarray,
    forward_orbitals_batch: np.ndarray,
    log_z: float,
    L: int,
) -> np.ndarray:
    """Vectorised log_gaussian_overlap_from_orbitals over all N_c clones.

    Replaces N_c serial calls to ``log_gaussian_overlap_from_orbitals`` with a
    single batched matrix multiply and a batched slogdet. The memory cost is
    N_c L×L complex matrices; at L=64, N_c=200 this is 200 * 64^2 * 16B ≈ 13 MB,
    acceptable.

    Parameters
    ----------
    backward_orbitals      : (2L, L) complex — backward Gaussian operator orbitals W
    forward_orbitals_batch : (N_c, 2L, L) complex — stacked forward state orbitals V_i
    log_z                  : float — log of the Gaussian operator normalisation z_scalar
    L                      : int — chain length (= forward_orbitals.shape[-1])

    Returns
    -------
    (N_c,) float64 array of log-overlaps; entries are -inf where det(W†V_i) = 0.

    Notes
    -----
    The mathematical identity used is (Kells et al., SciPost 2023, appendix)::

        det(I - C Gamma_i) = 4^L |det(W† V_i)|^2

    so::

        log Tr(G rho_i) = log_z + L log 4 + 2 Re log|det(W† V_i)|.

    This replaces an O((2L)^3) slogdet on a 2L×2L matrix with an O(L^3)
    slogdet on an L×L matrix — 8× cheaper at L=64 — and vectorises over N_c.
    """
    # W† V_i = (L, 2L) @ (2L, L) = (L, L) for each i.
    # NumPy matmul broadcasts (L, 2L) against (N_c, 2L, L) → (N_c, L, L).
    WdV_batch = backward_orbitals.conj().T @ forward_orbitals_batch  # (N_c, L, L)
    sign, logdet = np.linalg.slogdet(WdV_batch)   # (N_c,) each
    # W†V is complex, so slogdet returns a complex `sign` = exp(i·arg(det)).
    # The condition `sign > 0` would silently check Re(sign) > 0 — i.e. whether
    # arg(det) ∈ (-π/2, π/2) — and incorrectly returns -inf for the other half
    # of the unit circle.  The phase is irrelevant: we only need log|det(W†V)|,
    # which is `logdet` (always real; = -inf iff det=0).  Use isfinite(logdet)
    # as the sole validity gate.
    logdet_real = np.asarray(logdet, dtype=np.float64).real
    log_overlap = np.where(
        np.isfinite(logdet_real),
        log_z + L * np.log(4.0) + 2.0 * logdet_real,
        -np.inf,
    )
    return log_overlap.astype(np.float64)


__all__ = [
    "CloningCollapse",
    "CloningResult",
    "run_cloning",
    "sweep_zeta",
    "_systematic_resample",
]


class CloningCollapse(RuntimeError):
    """Raised when the entire clone population is killed in a single step."""


@dataclass
class CloningResult:
    theta_hat: float
    S_mean: float
    S_std: float
    S_history: np.ndarray
    log_Z_history: np.ndarray
    W_history: np.ndarray
    n_collapses: int
    n_burnin_steps: int
    eff_sample_size: float
    zeta: float
    L: int
    alpha: float
    w: float
    N_c: int
    T_total: float
    delta_tau: float
    final_covs: list  # N_c covariance matrices after last resampling step
    n_js_fallbacks: int = 0  # number of (clone, step) pairs where JS correction was NaN/inf


def _systematic_resample(
    covs: list[np.ndarray],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Systematic resampling of covariances only (kept for external callers)."""
    weights = np.asarray(weights, dtype=np.float64)
    total   = float(weights.sum())
    if total <= 0.0:
        raise CloningCollapse("Cannot resample: all weights zero.")
    N_c   = len(covs)
    F     = np.cumsum(weights / total); F[-1] = 1.0
    U     = float(rng.uniform(0.0, 1.0 / N_c))
    idxs  = np.clip(np.searchsorted(F, U + np.arange(N_c) / N_c, side="left"), 0, N_c - 1)
    return [covs[int(i)].copy() for i in idxs]


def _systematic_resample_pairs(
    covs: list[np.ndarray],
    orbs: list[np.ndarray],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Systematic resampling of (covariance, orbital) pairs.

    Resamples both arrays with the same index sequence so covariances and
    orbitals stay consistent without an extra orbitals_from_covariance call.
    """
    weights = np.asarray(weights, dtype=np.float64)
    total   = float(weights.sum())
    if total <= 0.0:
        raise CloningCollapse("Cannot resample: all weights zero.")
    N_c  = len(covs)
    F    = np.cumsum(weights / total); F[-1] = 1.0
    U    = float(rng.uniform(0.0, 1.0 / N_c))
    idxs = np.clip(np.searchsorted(F, U + np.arange(N_c) / N_c, side="left"), 0, N_c - 1)
    return [(covs[int(i)].copy(), orbs[int(i)].copy()) for i in idxs]


def _step_log_weight(n_jumps: int, zeta: float) -> float:
    """Per-clone trajectory weight w[i] under the tilt zeta^{n_jumps}."""
    if zeta == 0.0:
        return 1.0 if n_jumps == 0 else 0.0
    if zeta == 1.0:
        return 1.0
    return float(np.exp(n_jumps * np.log(zeta)))


def run_cloning(
    model: GaussianChainModel,
    zeta: float,
    T_total: float,
    N_c: int,
    rng: np.random.Generator,
    delta_tau: Optional[float] = None,
    n_burnin_frac: float = 0.25,
    record_entropy: bool = True,
    show_progress: bool = False,
    progress_desc: str = "",
    backward_data: Any = None,
    n_workers: int = 1,
) -> CloningResult:
    """Population-dynamics estimator of theta(zeta) and S_zeta.

    Each of N_c clones carries a Majorana covariance matrix; during each
    resampling window of length delta_tau, clones evolve independently under
    the exact Born-rule unraveling (``gaussian_born_rule_trajectory``). After
    the window, trajectory weights w_i = zeta^{n_jumps,i} are accumulated into
    log Z, the weighted half-chain entropy is recorded (BEFORE resampling),
    and the population is systematically resampled (skipped at zeta==1 since
    uniform weights make resampling a no-op but bias-free).

    Jack-Sollich feedback control
    ------------------------------
    If ``backward_data`` is not None, the per-window weight is augmented by the
    control ratio u(Gamma_i(t), t) / u(Gamma_i(t - delta_tau), t - delta_tau),
    where u is the Gaussian approximation to the Doob backward operator. This
    reduces weight variance near the phase transition without introducing bias.
    ``backward_data`` must expose an ``orbitals_at(t) -> (W, log_z)`` method
    (compatible with both ``GaussianBackwardData`` and ``LoadedBackwardPass``).
    Its ``.T`` attribute must match ``T_total`` to within 1e-6 * T_total.

    Implementation notes
    --------------------
    * A fresh sub-rng is spawned per clone per step from the master rng so that
      seeds never repeat across steps.
    * Initialisation copies ``model.gamma0`` into each clone; the trajectory
      driver reads its initial state from ``model.gamma0``/``model.orbitals0``,
      so we pass a shallow-copy ``GaussianChainModel`` with gamma0 set to the
      current clone covariance for the sub-step.
    """
    if zeta < 0.0:
        raise ValueError("zeta must be non-negative")
    if T_total <= 0.0:
        raise ValueError("T_total must be positive")
    if N_c < 1:
        raise ValueError("N_c must be >= 1")

    L = model.L
    alpha = model.alpha
    w = model.w

    if delta_tau is None:
        delta_tau = 1.0 / max(2.0 * alpha * (L - 1), 1e-6)

    n_steps = max(1, int(np.ceil(T_total / delta_tau)))
    # Recompute effective delta_tau so n_steps * delta_tau == T_total exactly
    delta_tau_eff = T_total / n_steps

    # Validate backward_data compatibility
    use_js = backward_data is not None
    if use_js:
        bwd_T = float(getattr(backward_data, 'T', T_total))
        if abs(bwd_T - T_total) > 1e-6 * T_total:
            raise ValueError(
                f"backward_data.T = {bwd_T:.6g} does not match T_total = {T_total:.6g}. "
                "Pre-compute the backward pass with the same T as the cloning run."
            )

    # Per-clone covariances and orbitals (independent copies of initial state).
    # Storing orbitals alongside covariances avoids re-calling
    # orbitals_from_covariance at the start of every trajectory segment.
    covs: list[np.ndarray] = [model.gamma0.copy() for _ in range(N_c)]
    orbs: list[np.ndarray] = [model.orbitals0.copy() for _ in range(N_c)]

    log_Z_acc = 0.0
    log_Z_history: list[float] = []
    W_history: list[float] = []
    S_history: list[float] = []
    n_collapses = 0
    n_js_fallbacks = 0
    final_weights = np.ones(N_c, dtype=np.float64)

    # Pre-compute burnin boundary so it can be used inside the loop.
    n_burnin_steps = int(n_steps * n_burnin_frac)

    # Lazy import to avoid circular; reuse dataclasses.replace pattern
    from dataclasses import replace

    # ---- Set up multiprocessing pool if requested ----
    # Each worker process holds a cached copy of the model in a global,
    # populated once via _init_clone_worker.  Per-step IPC carries only the
    # per-clone state (cov, orb, dt, sub_rng) — small relative to the per-clone
    # compute cost at L >= 32.  Pinning BLAS to 1 thread per worker (set
    # OMP_NUM_THREADS=1 in the SLURM script) is essential to prevent thread
    # contention across processes.
    pool = None
    if n_workers > 1:
        import multiprocessing as mp
        pool = mp.Pool(
            processes=n_workers,
            initializer=_init_clone_worker,
            initargs=(model,),
        )

    if show_progress:
        from tqdm import tqdm
        step_iter = tqdm(
            range(n_steps),
            desc=progress_desc or f"L={L} ζ={zeta:.2f}",
            unit="step",
            leave=False,
        )
    else:
        step_iter = range(n_steps)

    try:
        for _k in step_iter:
            # ---- Save pre-evolution orbitals for JS control (if enabled) ----
            # The JS weight ratio u(t) / u(t - delta_tau) needs the state at the
            # START of this window (before trajectory evolution). Since the
            # trajectory evolution replaces orbs[i] in-place (by reference), we
            # capture the current list of array references before the inner loop.
            # _systematic_resample_pairs already called .copy() on the arrays, so
            # these are independent copies from the previous resampling step.
            if use_js:
                orbs_pre = list(orbs)  # shallow copy of list; arrays are not yet mutated

            # ---- Evolve all clones independently ----
            sub_rngs = rng.spawn(N_c)
            n_jumps = np.zeros(N_c, dtype=np.int64)
            if pool is not None:
                # Parallel path: dispatch all clone trajectories to worker pool.
                # The model is read from each worker's process-local global, so
                # IPC carries only (cov, orb, dt, rng) per clone per step.
                args_list = [
                    (covs[i], orbs[i], delta_tau_eff, sub_rngs[i])
                    for i in range(N_c)
                ]
                results = pool.map(_worker_evolve_one_clone, args_list)
                for i, (cov_i, orb_i, nj_i) in enumerate(results):
                    covs[i]    = cov_i
                    orbs[i]    = orb_i
                    n_jumps[i] = nj_i
            else:
                # Serial path (n_workers == 1): identical to the original loop.
                for i in range(N_c):
                    sub_model = replace(model, gamma0=covs[i], orbitals0=orbs[i])
                    result    = gaussian_born_rule_trajectory(
                        sub_model, T=delta_tau_eff, rng=sub_rngs[i]
                    )
                    covs[i]    = np.asarray(result.final_covariance, dtype=np.float64)
                    orbs[i]    = result.final_orbitals
                    n_jumps[i] = int(result.n_jumps)

            # ---- Compute weights ----
            # For zeta==0 and zeta==1 the weight is trivially determined; the
            # general case uses log-sum-exp to handle the potentially large
            # JS correction log_u_now - log_u_prev without overflow.
            if zeta == 0.0:
                weights = (n_jumps == 0).astype(np.float64)
                lw_max = 0.0  # not used below, but avoids NameError
                log_W_k = float(np.log(weights.mean())) if weights.any() else -np.inf
            elif zeta == 1.0:
                weights = np.ones(N_c, dtype=np.float64)
                lw_max = 0.0
                log_W_k = 0.0
            else:
                log_w = n_jumps * np.log(zeta)  # (N_c,), all ≤ 0

                if use_js:
                    # Physical times bracketing this resampling window
                    t_start = _k * delta_tau_eff
                    t_end   = (_k + 1) * delta_tau_eff

                    W_prev, log_z_prev = backward_data.orbitals_at(t_start)
                    W_now,  log_z_now  = backward_data.orbitals_at(t_end)

                    # Stack pre-evolution orbitals: (N_c, 2L, L)
                    orbs_pre_batch = np.stack(orbs_pre, axis=0).astype(np.complex128)
                    # Stack post-evolution orbitals: (N_c, 2L, L)
                    orbs_now_batch = np.stack(orbs, axis=0).astype(np.complex128)

                    log_u_prev = _batched_log_u(W_prev, orbs_pre_batch, log_z_prev, L)
                    log_u_now  = _batched_log_u(W_now,  orbs_now_batch, log_z_now,  L)

                    log_js = log_u_now - log_u_prev  # (N_c,)
                    valid  = np.isfinite(log_js)
                    n_js_fallbacks += int((~valid).sum())
                    # Clones where the overlap is numerically invalid fall back to
                    # the standard zeta^n weight (log_js correction = 0 for them).
                    log_w = log_w + np.where(valid, log_js, 0.0)

                # log-sum-exp trick: shift by max(log_w) before exp to prevent
                # overflow. For standard cloning (no JS) log_w ≤ 0 so lw_max ≤ 0
                # and exp never overflows; the trick is a no-op in that case.
                lw_max = float(log_w.max())
                if not np.isfinite(lw_max):
                    weights  = np.zeros(N_c, dtype=np.float64)
                    log_W_k  = -np.inf
                else:
                    w_rel    = np.exp(log_w - lw_max)   # (N_c,) in (0, 1] range
                    mean_rel = float(w_rel.mean())
                    log_W_k  = (np.log(mean_rel) + lw_max) if mean_rel > 0.0 else -np.inf
                    weights  = w_rel  # relative weights suffice for resampling and entropy

            W_k = float(np.exp(log_W_k)) if np.isfinite(log_W_k) else 0.0
            W_history.append(W_k)
            if W_k <= 0.0:
                n_collapses += 1
                raise CloningCollapse(
                    f"Population collapsed at step {_k + 1}/{n_steps} "
                    f"(zeta={zeta}, n_jumps min={int(n_jumps.min())})."
                )
            log_Z_acc += log_W_k
            log_Z_history.append(log_Z_acc)

            if record_entropy:
                S_vals = _batched_entanglement_entropy(covs, L // 2)
                w_sum = float(weights.sum())
                if w_sum > 0.0:
                    S_zeta_k = float(np.sum(weights * S_vals) / w_sum)
                else:
                    S_zeta_k = float(np.mean(S_vals))
                S_history.append(S_zeta_k)
                if show_progress:
                    step_iter.set_postfix({"S": f"{S_zeta_k:.3f}", "W": f"{W_k:.3e}"})

            final_weights = weights.copy()

            # Resampling step (skip at zeta==1: uniform weights -> trivial no-op,
            # keeps streams identical to independent evolution).
            if zeta != 1.0:
                pairs   = _systematic_resample_pairs(covs, orbs, weights, rng)
                covs, orbs = zip(*pairs)
                covs = list(covs)
                orbs = list(orbs)
            # else: leave covs untouched (all weights equal anyway)
    finally:
        # Always shut down the pool, even if a CloningCollapse propagated
        # out of the for loop.  Without this, child processes leak and
        # progressively eat into the per-task SLURM memory allocation.
        if pool is not None:
            pool.close()
            pool.join()

    log_Z_arr = np.asarray(log_Z_history, dtype=np.float64)
    W_arr = np.asarray(W_history, dtype=np.float64)
    S_arr = np.asarray(S_history, dtype=np.float64) if record_entropy else np.asarray([])

    theta_hat = log_Z_acc / T_total

    if record_entropy and len(S_arr) > n_burnin_steps:
        S_mean = float(np.mean(S_arr[n_burnin_steps:]))
        S_std = float(np.std(S_arr[n_burnin_steps:]))
    else:
        S_mean = float("nan")
        S_std = float("nan")

    fw_sum = float(final_weights.sum())
    fw_sq = float(np.sum(final_weights ** 2))
    eff_sample_size = (fw_sum ** 2) / fw_sq if fw_sq > 0 else 0.0

    return CloningResult(
        theta_hat=theta_hat,
        S_mean=S_mean,
        S_std=S_std,
        S_history=S_arr,
        log_Z_history=log_Z_arr,
        W_history=W_arr,
        n_collapses=n_collapses,
        n_burnin_steps=n_burnin_steps,
        eff_sample_size=eff_sample_size,
        zeta=float(zeta),
        L=int(L),
        alpha=float(alpha),
        w=float(w),
        N_c=int(N_c),
        T_total=float(T_total),
        delta_tau=float(delta_tau_eff),
        final_covs=covs,
        n_js_fallbacks=int(n_js_fallbacks),
    )


def sweep_zeta(
    model: GaussianChainModel,
    zeta_vals,
    T_total: float,
    N_c: int,
    rng: np.random.Generator,
    **kwargs,
) -> list[CloningResult]:
    """Run ``run_cloning`` over an array of zeta values, one sub-rng each.

    On CloningCollapse we re-raise with a message indicating which zeta
    failed; the caller can catch per-zeta if partial sweeps are desired.
    """
    sub_rngs = rng.spawn(len(zeta_vals))
    out: list[CloningResult] = []
    for idx, z in enumerate(zeta_vals):
        try:
            out.append(
                run_cloning(model, float(z), T_total, N_c, sub_rngs[idx], **kwargs)
            )
        except CloningCollapse as exc:
            raise CloningCollapse(
                f"sweep_zeta collapsed at zeta={float(z):.4f} "
                f"(index {idx}): {exc}"
            ) from exc
    return out
