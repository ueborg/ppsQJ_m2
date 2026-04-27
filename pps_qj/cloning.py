"""Population-dynamics (cloning) algorithm for the tilted path measure.

Implements P_zeta(Traj) propto zeta^{N_T} P_Born(Traj) by evolving a population
of N_c Gaussian (covariance-matrix) clones in parallel, tilting trajectory
weights by zeta^{n_jumps} per resampling window, and performing systematic
resampling between windows.  Estimates the scaled-cumulant generating function
theta(zeta) = lim_{T->inf} (1/T) log <zeta^{N_T}>_Born together with the
ensemble-averaged half-chain entanglement S_zeta.

Parallelism
-----------
This module is intentionally single-threaded.  Parallelism is applied at the
*realisation* level by the worker (worker_clone_pps.py), which runs N_REAL
independent calls to run_cloning in a multiprocessing.Pool.  This granularity
has negligible IPC overhead (one result dict per realisation) and near-ideal
parallel efficiency.  Intra-clone parallelism was attempted and abandoned:
pool creation and IPC overhead dominated at every tested worker count.
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
# Helpers
# ---------------------------------------------------------------------------

def _batched_entanglement_entropy(
    covs: list[np.ndarray],
    ell: int,
    base: float = 2.0,
) -> np.ndarray:
    """Vectorised half-chain entanglement entropy for a population of clones.

    Uses eigvalsh(iC) rather than eigvals(C): iC is complex Hermitian for a
    real antisymmetric C, so eigvalsh is both correct and ~2x faster.
    Falls back to serial on LinAlgError (can occur at L>=32 with drift).
    """
    two_ell = 2 * ell
    log_fn  = np.log2 if base == 2.0 else np.log
    try:
        subs = np.stack([c[:two_ell, :two_ell] for c in covs], axis=0)
        eigs = np.linalg.eigvalsh((1j * subs).astype(np.complex128))
        nus  = np.abs(eigs[:, ell:])
        nus  = np.clip(nus, 0.0, 1.0)
        p_plus  = np.clip(0.5 * (1.0 + nus), 1e-15, 1.0 - 1e-15)
        p_minus = np.clip(0.5 * (1.0 - nus), 1e-15, 1.0 - 1e-15)
        S = -np.sum(p_plus * log_fn(p_plus) + p_minus * log_fn(p_minus), axis=-1)
        return S.astype(np.float64)
    except np.linalg.LinAlgError:
        return np.array([entanglement_entropy(c, ell) for c in covs], dtype=np.float64)


def _batched_log_u(
    backward_orbitals: np.ndarray,
    forward_orbitals_batch: np.ndarray,
    log_z: float,
    L: int,
) -> np.ndarray:
    """Vectorised log Tr(G_t rho_i) for all N_c clones.

    Uses the identity det(I - C Gamma_i) = 4^L |det(W† V_i)|^2, so the
    2L x 2L slogdet reduces to an L x L slogdet — 8x cheaper at L=64.
    W†V is complex, so slogdet returns sign=exp(i*phi); validity gate is
    isfinite(logdet), not sign>0.
    """
    WdV_batch = backward_orbitals.conj().T @ forward_orbitals_batch
    _, logdet = np.linalg.slogdet(WdV_batch)
    logdet_real = np.asarray(logdet, dtype=np.float64).real
    return np.where(
        np.isfinite(logdet_real),
        log_z + L * np.log(4.0) + 2.0 * logdet_real,
        -np.inf,
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CloningCollapse",
    "CloningResult",
    "run_cloning",
    "sweep_zeta",
    "_systematic_resample",
    "_batched_entanglement_entropy",
    "_batched_log_u",
]


class CloningCollapse(RuntimeError):
    """Raised when the entire clone population collapses to zero weight."""


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
    final_covs: list
    n_js_fallbacks: int = 0


def _systematic_resample(
    covs: list[np.ndarray],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise CloningCollapse("Cannot resample: all weights zero.")
    N_c = len(covs)
    F = np.cumsum(weights / total); F[-1] = 1.0
    U = float(rng.uniform(0.0, 1.0 / N_c))
    idxs = np.clip(np.searchsorted(F, U + np.arange(N_c) / N_c, side="left"), 0, N_c - 1)
    return [covs[int(i)].copy() for i in idxs]


def _systematic_resample_pairs(
    covs: list[np.ndarray],
    orbs: list[np.ndarray],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Resample (covariance, orbital) pairs with the same index sequence."""
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise CloningCollapse("Cannot resample: all weights zero.")
    N_c = len(covs)
    F = np.cumsum(weights / total); F[-1] = 1.0
    U = float(rng.uniform(0.0, 1.0 / N_c))
    idxs = np.clip(np.searchsorted(F, U + np.arange(N_c) / N_c, side="left"), 0, N_c - 1)
    return [(covs[int(i)].copy(), orbs[int(i)].copy()) for i in idxs]


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
) -> CloningResult:
    """Population-dynamics estimator of theta(zeta) and S_zeta.

    Runs the standard cloning algorithm serially.  For parallelism across
    independent realisations, use the realisation-level pool in the worker.

    Parameters
    ----------
    backward_data : optional GaussianBackwardData or LoadedBackwardPass
        If supplied, enables Jack-Sollich feedback control: per-window weights
        become zeta^{n_i} * u(Gamma(t))/u(Gamma(t-dt)).  The Gaussian G_t
        approximation reduces weight variance where it is accurate.  It was
        found to be ineffective near the MIPT critical point (variance ratio
        ~300x worse than standard cloning at the test point), so production
        runs use backward_data=None.
    """
    if zeta < 0.0:
        raise ValueError("zeta must be non-negative")
    if T_total <= 0.0:
        raise ValueError("T_total must be positive")
    if N_c < 1:
        raise ValueError("N_c must be >= 1")

    L, alpha, w = model.L, model.alpha, model.w

    if delta_tau is None:
        delta_tau = 1.0 / max(2.0 * alpha * (L - 1), 1e-6)
    n_steps = max(1, int(np.ceil(T_total / delta_tau)))
    delta_tau_eff = T_total / n_steps

    use_js = backward_data is not None
    if use_js:
        bwd_T = float(getattr(backward_data, 'T', T_total))
        if abs(bwd_T - T_total) > 1e-6 * T_total:
            raise ValueError(
                f"backward_data.T={bwd_T:.6g} != T_total={T_total:.6g}"
            )

    covs: list[np.ndarray] = [model.gamma0.copy() for _ in range(N_c)]
    orbs: list[np.ndarray] = [model.orbitals0.copy() for _ in range(N_c)]

    log_Z_acc = 0.0
    log_Z_history: list[float] = []
    W_history: list[float] = []
    S_history: list[float] = []
    n_collapses = 0
    n_js_fallbacks = 0
    final_weights = np.ones(N_c, dtype=np.float64)
    n_burnin_steps = int(n_steps * n_burnin_frac)

    from dataclasses import replace

    if show_progress:
        from tqdm import tqdm
        step_iter = tqdm(
            range(n_steps),
            desc=progress_desc or f"L={L} ζ={zeta:.2f}",
            unit="step", leave=False,
        )
    else:
        step_iter = range(n_steps)

    for _k in step_iter:
        # JS needs states before trajectory evolution
        if use_js:
            orbs_pre = list(orbs)

        # --- Evolve all clones (serial) ---
        sub_rngs = rng.spawn(N_c)
        n_jumps = np.zeros(N_c, dtype=np.int64)
        for i in range(N_c):
            sub_model = replace(model, gamma0=covs[i], orbitals0=orbs[i])
            r = gaussian_born_rule_trajectory(sub_model, T=delta_tau_eff, rng=sub_rngs[i])
            covs[i]    = np.asarray(r.final_covariance, dtype=np.float64)
            orbs[i]    = r.final_orbitals
            n_jumps[i] = int(r.n_jumps)

        # --- Compute per-step weight W_k = mean_i(w_i) ---
        if zeta == 0.0:
            weights = (n_jumps == 0).astype(np.float64)
            log_W_k = float(np.log(weights.mean())) if weights.any() else -np.inf
        elif zeta == 1.0:
            weights = np.ones(N_c, dtype=np.float64)
            log_W_k = 0.0
        else:
            log_w = n_jumps * np.log(zeta)

            if use_js:
                t_s = _k * delta_tau_eff
                t_e = (_k + 1) * delta_tau_eff
                W_prev, lz_prev = backward_data.orbitals_at(t_s)
                W_now,  lz_now  = backward_data.orbitals_at(t_e)
                orbs_pre_batch = np.stack(orbs_pre, axis=0).astype(np.complex128)
                orbs_now_batch = np.stack(orbs,     axis=0).astype(np.complex128)
                log_u_prev = _batched_log_u(W_prev, orbs_pre_batch, lz_prev, L)
                log_u_now  = _batched_log_u(W_now,  orbs_now_batch, lz_now,  L)
                log_js = log_u_now - log_u_prev
                valid = np.isfinite(log_js)
                n_js_fallbacks += int((~valid).sum())
                log_w = log_w + np.where(valid, log_js, 0.0)

            lw_max = float(log_w.max())
            if not np.isfinite(lw_max):
                weights = np.zeros(N_c, dtype=np.float64)
                log_W_k = -np.inf
            else:
                w_rel    = np.exp(log_w - lw_max)
                mean_rel = float(w_rel.mean())
                log_W_k  = (np.log(mean_rel) + lw_max) if mean_rel > 0.0 else -np.inf
                weights  = w_rel

        W_k = float(np.exp(log_W_k)) if np.isfinite(log_W_k) else 0.0
        W_history.append(W_k)
        if W_k <= 0.0:
            n_collapses += 1
            raise CloningCollapse(
                f"Population collapsed at step {_k+1}/{n_steps} "
                f"(zeta={zeta}, min_jumps={int(n_jumps.min())})."
            )

        log_Z_acc += log_W_k
        log_Z_history.append(log_Z_acc)

        if record_entropy:
            S_vals = _batched_entanglement_entropy(covs, L // 2)
            w_sum  = float(weights.sum())
            S_zeta_k = float(
                np.sum(weights * S_vals) / w_sum if w_sum > 0.0
                else np.mean(S_vals)
            )
            S_history.append(S_zeta_k)
            if show_progress:
                step_iter.set_postfix({"S": f"{S_zeta_k:.3f}", "W": f"{W_k:.3e}"})

        final_weights = weights.copy()

        if zeta != 1.0:
            pairs = _systematic_resample_pairs(covs, orbs, weights, rng)
            covs, orbs = zip(*pairs)
            covs = list(covs)
            orbs = list(orbs)

    log_Z_arr = np.asarray(log_Z_history, dtype=np.float64)
    W_arr     = np.asarray(W_history,     dtype=np.float64)
    S_arr     = np.asarray(S_history,     dtype=np.float64) if record_entropy else np.asarray([])

    theta_hat = log_Z_acc / T_total

    if record_entropy and len(S_arr) > n_burnin_steps:
        S_mean = float(np.mean(S_arr[n_burnin_steps:]))
        S_std  = float(np.std(S_arr[n_burnin_steps:]))
    else:
        S_mean = float("nan")
        S_std  = float("nan")

    fw_sum = float(final_weights.sum())
    fw_sq  = float(np.sum(final_weights ** 2))
    eff_sample_size = (fw_sum ** 2) / fw_sq if fw_sq > 0 else 0.0

    return CloningResult(
        theta_hat=theta_hat,
        S_mean=S_mean, S_std=S_std,
        S_history=S_arr,
        log_Z_history=log_Z_arr,
        W_history=W_arr,
        n_collapses=n_collapses,
        n_burnin_steps=n_burnin_steps,
        eff_sample_size=eff_sample_size,
        zeta=float(zeta), L=int(L), alpha=float(alpha), w=float(w),
        N_c=int(N_c), T_total=float(T_total), delta_tau=float(delta_tau_eff),
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
    """Run run_cloning over an array of zeta values, one sub-rng each."""
    sub_rngs = rng.spawn(len(zeta_vals))
    out: list[CloningResult] = []
    for idx, z in enumerate(zeta_vals):
        try:
            out.append(run_cloning(model, float(z), T_total, N_c, sub_rngs[idx], **kwargs))
        except CloningCollapse as exc:
            raise CloningCollapse(
                f"sweep_zeta collapsed at zeta={float(z):.4f} (index {idx}): {exc}"
            ) from exc
    return out
