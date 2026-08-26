"""Genealogy-aware adaptive Feynman--Kac cloning for the QJ PPS ensemble.

This module is deliberately separate from :mod:`pps_qj.cloning`.  The current
production sampler resamples at every cloning window for every zeta != 1.
Here the same guided proposal and exact Radon--Nikodym increment are used, but
normalized cumulative particle weights are carried across windows and
resampling is applied only when requested.

Target
------
For proposal rate multiplier ``c`` the exact segment potential is

    G = (zeta/c)**n_jumps * exp(-(1-c) * DeltaLambda).

The default remains the validated matched guide ``c=zeta``.  If resampling is
skipped, particle weights are *not* reset.  The Feynman--Kac normalizer update is

    Z_k / Z_{k-1} = sum_i p_i G_i,

followed by ``p_i <- p_i G_i / sum_j p_j G_j``.

Thus adaptive resampling changes only the particle approximation, not the target
path measure.  The default production algorithm is untouched.

This is experimental until certified at the final-locator level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np

from .gaussian_backend import GaussianChainModel, gaussian_born_rule_trajectory


def _spawn_rngs(rng: np.random.Generator, n: int):
    n = int(n)
    if hasattr(rng, "spawn"):
        return list(rng.spawn(n))
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=n)
    return [np.random.default_rng(int(s)) for s in seeds]


def _systematic_resample_idxs(
    weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    p = np.asarray(weights, dtype=np.float64)
    s = float(p.sum())
    if s <= 0.0:
        raise RuntimeError("cannot resample zero total weight")
    p = p / s
    N = p.size
    F = np.cumsum(p)
    F[-1] = 1.0
    u0 = float(rng.uniform(0.0, 1.0 / N))
    return np.searchsorted(
        F, u0 + np.arange(N, dtype=np.float64) / N, side="left"
    ).astype(np.intp)


def _residual_stratified_resample_idxs(
    weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Unbiased residual-stratified resampling."""
    p = np.asarray(weights, dtype=np.float64)
    p = p / float(p.sum())
    N = p.size
    deterministic = np.floor(N * p).astype(np.int64)
    idx = np.repeat(np.arange(N, dtype=np.intp), deterministic)
    R = N - idx.size
    if R:
        residual = N * p - deterministic
        sr = float(residual.sum())
        if sr <= 0.0:
            extra = rng.integers(0, N, size=R, dtype=np.intp)
        else:
            residual /= sr
            F = np.cumsum(residual)
            F[-1] = 1.0
            u = (np.arange(R, dtype=np.float64) + rng.random(R)) / R
            extra = np.searchsorted(F, u, side="left").astype(np.intp)
        idx = np.concatenate([idx, extra])
    rng.shuffle(idx)
    return idx.astype(np.intp)


def _weighted_genealogical_ess(labels: np.ndarray, p: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.intp)
    p = np.asarray(p, dtype=np.float64)
    p = p / float(p.sum())
    _, inv = np.unique(labels, return_inverse=True)
    fam = np.bincount(inv, weights=p)
    s2 = float(np.sum(fam * fam))
    return 1.0 / s2 if s2 > 0.0 else 0.0


def _lagged_labels(
    parent_maps: list[np.ndarray], lag_windows: int, N_c: int
) -> np.ndarray:
    idx = np.arange(N_c, dtype=np.intp)
    if lag_windows <= 0:
        return idx
    start = max(0, len(parent_maps) - int(lag_windows))
    for k in range(len(parent_maps) - 1, start - 1, -1):
        idx = parent_maps[k][idx]
    return idx


@dataclass
class AdaptiveCloningResult:
    theta_hat: float
    final_covs: list[np.ndarray]
    final_orbitals: list[np.ndarray]
    final_weights: np.ndarray
    eff_sample_size: float
    ess_history: np.ndarray
    min_ess_frac_postburnin: float
    mean_ess_frac_postburnin: float
    n_resampling_events: int
    coalescence_burden: float
    root_genealogical_ess: float
    n_distinct_root_ancestors: int
    lagged_gess: dict[int, float] = field(default_factory=dict)
    resampling_steps: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int64)
    )
    delta_tau: float = float("nan")
    n_steps: int = 0
    mean_jumps_per_clone_window: float = float("nan")


def run_cloning_adaptive(
    model: GaussianChainModel,
    zeta: float,
    T_total: float,
    N_c: int,
    rng: np.random.Generator,
    *,
    delta_tau: Optional[float] = None,
    proposal_c: Optional[float] = None,
    resampling_mode: str = "adaptive",
    ess_threshold: float = 0.90,
    resampling_period: int = 4,
    resampler: str = "systematic",
    n_burnin_frac: float = 0.25,
    lag_windows: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    jump_update_method: str = "lowrank",
    refresh_every: int = 100,
    solver_method: str = "newton",
    eps_hazard: float = 1e-9,
) -> AdaptiveCloningResult:
    """Run exact-target guided SMC with configurable resampling schedule.

    ``resampling_mode='always'`` reproduces the current production resampling
    schedule. ``'adaptive'`` resamples only when cumulative ESS/Nc falls below
    ``ess_threshold``. ``'periodic'`` resamples every ``resampling_period``
    windows.  The proposal remains the matched guide c=zeta unless explicitly
    overridden.
    """
    if not (0.0 <= zeta <= 1.0):
        raise ValueError("zeta must be in [0,1]")
    if T_total <= 0 or N_c < 1:
        raise ValueError("T_total>0 and N_c>=1 required")
    if resampling_mode not in ("always", "adaptive", "periodic"):
        raise ValueError("resampling_mode must be always/adaptive/periodic")
    if not (0.0 < ess_threshold <= 1.0):
        raise ValueError("ess_threshold must be in (0,1]")
    if resampling_period < 1:
        raise ValueError("resampling_period must be >=1")
    if resampler not in ("systematic", "residual_stratified"):
        raise ValueError("unknown resampler")

    c = float(zeta if proposal_c is None else proposal_c)
    if not (0.0 < c <= 1.0):
        raise ValueError("proposal_c must be in (0,1]; zeta=0 needs explicit c>0")

    L, alpha = int(model.L), float(model.alpha)
    if delta_tau is None:
        delta_tau = 1.0 / max(2.0 * alpha * (L - 1), 1e-6)
    n_steps = max(1, int(np.ceil(float(T_total) / float(delta_tau))))
    dt = float(T_total) / n_steps
    n_burn = int(n_steps * float(n_burnin_frac))

    covs = [model.gamma0.copy() for _ in range(N_c)]
    orbs = [model.orbitals0.copy() for _ in range(N_c)]
    sub_rngs = _spawn_rngs(rng, N_c)
    root_ids = np.arange(N_c, dtype=np.intp)
    parent_maps: list[np.ndarray] = []
    p = np.full(N_c, 1.0 / N_c, dtype=np.float64)

    jp = model.jump_pairs
    ja = np.asarray([x[0] for x in jp], dtype=np.intp)
    jb = np.asarray([x[1] for x in jp], dtype=np.intp)

    ess_hist: list[float] = []
    resampling_steps: list[int] = []
    log_Z = 0.0
    coalescence_burden = 0.0
    jump_total = 0

    for k in range(n_steps):
        n_jumps = np.zeros(N_c, dtype=np.int64)
        delta_Lambda = np.zeros(N_c, dtype=np.float64)

        for i in range(N_c):
            tr = gaussian_born_rule_trajectory(
                model, T=dt, rng=sub_rngs[i],
                gamma0_override=covs[i],
                orbitals0_override=orbs[i],
                ja_cached=ja, jb_cached=jb,
                proposal_c=c,
                jump_update_method=jump_update_method,
                refresh_every=refresh_every,
                solver_method=solver_method,
                eps_hazard=eps_hazard,
            )
            covs[i] = tr.final_covariance
            orbs[i] = tr.final_orbitals
            n_jumps[i] = tr.n_jumps
            delta_Lambda[i] = tr.Lambda

        jump_total += int(n_jumps.sum())

        if zeta == 0.0:
            log_g = np.where(
                n_jumps == 0, -(1.0 - c) * delta_Lambda, -np.inf
            )
        else:
            log_g = (
                n_jumps * math.log(zeta / c)
                - (1.0 - c) * delta_Lambda
            )

        finite = np.isfinite(log_g)
        if not finite.any():
            raise RuntimeError(f"population collapsed at step {k+1}/{n_steps}")
        m = float(np.max(log_g[finite]))
        g_rel = np.where(finite, np.exp(log_g - m), 0.0)
        norm_rel = float(np.dot(p, g_rel))
        if norm_rel <= 0.0:
            raise RuntimeError(
                f"population normalizer collapsed at step {k+1}/{n_steps}"
            )

        log_Z += m + math.log(norm_rel)
        p = p * g_rel / norm_rel

        ess = 1.0 / float(np.sum(p * p))
        ess_hist.append(ess)

        exactly_uniform_target = (zeta == 1.0 and c == 1.0)
        if exactly_uniform_target:
            do_resample = False
        elif resampling_mode == "always":
            do_resample = True
        elif resampling_mode == "adaptive":
            do_resample = (ess / N_c) < ess_threshold
        else:
            do_resample = ((k + 1) % resampling_period == 0)

        if do_resample:
            if resampler == "systematic":
                idx = _systematic_resample_idxs(p, rng)
            else:
                idx = _residual_stratified_resample_idxs(p, rng)

            counts = np.bincount(idx, minlength=N_c)
            if N_c > 1:
                coalescence_burden += float(
                    np.sum(counts * (counts - 1)) / (N_c * (N_c - 1))
                )

            covs = [covs[int(i)].copy() for i in idx]
            orbs = [orbs[int(i)].copy() for i in idx]
            root_ids = root_ids[idx]
            parent_maps.append(idx.copy())
            p.fill(1.0 / N_c)
            resampling_steps.append(k)
        else:
            parent_maps.append(np.arange(N_c, dtype=np.intp))

    p = p / float(p.sum())
    final_ess = 1.0 / float(np.sum(p * p))
    root_gess = _weighted_genealogical_ess(root_ids, p)

    lagged = {}
    for lag in lag_windows:
        if lag <= n_steps:
            labs = _lagged_labels(parent_maps, int(lag), N_c)
            lagged[int(lag)] = _weighted_genealogical_ess(labs, p)

    ess_arr = np.asarray(ess_hist, dtype=np.float64)
    pb = ess_arr[n_burn:] if ess_arr.size > n_burn else ess_arr
    min_ess_frac = float(np.min(pb) / N_c) if pb.size else float("nan")
    mean_ess_frac = float(np.mean(pb) / N_c) if pb.size else float("nan")

    _, inv = np.unique(root_ids, return_inverse=True)
    fam_mass = np.bincount(inv, weights=p)
    n_distinct = int(np.sum(fam_mass > 1e-15))

    return AdaptiveCloningResult(
        theta_hat=float(log_Z / T_total),
        final_covs=covs,
        final_orbitals=orbs,
        final_weights=p.copy(),
        eff_sample_size=float(final_ess),
        ess_history=ess_arr,
        min_ess_frac_postburnin=min_ess_frac,
        mean_ess_frac_postburnin=mean_ess_frac,
        n_resampling_events=len(resampling_steps),
        coalescence_burden=float(coalescence_burden),
        root_genealogical_ess=float(root_gess),
        n_distinct_root_ancestors=n_distinct,
        lagged_gess=lagged,
        resampling_steps=np.asarray(resampling_steps, dtype=np.int64),
        delta_tau=dt,
        n_steps=n_steps,
        mean_jumps_per_clone_window=float(jump_total / (N_c * n_steps)),
    )
