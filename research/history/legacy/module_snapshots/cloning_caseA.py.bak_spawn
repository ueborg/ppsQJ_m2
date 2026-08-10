from __future__ import annotations

"""Case A cloning (population-dynamics) estimator -- scalar production path.

Focused Case A analogue of ``pps_qj.cloning.run_cloning``. Implements ONLY the
scalar production path actually used in Case B production runs. The Jack-Sollich
feedback, batched backend, intermediate snapshots, and Renyi extraction are
intentionally omitted (all off in Case B production), keeping the validated
Case B ``cloning.py`` untouched (CASE_A spec section 9).

Reuses the generic, backend-agnostic pieces from ``cloning.py``:
  * ``_systematic_resample_idxs``     -- systematic-resampling index draw
  * ``_batched_entanglement_entropy`` -- vectorised half-chain S over clones
  * ``CloningResult`` / ``CloningCollapse`` -- same container Case B returns, so
    the worker aggregation and B_L computation are shared verbatim.

The only Case-A-specific pieces are the model (``GaussianCaseAModel``), the
trajectory (``gaussian_born_rule_trajectory_caseA``), and ``delta_tau``, which is
set from the total monitored rate ``gamma*L + alpha*(L-1)`` rather than Case B's
``alpha*(L-1)``.

For ``zeta == 1`` there is no resampling and the weights are unity, so the N_c
"clones" are independent Born-rule trajectories: the Born-rule (no
post-selection) ensemble is exactly the zeta=1 special case of this function.
"""

from typing import Optional

import numpy as np

from .cloning import (
    CloningResult,
    CloningCollapse,
    _systematic_resample_idxs,
    _batched_entanglement_entropy,
)
from .gaussian_backend_caseA import (
    GaussianCaseAModel,
    gaussian_born_rule_trajectory_caseA,
)


def run_cloning_caseA(
    model: GaussianCaseAModel,
    zeta: float,
    T_total: float,
    N_c: int,
    rng: np.random.Generator,
    delta_tau: Optional[float] = None,
    n_burnin_frac: float = 0.25,
    record_entropy: bool = True,
    entropy_stride: int = 1,
    proposal_c: Optional[float] = None,
) -> CloningResult:
    """Population-dynamics estimator of theta(zeta) and S_zeta for Case A.

    Mirrors ``run_cloning`` (scalar path) field-for-field so the returned
    ``CloningResult`` is a drop-in for the Case B aggregation/B_L pipeline.
    """
    if zeta < 0.0:
        raise ValueError("zeta must be non-negative")
    if T_total <= 0.0:
        raise ValueError("T_total must be positive")
    if N_c < 1:
        raise ValueError("N_c must be >= 1")
    if proposal_c is not None and not (0.0 < proposal_c <= 1.0):
        raise ValueError("proposal_c must be in (0, 1]")

    L = model.L
    gamma_rate = model.gamma_rate
    alpha_rate = model.alpha_rate

    if delta_tau is None:
        total_rate = gamma_rate * L + alpha_rate * (L - 1)
        delta_tau = 1.0 / max(2.0 * total_rate, 1e-6)
    n_steps = max(1, int(np.ceil(T_total / delta_tau)))
    delta_tau_eff = T_total / n_steps

    covs = [model.gamma0.copy() for _ in range(N_c)]
    orbs = [model.orbitals0.copy() for _ in range(N_c)]
    sub_rngs = rng.spawn(N_c)

    log_Z_acc = 0.0
    log_Z_history: list[float] = []
    W_history: list[float] = []
    S_history: list[float] = []
    S_sq_history: list[float] = []
    n_T_mean_history: list[float] = []
    n_T_sq_history: list[float] = []
    covar_Sn_history: list[float] = []
    ess_history: list[float] = []
    ancestor_ids = np.arange(N_c, dtype=np.intp)
    n_collapses = 0
    final_weights = np.ones(N_c, dtype=np.float64)
    n_burnin_steps = int(n_steps * n_burnin_frac)

    for _k in range(n_steps):
        # --- Evolve all clones one cloning window (scalar) ---
        n_jumps = np.zeros(N_c, dtype=np.int64)
        delta_Lambda = np.zeros(N_c, dtype=np.float64)
        _pc = 1.0 if proposal_c is None else proposal_c
        for i in range(N_c):
            r = gaussian_born_rule_trajectory_caseA(
                model, T=delta_tau_eff, rng=sub_rngs[i],
                gamma0_override=covs[i],
                orbitals0_override=orbs[i],
                proposal_c=_pc,
            )
            covs[i] = r.final_covariance
            orbs[i] = r.final_orbitals
            n_jumps[i] = r.n_jumps
            delta_Lambda[i] = r.Lambda

        # --- Per-step weight W_k = mean_i(w_i), tilt zeta^{n_jumps} ---
        if zeta == 0.0:
            weights = (n_jumps == 0).astype(np.float64)
            log_W_k = float(np.log(weights.mean())) if weights.any() else -np.inf
        elif zeta == 1.0:
            weights = np.ones(N_c, dtype=np.float64)
            log_W_k = 0.0
        else:
            if proposal_c is not None:
                log_w = n_jumps * np.log(zeta / proposal_c) - (1.0 - proposal_c) * delta_Lambda
            else:
                log_w = n_jumps * np.log(zeta)
            lw_max = float(log_w.max())
            if not np.isfinite(lw_max):
                weights = np.zeros(N_c, dtype=np.float64)
                log_W_k = -np.inf
            else:
                w_rel = np.exp(log_w - lw_max)
                mean_rel = float(w_rel.mean())
                log_W_k = (np.log(mean_rel) + lw_max) if mean_rel > 0.0 else -np.inf
                weights = w_rel

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

        # --- Record weighted entropy + activity diagnostics ---
        if record_entropy and (_k % max(1, entropy_stride) == 0):
            S_vals = _batched_entanglement_entropy(covs, L // 2)
            w_sum = float(weights.sum())
            if w_sum > 0.0:
                w_norm = weights / w_sum
                S_zeta_k = float(np.dot(w_norm, S_vals))
                S_sq_k = float(np.dot(w_norm, S_vals ** 2))
                n_mean_k = float(np.dot(w_norm, n_jumps))
                n_sq_k = float(np.dot(w_norm, n_jumps ** 2))
                covar_k = float(np.dot(w_norm, S_vals * n_jumps))
            else:
                S_zeta_k = float(np.mean(S_vals))
                S_sq_k = float(np.mean(S_vals ** 2))
                n_mean_k = float(np.mean(n_jumps))
                n_sq_k = float(np.mean(n_jumps ** 2))
                covar_k = float(np.mean(S_vals * n_jumps))
            S_history.append(S_zeta_k)
            S_sq_history.append(S_sq_k)
            n_T_mean_history.append(n_mean_k)
            n_T_sq_history.append(n_sq_k)
            covar_Sn_history.append(covar_k)

        final_weights = weights.copy()

        # --- Per-step ESS (pre-resampling) ---
        w_sum_step = float(weights.sum())
        w_sq_step = float(np.sum(weights ** 2))
        ess_history.append((w_sum_step ** 2) / w_sq_step if w_sq_step > 0.0 else 0.0)

        # --- Systematic resampling (skip at zeta=1: independent trajectories) ---
        if zeta != 1.0:
            idxs = _systematic_resample_idxs(weights, rng)
            covs = [covs[int(i)].copy() for i in idxs]
            orbs = [orbs[int(i)].copy() for i in idxs]
            ancestor_ids = ancestor_ids[idxs]

    # --- Post-processing (mirror of run_cloning) ---
    theta_hat = log_Z_acc / T_total
    S_arr = np.asarray(S_history, dtype=np.float64) if record_entropy else np.asarray([])
    n_T_mean_val = chi_k_val = S_var_val = covar_Sk_val = float("nan")
    S_sq_arr = n_T_mean_arr = n_T_sq_arr = covar_Sn_arr = np.asarray([])

    _stride_eff = max(1, int(entropy_stride))
    n_burnin_recorded = int(np.ceil(n_burnin_steps / _stride_eff))

    if record_entropy and len(S_arr) > n_burnin_recorded:
        S_mean = float(np.mean(S_arr[n_burnin_recorded:]))
        S_std = float(np.std(S_arr[n_burnin_recorded:]))

        S_sq_arr = np.asarray(S_sq_history, dtype=np.float64)
        n_T_mean_arr = np.asarray(n_T_mean_history, dtype=np.float64)
        n_T_sq_arr = np.asarray(n_T_sq_history, dtype=np.float64)
        covar_Sn_arr = np.asarray(covar_Sn_history, dtype=np.float64)

        S_pb = S_arr[n_burnin_recorded:]
        Ssq_pb = S_sq_arr[n_burnin_recorded:]
        nm_pb = n_T_mean_arr[n_burnin_recorded:]
        nsq_pb = n_T_sq_arr[n_burnin_recorded:]
        cov_pb = covar_Sn_arr[n_burnin_recorded:]

        norm = float(L * delta_tau_eff)   # L * dtau : counts -> density
        n_T_mean_val = float(np.mean(nm_pb)) / norm
        chi_k_val = float(np.mean(nsq_pb - nm_pb ** 2)) / norm
        S_var_val = float(np.mean(Ssq_pb - S_pb ** 2))
        covar_Sk_val = float(np.mean(cov_pb - S_pb * nm_pb)) / norm
    else:
        S_mean = float("nan")
        S_std = float("nan")

    fw_sum = float(final_weights.sum())
    fw_sq = float(np.sum(final_weights ** 2))
    eff_sample_size = (fw_sum ** 2) / fw_sq if fw_sq > 0 else 0.0

    ess_arr = np.asarray(ess_history, dtype=np.float64)
    if ess_arr.size > n_burnin_steps:
        min_ess_frac_pb = float(ess_arr[n_burnin_steps:].min()) / float(N_c)
    else:
        min_ess_frac_pb = float("nan")
    n_distinct_ancestors = int(np.unique(ancestor_ids).size)

    return CloningResult(
        theta_hat=theta_hat,
        S_mean=S_mean, S_std=S_std,
        S_history=S_arr,
        log_Z_history=np.asarray(log_Z_history, dtype=np.float64),
        W_history=np.asarray(W_history, dtype=np.float64),
        n_collapses=n_collapses,
        n_burnin_steps=n_burnin_steps,
        eff_sample_size=eff_sample_size,
        zeta=float(zeta), L=int(L),
        # alpha/w are metadata only; for Case A we record the two rates here
        # (alpha = bond/Bogoliubov rate, w = site/density rate gamma).
        alpha=float(alpha_rate), w=float(gamma_rate),
        N_c=int(N_c), T_total=float(T_total), delta_tau=float(delta_tau_eff),
        final_covs=covs,
        n_T_mean=n_T_mean_val, chi_k=chi_k_val,
        S_var=S_var_val, covar_Sk=covar_Sk_val,
        n_T_mean_history=n_T_mean_arr, n_T_sq_history=n_T_sq_arr,
        S_sq_history=S_sq_arr, covar_Sn_history=covar_Sn_arr,
        ess_history=ess_arr,
        min_ess_frac_postburnin=min_ess_frac_pb,
        n_distinct_ancestors=n_distinct_ancestors,
    )
