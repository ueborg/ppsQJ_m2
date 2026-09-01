"""Instrumented guided-cloning run: full per-window genealogy + weight diagnostics.

This is a DIAGNOSTIC RE-IMPLEMENTATION of the inner loop of
pps_qj.cloning.run_cloning, restricted to the certified production configuration
(guided proposal_c = zeta, exact RN compensator, systematic resampling,
fixed population).  It calls the SAME primitives:

    gaussian_born_rule_trajectory   (mutation)
    _systematic_resample_idxs       (selection)

so the sampler is unchanged.  What is added is bookkeeping the production path
does not keep: the full ancestor matrix over time, per-window normalised
weights, and optional per-window observables.

It also supports the interventions the diagnosis needs:
    resample_mode = 'every'   production: resample at every window
                  = 'never'   pure sequential importance sampling (no selection)
                  = 'ess'     adaptive: resample only when ESS/N_c < ess_threshold
                  = 'every_k' resample every k-th window

Exactness note: 'never' and 'ess' carry the accumulated importance weights, so
the weighted estimator targets the SAME measure P_zeta.  'every' is the
production path.  No mode changes the target.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from pps_qj.gaussian_backend import (
    build_gaussian_chain_model, gaussian_born_rule_trajectory,
)
from pps_qj.cloning import _systematic_resample_idxs
from pps_qj import gaussian_backend as _gb


# --- Counter for the unlogged brentq fallback -------------------------------
# pps_qj/gaussian_backend.py:537-538 sets dt_star = 0.5*T_rem on a brentq
# ValueError and still credits the weight with -log(U_eff), i.e. the exact
# hazard at a root that was NOT found. It is presumably rare, but it is an
# uncontrolled approximation to the target measure and nothing counts it.
# We count it here NON-INVASIVELY (production code is untouched) by wrapping
# the brentq symbol that gaussian_backend resolved at import time.
_BRENTQ_FALLBACKS = [0]
_orig_brentq = _gb.brentq


def _counting_brentq(*a, **kw):
    try:
        return _orig_brentq(*a, **kw)
    except ValueError:
        _BRENTQ_FALLBACKS[0] += 1
        raise


_gb.brentq = _counting_brentq


def _multinomial_resample_idxs(weights, rng):
    """Multinomial selection, the scheme every clean SMC theorem assumes.

    Present ONLY as a controlled comparison arm. Systematic resampling is and
    remains the production scheme (docs/PRODUCTION_ALGORITHM.md section 3).
    Both are unbiased (E[count_i] = N*w_i); they differ in the offspring
    variance, and that difference is what the chunking-invariance arm tests.
    """
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Cannot resample: all weights zero.")
    N = w.size
    return rng.choice(N, size=N, replace=True, p=w / total).astype(np.intp)
from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L


def alpha_w_from_lam(lam):
    return float(lam), float(1.0 - lam)


@dataclass
class InstrumentedResult:
    L: int; zeta: float; lam: float; N_c: int; T: float
    delta_tau: float; n_steps: int; resample_mode: str; seed: int
    resample_scheme: str
    brentq_fallbacks: int
    # per-window [n_steps]
    ess: np.ndarray                  # instantaneous ESS from window weights
    ess_cum: np.ndarray              # ESS of cumulative (carried) weights
    logw_var: np.ndarray             # Var(log w) of the window weights
    w_max: np.ndarray                # largest normalised window weight
    n_jumps_mean: np.ndarray
    dLambda_mean: np.ndarray
    dLambda_var: np.ndarray
    resampled: np.ndarray            # bool, did a selection happen this window
    n_distinct_anc: np.ndarray       # distinct founders alive, per window
    gess: np.ndarray                 # genealogical ESS w.r.t. founders
    max_family_frac: np.ndarray
    # ancestor matrix [n_steps, N_c]: founder index carried by each slot
    anc_matrix: np.ndarray
    # per-window resampling index map (empty rows where no selection happened);
    # lets pairwise most-recent-common-ancestor times be reconstructed exactly.
    idxs_history: list
    # final population
    final_weights: np.ndarray        # normalised cumulative weights at t=T
    obs: dict                        # per-clone observables at t=T
    n_resampling_events: int
    wall_s: float


def run_instrumented(L, zeta, lam, N_c, T, seed,
                     dtau_mult=6.0, n_burnin_frac=0.25,
                     resample_mode='every', ess_threshold=0.5, every_k=1,
                     proposal_c='zeta', record_anc=True,
                     jump_update_method='lowrank', refresh_every=100,
                     resample_scheme='systematic'):
    # NOTE: dtau_mult defaults to the CERTIFIED production value 6, not the
    # corpus value 12. GENCOL's copy of this file defaulted to 12; a default
    # that silently deviates from the certified baseline is a trap.
    import time
    t0 = time.time()
    _fb0 = _BRENTQ_FALLBACKS[0]
    if resample_scheme not in ('systematic', 'multinomial'):
        raise ValueError("resample_scheme must be 'systematic' or 'multinomial'")
    _select = (_systematic_resample_idxs if resample_scheme == 'systematic'
               else _multinomial_resample_idxs)
    alpha, w = alpha_w_from_lam(lam)
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    delta_tau = dtau_mult / max(2.0 * alpha * (L - 1), 1e-6)
    n_steps = max(1, int(np.ceil(T / delta_tau)))
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    sub_rngs = list(rng.spawn(N_c))

    pc = float(zeta) if proposal_c == 'zeta' else (
        1.0 if proposal_c is None else float(proposal_c))

    covs = [model.gamma0.copy() for _ in range(N_c)]
    orbs = [model.orbitals0.copy() for _ in range(N_c)]
    ja = np.array([p[0] for p in model.jump_pairs], dtype=np.intp)
    jb = np.array([p[1] for p in model.jump_pairs], dtype=np.intp)

    anc = np.arange(N_c, dtype=np.intp)
    log_carry = np.zeros(N_c)          # accumulated log importance weight
    H = dict(ess=[], ess_cum=[], logw_var=[], w_max=[], n_jumps_mean=[],
             dLambda_mean=[], dLambda_var=[], resampled=[],
             n_distinct_anc=[], gess=[], max_family_frac=[])
    anc_rows = []
    idxs_hist = []
    n_resamp = 0

    for k in range(n_steps):
        n_jumps = np.zeros(N_c, dtype=np.int64)
        dLam = np.zeros(N_c)
        for i in range(N_c):
            r = gaussian_born_rule_trajectory(
                model, T=dt, rng=sub_rngs[i],
                gamma0_override=covs[i], orbitals0_override=orbs[i],
                ja_cached=ja, jb_cached=jb, proposal_c=pc,
                jump_update_method=jump_update_method,
                refresh_every=refresh_every,
                solver_method='brentq', eps_hazard=1e-9)
            covs[i] = r.final_covariance; orbs[i] = r.final_orbitals
            n_jumps[i] = r.n_jumps; dLam[i] = r.Lambda

        if zeta == 1.0:
            log_w = np.zeros(N_c)
        else:
            log_w = n_jumps * np.log(zeta / pc) - (1.0 - pc) * dLam

        lw = log_w - log_w.max()
        wr = np.exp(lw); wn = wr / wr.sum()
        H['ess'].append(1.0 / np.sum(wn ** 2))
        H['logw_var'].append(float(np.var(log_w)))
        H['w_max'].append(float(wn.max()))
        H['n_jumps_mean'].append(float(n_jumps.mean()))
        H['dLambda_mean'].append(float(dLam.mean()))
        H['dLambda_var'].append(float(np.var(dLam)))

        log_carry = log_carry + log_w
        lc = log_carry - log_carry.max()
        wc = np.exp(lc); wc = wc / wc.sum()
        H['ess_cum'].append(1.0 / np.sum(wc ** 2))

        # --- selection ---
        do_resample = (
            zeta != 1.0 and (
                (resample_mode == 'every') or
                (resample_mode == 'every_k' and (k % every_k) == (every_k - 1)) or
                (resample_mode == 'ess' and (1.0 / np.sum(wc ** 2)) < ess_threshold * N_c)
            ))
        if do_resample:
            n_resamp += 1
            wsel = wc if resample_mode in ('ess', 'every_k') else wn
            idxs = _select(wsel, rng)
            covs = [covs[int(i)].copy() for i in idxs]
            orbs = [orbs[int(i)].copy() for i in idxs]
            anc = anc[idxs]
            idxs_hist.append(idxs.copy())
            log_carry = np.zeros(N_c)      # weights reset after selection
        else:
            idxs_hist.append(None)
        H['resampled'].append(bool(do_resample))

        cnt = np.bincount(anc, minlength=N_c).astype(np.float64)
        H['n_distinct_anc'].append(int((cnt > 0).sum()))
        H['gess'].append(float(cnt.sum() ** 2 / np.sum(cnt ** 2)))
        H['max_family_frac'].append(float(cnt.max() / N_c))
        if record_anc:
            anc_rows.append(anc.copy())

    lc = log_carry - log_carry.max(); wfin = np.exp(lc); wfin /= wfin.sum()
    obs = _batched_compute_B_L(covs, L)
    return InstrumentedResult(
        L=L, zeta=zeta, lam=lam, N_c=N_c, T=T, delta_tau=dt, n_steps=n_steps,
        resample_mode=resample_mode, seed=seed,
        resample_scheme=resample_scheme,
        brentq_fallbacks=int(_BRENTQ_FALLBACKS[0] - _fb0),
        ess=np.array(H['ess']), ess_cum=np.array(H['ess_cum']),
        logw_var=np.array(H['logw_var']), w_max=np.array(H['w_max']),
        n_jumps_mean=np.array(H['n_jumps_mean']),
        dLambda_mean=np.array(H['dLambda_mean']),
        dLambda_var=np.array(H['dLambda_var']),
        resampled=np.array(H['resampled']),
        n_distinct_anc=np.array(H['n_distinct_anc']),
        gess=np.array(H['gess']),
        max_family_frac=np.array(H['max_family_frac']),
        anc_matrix=np.array(anc_rows) if record_anc else np.array([]),
        idxs_history=idxs_hist,
        final_weights=wfin, obs=obs, n_resampling_events=n_resamp,
        wall_s=time.time() - t0)


def pairwise_mrca(idxs_history, N_c):
    """Exact pairwise most-recent-common-ancestor time for the FINAL population.

    Returns an (N_c, N_c) integer matrix M where M[i,j] is the number of windows
    back from t=T at which final slots i and j last shared a lineage.  M[i,j] =
    n_windows_walked means "had not coalesced within the recorded history".

    Walks the selection maps backwards: if slot i at window k came from slot
    idxs[i] at window k-1, then lineage labels propagate by label = idxs[label].
    """
    lab = np.arange(N_c, dtype=np.intp)
    M = np.full((N_c, N_c), -1, dtype=np.int64)
    depth = 0
    for k in range(len(idxs_history) - 1, -1, -1):
        idxs = idxs_history[k]
        if idxs is None:
            continue
        depth += 1
        lab = idxs[lab]                      # lineage label one window earlier
        same = (lab[:, None] == lab[None, :])
        newly = same & (M < 0)
        M[newly] = depth
    M[M < 0] = depth + 1                     # not coalesced within history
    np.fill_diagonal(M, 0)
    return M, depth


def weighted_stats(vals, w):
    """Self-normalised weighted mean and the weighted variance across clones."""
    vals = np.asarray(vals, float); w = np.asarray(w, float)
    ok = np.isfinite(vals)
    if not ok.any():
        return np.nan, np.nan
    v = vals[ok]; ww = w[ok]; ww = ww / ww.sum()
    m = float(np.dot(ww, v))
    var = float(np.dot(ww, (v - m) ** 2))
    return m, var
