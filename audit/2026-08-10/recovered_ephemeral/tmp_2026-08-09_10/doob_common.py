"""Shared trajectory sampler for the ppsQJ_m2 algorithm pilots."""
import os, sys
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from pps_qj.gaussian_backend import (build_gaussian_chain_model,
                                     gaussian_born_rule_trajectory,
                                     entanglement_entropy)

KW = dict(jump_update_method="lowrank", solver_method="newton", bisection_tol=1e-6)


def build(L, lam):
    """alpha = lam, w = 1 - lam  (grid_pps._alpha_w_from_lam)."""
    model = build_gaussian_chain_model(L=L, w=1.0 - lam, alpha=lam)
    jp = model.jump_pairs
    ja = np.array([p[0] for p in jp], dtype=np.intp)
    jb = np.array([p[1] for p in jp], dtype=np.intp)
    return model, ja, jb


def features(model, cov, ja, jb, L):
    """q_j = (1 - Gamma[ja,jb])/2 ; r = 2 alpha sum_j q_j."""
    q = np.clip(0.5 * (1.0 - cov[ja, jb]), 0.0, 1.0)
    K = q.sum()
    f = np.array([K, (q * q).sum(), (q[:-1] * q[1:]).sum(),
                  (q[:-2] * q[2:]).sum(), entanglement_entropy(cov, L // 2)])
    return 2.0 * model.alpha * K, f


def run_one(L, lam, zeta, T, delta, seed):
    model, ja, jb = build(L, lam)
    rng = np.random.default_rng(seed)
    cov = np.asarray(model.gamma0, dtype=np.float64).copy()
    orb = np.asarray(model.orbitals0, dtype=np.complex128).copy()
    n = int(round(T / delta))
    r_s = np.empty(n); X_s = np.empty((n, 5)); dL = np.empty(n); nj = 0
    for k in range(n):
        r_s[k], X_s[k] = features(model, cov, ja, jb, L)
        res = gaussian_born_rule_trajectory(
            model, delta, rng, gamma0_override=cov, orbitals0_override=orb,
            ja_cached=ja, jb_cached=jb, proposal_c=zeta, **KW)
        cov, orb = res.final_covariance, res.final_orbitals
        dL[k] = res.Lambda; nj += res.n_jumps
    return r_s, X_s, dL, nj
