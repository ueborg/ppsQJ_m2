"""Controlled (approximate-Doob) quantum-jump sampler for QJ-PPS.

Controlled rates          r_hat_j = zeta * r_j * exp(a * Delta_j K),   Delta_j K >= 0, a < 0
                       => tilt t_j = exp(a Delta_j K) in (0, 1]
                       => the controlled process is DOMINATED by the guided c=zeta
                          process, so M = 1 thinning off the exact branch-norm sampler:
                          propose a candidate at guided rate zeta*r, pick channel prop q_j,
                          ACCEPT the jump w.p. t_j, else continue un-jumped.

Exact residual weight (target pi_zeta = zeta^N P_Born, proposal Q_hat):
    log W_res = - a * sum_{accepted} Delta_j K  -  Lambda_T  +  zeta * I,
    I = int r(Gamma) <t>_q(Gamma) dt,     <t>_q = sum_j q_j t_j / sum_j q_j.
Lambda_T comes EXACTLY from the branch norms.  Only I is quadratic-rule: it is
accumulated as sum_intervals dLambda_i * (<t>_i + <t>_{i+1})/2 with dLambda_i exact,
and `dt_max` forces extra evaluation points so the quadrature can be refined and gated.

a = 0 reduces analytically to the production scheme: t_j == 1, I == Lambda_T,
log W_res = -(1-zeta) Lambda_T.
"""
import os, sys
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from pps_qj.gaussian_backend import (build_gaussian_chain_model,
                                     covariance_from_orbitals,
                                     _lowrank_jump_orbital_update,
                                     orbitals_from_covariance,
                                     _solve_waiting_time_newton,
                                     entanglement_entropy,
                                     topological_entanglement_entropy)


def tilt_factors(cov, ja, jb, a):
    """t_j = exp(a * [K(J_j Gamma) - K(Gamma)]) for every channel, O(L^2)."""
    sig = cov[ja, jb]
    q = np.clip(0.5 * (1.0 - sig), 0.0, 1.0)
    K = q.sum()
    denom = 1.0 - sig
    A = cov[np.ix_(ja, ja)]; B = cov[np.ix_(jb, jb)]
    C = cov[np.ix_(ja, jb)]; D = cov[np.ix_(jb, ja)]
    sig_new = sig[:, None] + (A * B - C * D) / denom[None, :]
    n = len(ja); di = np.arange(n)
    sig_new[di, di] = -1.0
    q_new = np.clip(0.5 * (1.0 - sig_new), 0.0, 1.0)
    dK = q_new.sum(axis=0) - K
    return q, dK, np.exp(a * dK)


def controlled_trajectory(model, T, rng, a, zeta, ja, jb,
                          gamma0=None, orbitals0=None, dt_max=np.inf,
                          refresh_every=100, eps_hazard=1e-9):
    evals, V, V_inv, VhV = (model.h_eff_evals, model.h_eff_V,
                            model.h_eff_V_inv, model.h_eff_VhV)
    nm = len(model.jump_pairs)
    K_gen = model.h_effective
    cov = (np.asarray(model.gamma0, float) if gamma0 is None else gamma0).copy()
    orb = (np.asarray(model.orbitals0, complex) if orbitals0 is None
           else orbitals0).copy()
    inv_c = 1.0 / zeta
    t = 0.0
    Lam = 0.0            # exact integrated PHYSICAL hazard
    I = 0.0              # int r <t>_q dt   (trapezoid in <t>_q weighted by exact dLambda)
    sum_dK = 0.0         # sum over ACCEPTED jumps of Delta_j K
    n_acc = n_cand = 0
    nref = 0

    q, dK, tl = tilt_factors(cov, ja, jb, a)
    tbar_prev = float((q * tl).sum() / max(q.sum(), 1e-300))

    def branch_norm(dt, coeffs):
        if dt <= 0.0:
            return 1.0, None, None
        A = np.exp(evals * dt)[:, None] * coeffs
        gram = A.conj().T @ (VhV @ A)
        try:
            Lc = np.linalg.cholesky(gram)
            lh = float(np.sum(np.log(np.abs(np.diag(Lc)))))
            return float(np.exp(lh - model.alpha * nm * dt)), A, Lc
        except np.linalg.LinAlgError:
            s, ld = np.linalg.slogdet(gram)
            if s <= 0 or not np.isfinite(ld):
                return 0.0, None, None
            return float(np.exp(0.5 * ld - model.alpha * nm * dt)), None, None

    while t < T:
        T_rem = min(T - t, dt_max)
        coeffs = V_inv @ orb
        U_eff = max(float(rng.uniform()) ** inv_c, 1e-300)
        bn, A_, Lc_ = branch_norm(T_rem, coeffs)
        if bn >= U_eff:
            # no candidate before T_rem: advance to T_rem un-jumped
            dLam = -np.log(max(bn, 1e-300))
            if Lc_ is not None:
                from scipy.linalg import solve_triangular
                Y = V @ A_
                orb = solve_triangular(Lc_.conj(), Y.T, lower=True).T
            else:
                orb, _ = np.linalg.qr(V @ (np.exp(evals * T_rem)[:, None] * coeffs),
                                      mode="reduced")
            cov = covariance_from_orbitals(orb)
            t += T_rem
            q, dK, tl = tilt_factors(cov, ja, jb, a)
            tbar = float((q * tl).sum() / max(q.sum(), 1e-300))
            Lam += dLam
            I += dLam * 0.5 * (tbar_prev + tbar)
            tbar_prev = tbar
            continue

        dt_star, orb = _solve_waiting_time_newton(
            coeffs, U_eff, T_rem, evals, V, VhV, K_gen,
            model.alpha, nm, eps_hazard)
        cov = covariance_from_orbitals(orb)
        t += dt_star
        dLam = -np.log(U_eff)
        q, dK, tl = tilt_factors(cov, ja, jb, a)
        tbar = float((q * tl).sum() / max(q.sum(), 1e-300))
        Lam += dLam
        I += dLam * 0.5 * (tbar_prev + tbar)
        tbar_prev = tbar

        tot = q.sum()
        if tot < 1e-15:
            break
        ch = int(rng.choice(nm, p=q / tot))
        n_cand += 1
        if float(rng.uniform()) < tl[ch]:            # M = 1 thinning accept
            sum_dK += dK[ch]
            n_acc += 1
            orb, cov = _lowrank_jump_orbital_update(orb, cov, model.jump_pairs[ch])
            nref += 1
            if refresh_every and nref >= refresh_every:
                orb = orbitals_from_covariance(cov); nref = 0
            q, dK, tl = tilt_factors(cov, ja, jb, a)
            tbar_prev = float((q * tl).sum() / max(q.sum(), 1e-300))

    log_w = -a * sum_dK - Lam + zeta * I
    return dict(cov=cov, orb=orb, Lambda=Lam, I=I, sum_dK=sum_dK,
                n_acc=n_acc, n_cand=n_cand, log_w=log_w)


def observables(cov):
    L = cov.shape[0] // 2
    return entanglement_entropy(cov, L // 2), topological_entanglement_entropy(cov)
