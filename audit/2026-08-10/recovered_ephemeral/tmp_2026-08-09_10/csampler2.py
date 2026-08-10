"""Controlled (approximate-Doob) QJ sampler, v2: Simpson compensator + time-dependent taper.

Controlled rates   r_hat_j(t) = zeta r_j exp(a(t) dK_j),  dK_j >= 0, a(t) <= 0
                => tilt <= 1 => DOMINATED by the guided c=zeta process => M=1 thinning
                   off the exact branch-norm/Newton sampler.

Weight, written so the a=0 branch is EXACT (no quadrature touches it):
    int r_hat dt = zeta*Lambda + zeta*J,     J = int r (<t>_q - 1) dt
    log W_res    = -sum_m a(t_m) dK_m - (1-zeta) Lambda + zeta J
J == 0 identically at a=0  =>  log W_res = -(1-zeta) Lambda, the production weight.
J is the SMALL correction (|<t>-1| ~ 0.2), integrated by Simpson in time.

`a` may be a float or a callable a(t).  Frozen at the window-start value inside a
call; the caller supplies the true a at window edges for the twisted increment
    dl_twist = dl_raw + a(u) K_u - a(t) K_t
which telescopes to log W_res + a(T)K_T - a(0)K_0.  With a(T)=0 there is NO
terminal untwisting.
"""
import os, sys
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from scipy.linalg import solve_triangular
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from pps_qj.gaussian_backend import (covariance_from_orbitals,
                                     _lowrank_jump_orbital_update,
                                     orbitals_from_covariance,
                                     _solve_waiting_time_newton,
                                     entanglement_entropy,
                                     topological_entanglement_entropy)


def tilt_factors(cov, ja, jb, a):
    sig = cov[ja, jb]
    q = np.clip(0.5 * (1.0 - sig), 0.0, 1.0)
    K = q.sum()
    if a == 0.0:
        return q, np.zeros(len(ja)), np.ones(len(ja)), K
    denom = 1.0 - sig
    A = cov[np.ix_(ja, ja)]; B = cov[np.ix_(jb, jb)]
    C = cov[np.ix_(ja, jb)]; D = cov[np.ix_(jb, ja)]
    sig_new = sig[:, None] + (A * B - C * D) / denom[None, :]
    n = len(ja); di = np.arange(n)
    sig_new[di, di] = -1.0
    dK = np.clip(0.5 * (1.0 - sig_new), 0.0, 1.0).sum(axis=0) - K
    return q, dK, np.exp(a * dK), K


def _f_corr(model, cov, ja, jb, a):
    """f = r * (<t>_q - 1);  zero at a=0."""
    q, _, tl, K = tilt_factors(cov, ja, jb, a)
    tot = q.sum()
    if tot < 1e-300:
        return 0.0, K
    return float(2.0 * model.alpha * ((q * tl).sum() - tot)), K


def controlled_trajectory(model, T, rng, a, zeta, ja, jb, gamma0=None,
                          orbitals0=None, refresh_every=100, eps_hazard=1e-9,
                          simpson=True, t0=0.0):
    a_val = float(a(t0)) if callable(a) else float(a)
    evals, V, V_inv, VhV = (model.h_eff_evals, model.h_eff_V,
                            model.h_eff_V_inv, model.h_eff_VhV)
    nm = len(model.jump_pairs); K_gen = model.h_effective
    cov = (np.asarray(model.gamma0, float) if gamma0 is None else gamma0).copy()
    orb = (np.asarray(model.orbitals0, complex) if orbitals0 is None else orbitals0).copy()
    inv_c = 1.0 / zeta
    t = 0.0; Lam = 0.0; J = 0.0; sum_adK = 0.0
    n_acc = n_cand = 0; nref = 0
    f_prev, K_start = _f_corr(model, cov, ja, jb, a_val)
    K_prev = K_start

    def bnorm(dt, coeffs):
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

    def normalise(A, Lc, dt, coeffs):
        if Lc is not None:
            return solve_triangular(Lc.conj(), (V @ A).T, lower=True).T
        return np.linalg.qr(V @ (np.exp(evals * dt)[:, None] * coeffs),
                            mode="reduced")[0]

    def add_J(dt, coeffs, f_end):
        """Simpson (or trapezoid) in TIME on the small correction f."""
        nonlocal J
        if a_val == 0.0 or dt <= 0.0:
            return
        if simpson:
            _, Am, Lm = bnorm(0.5 * dt, coeffs)
            om = normalise(Am, Lm, 0.5 * dt, coeffs)
            fm, _ = _f_corr(model, covariance_from_orbitals(om), ja, jb, a_val)
            J += dt * (f_prev + 4.0 * fm + f_end) / 6.0
        else:
            J += dt * 0.5 * (f_prev + f_end)

    while t < T:
        T_rem = T - t
        coeffs = V_inv @ orb
        U_eff = max(float(rng.uniform()) ** inv_c, 1e-300)
        bn, A_, Lc_ = bnorm(T_rem, coeffs)
        if bn >= U_eff:
            orb = normalise(A_, Lc_, T_rem, coeffs)
            cov = covariance_from_orbitals(orb)
            f_end, K_prev = _f_corr(model, cov, ja, jb, a_val)
            add_J(T_rem, coeffs, f_end)
            Lam += -np.log(max(bn, 1e-300))
            f_prev = f_end; t += T_rem
            continue
        dt_star, orb = _solve_waiting_time_newton(coeffs, U_eff, T_rem, evals, V,
                                                  VhV, K_gen, model.alpha, nm,
                                                  eps_hazard)
        cov = covariance_from_orbitals(orb)
        q, dK, tl, K_prev = tilt_factors(cov, ja, jb, a_val)
        f_end = (0.0 if a_val == 0.0 else
                 float(2.0 * model.alpha * ((q * tl).sum() - q.sum())))
        add_J(dt_star, coeffs, f_end)
        Lam += -np.log(U_eff); t += dt_star; f_prev = f_end
        tot = q.sum()
        if tot < 1e-15:
            break
        ch = int(rng.choice(nm, p=q / tot)); n_cand += 1
        if a_val == 0.0 or float(rng.uniform()) < tl[ch]:
            sum_adK += a_val * dK[ch]; n_acc += 1
            orb, cov = _lowrank_jump_orbital_update(orb, cov, model.jump_pairs[ch])
            nref += 1
            if refresh_every and nref >= refresh_every:
                orb = orbitals_from_covariance(cov); nref = 0
            f_prev, K_prev = _f_corr(model, cov, ja, jb, a_val)

    log_w = -sum_adK - (1.0 - zeta) * Lam + zeta * J
    return dict(cov=cov, orb=orb, Lambda=Lam, J=J, log_w=log_w, a_used=a_val,
                K_start=K_start, K_end=K_prev, n_acc=n_acc, n_cand=n_cand)


def observables(cov):
    L = cov.shape[0] // 2
    return entanglement_entropy(cov, L // 2), topological_entanglement_entropy(cov)
