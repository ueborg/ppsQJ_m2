#!/usr/bin/env python3
"""chi2_response.py -- chi_2(L) = d^2 <I_CMI>/d zeta^2 |_{zeta=0} via the exact
PPS click-sector expansion (0/1/2 clicks). NOT cloning: fixed-click-number
response sampling, reusing the validated Gaussian primitives.

    chi_2 = 2[ Ahat2 - Ahat1*Zhat1 + O0*Zhat1^2 - O0*Zhat2 ]
    Zhat_N = Z_N/Z_0 ; Ahat_N = A_N/Z_0 ; O0 = A_0/Z_0 = no-click CMI.

Endpoint scaling: alpha=lambda=u/sqrt(L), w=1-alpha, T=c*L.
Diagnostic: chi_2 ~ L (x_J=1)  vs  chi_2 ~ L^2 (x_J=1/2). Plot chi_2/L vs L.

Deterministic quadrature (Option A), for small L validation. Reuses the
gaussian_backend cached-eig no-click propagator, apply_projective_jump, and
_batched_compute_B_L (same CMI tripartition as the boundary runs). L must be %4.
"""
import os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pps_qj.gaussian_backend import (build_gaussian_chain_model, covariance_from_orbitals,
                                     orbitals_from_covariance, apply_projective_jump)
from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L


def make_ctx(L, u, c):
    alpha = u / np.sqrt(L); w = 1.0 - alpha; T = c * float(L)
    m = build_gaussian_chain_model(L, w, alpha)
    return dict(L=L, alpha=alpha, T=T, evals=m.h_eff_evals, V=m.h_eff_V, Vi=m.h_eff_V_inv,
                jp=list(m.jump_pairs), orb0=np.asarray(m.orbitals0, np.complex128))


def prop_noclick(ctx, orb, dt):
    """No-click propagate orbitals by dt -> (orb_normalized, log_norm^2)."""
    if dt <= 0.0:
        return orb, 0.0
    Y = ctx["V"] @ (np.exp(ctx["evals"] * dt)[:, None] * (ctx["Vi"] @ orb))
    Q, R = np.linalg.qr(Y, mode="reduced")
    return Q, 2.0 * float(np.sum(np.log(np.abs(np.diag(R)))))


def click(ctx, orb, jbond):
    """Apply click on bond jbond -> (p_j, orb_after). p_j = 0.5(1-Gamma[a,b])."""
    pj, cov_new = apply_projective_jump(covariance_from_orbitals(orb), ctx["jp"][jbond])
    return pj, orbitals_from_covariance(cov_new)


def cmi(ctx, orb):
    return float(_batched_compute_B_L([covariance_from_orbitals(orb)], ctx["L"])["CMI"][0])


def chi2_deterministic(L, u=0.75, c=1.0, Nt=16, tau_min=0.0, jmin=0):
    ctx = make_ctx(L, u, c)
    T = ctx["T"]; Nb = len(ctx["jp"]); la = np.log(ctx["alpha"])
    dt = T / Nt; times = (np.arange(Nt) + 0.5) * dt

    orbT, logZ0 = prop_noclick(ctx, ctx["orb0"], T)   # no-click reference 0->T
    O0 = cmi(ctx, orbT)

    Zh1 = Ah1 = Zh2 = Ah2 = 0.0
    for m1 in range(Nt):
        t1 = times[m1]
        orb_t1, ln0 = prop_noclick(ctx, ctx["orb0"], t1)
        for j1 in range(Nb):
            pj1, oc1 = click(ctx, orb_t1, j1)
            if pj1 <= 1e-14:
                continue
            lp1 = ln0 + la + np.log(pj1)
            o1T, lnT = prop_noclick(ctx, oc1, T - t1)                 # 1-click
            w1 = np.exp(lp1 + lnT - logZ0) * dt
            Zh1 += w1; Ah1 += w1 * cmi(ctx, o1T)
            orun, lnrun, tprev = oc1, 0.0, t1                          # 2-click, share t1->t2
            for m2 in range(m1 + 1, Nt):
                t2 = times[m2]
                orun, lstep = prop_noclick(ctx, orun, t2 - tprev); lnrun += lstep; tprev = t2
                if abs(t2 - t1) < tau_min:
                    continue
                for j2 in range(Nb):
                    if abs(j2 - j1) < jmin:
                        continue
                    pj2, oc2 = click(ctx, orun, j2)
                    if pj2 <= 1e-14:
                        continue
                    o2T, lnT2 = prop_noclick(ctx, oc2, T - t2)
                    w2 = np.exp(lp1 + lnrun + la + np.log(pj2) + lnT2 - logZ0) * dt * dt
                    Zh2 += w2; Ah2 += w2 * cmi(ctx, o2T)

    chi2 = 2.0 * (Ah2 - Ah1 * Zh1 + O0 * Zh1**2 - O0 * Zh2)
    return dict(L=L, O0=O0, Zh1=Zh1, Ah1=Ah1, Zh2=Zh2, Ah2=Ah2, chi2=chi2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", type=int, nargs="+", default=[12, 16, 20])
    ap.add_argument("--u", type=float, default=0.75); ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--Nt", type=int, default=16)
    ap.add_argument("--tau-min", type=float, default=0.0); ap.add_argument("--jmin", type=int, default=0)
    a = ap.parse_args()
    print(f"# chi2 deterministic u={a.u} c={a.c} Nt={a.Nt} tau_min={a.tau_min} jmin={a.jmin}")
    print(f"# {'L':>4} {'chi2':>11} {'chi2/L':>9} {'chi2/L^2':>9} | {'O0':>7} {'Zh1':>7} {'Zh2':>7} {'Ah1':>7} {'Ah2':>7}   wall")
    for L in a.Ls:
        assert L % 4 == 0, "L must be divisible by 4 (CMI tripartition)"
        t0 = time.time(); r = chi2_deterministic(L, a.u, a.c, a.Nt, a.tau_min, a.jmin)
        print(f"  {L:4d} {r['chi2']:11.4f} {r['chi2']/L:9.4f} {r['chi2']/L**2:9.5f} | "
              f"{r['O0']:7.4f} {r['Zh1']:7.4f} {r['Zh2']:7.4f} {r['Ah1']:7.4f} {r['Ah2']:7.4f}  {time.time()-t0:5.1f}s", flush=True)
