#!/usr/bin/env python3
"""
delta_B_zeta0.py -- GATE 1 (a)+(b): measure Delta_B at the REAL zeta=0 no-click
anchor and confirm the SSH reduction, using the real Gaussian backend.

Closes the two pending gate-1 items (HANDOFF):
  (a) Delta_B from the real no-click Majorana covariance (ties to measured 1.009),
  (b) the reduction check -- the real Case B no-click state has the Fermi-step
      critical structure, evidenced by the r^{-2} bond correlator (Delta_B~1)
      together with the odd-r null (two-chain decoupling).

At zeta=0 the no-click state is a SINGLE deterministic Gaussian state (no ensemble),
so Delta_B comes from the connected SINGLE-STATE bond correlator (= worker_opdim's cq):

    <B_x B_y>_c = Gamma[2x,2y+3] Gamma[2x+3,2y] - Gamma[2x,2y] Gamma[2x+3,2y+3]

fit to r^{-2 Delta_B} on EVEN r (odd r are exact zeros by the two-chain decoupling).

Conventions (from worker_zeta0_pps / gaussian_backend): alpha=lambda, w=1-lambda;
Gamma_{ab}=(i/2)<[g_a,g_b]>, b[x]=Gamma[2x,2x+3].  Critical segment 0<lambda<0.8.

Run on the Mac/Habrok (needs the backend). Deterministic, O(L^3) per (L,lam); seconds.
  python analysis/delta_B_zeta0.py
"""
import numpy as np
from scipy.linalg import expm
from pps_qj.gaussian_backend import build_gaussian_chain_model, covariance_from_orbitals


def steady_cov(L, lam, T_mult=15.0, dt=1.0):
    """No-click steady-state Majorana covariance (replicates the worker_zeta0 loop)."""
    w, alpha = 1.0 - lam, lam
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    T = max(T_mult / max(alpha, 1e-6), 3.0 * L)        # selection rate ~kappa=alpha/4; ballistic L
    n_steps = max(1, int(round(T / dt)))
    M = expm(model.h_effective * (T / n_steps))
    orb = model.orbitals0.copy()
    for _ in range(n_steps):
        orb = M @ orb
        Q, _ = np.linalg.qr(orb, mode="reduced")
        orb = Q
    return covariance_from_orbitals(orb)


def cq_connected(G, L, bulk_frac=0.25):
    """Connected single-state bond correlator <B_x B_{x+r}>_c, bulk- and x-averaged.
    Identical Wick form to worker_opdim's cq."""
    G = np.asarray(G, dtype=np.float64)
    w0 = int(round(bulk_frac * L))
    b_lo, b_hi = w0, (L - 1) - w0
    r_max = max((b_hi - b_lo) - 1, 0)
    rs = np.arange(1, r_max + 1)
    cq = np.full(r_max, np.nan)
    for k, r in enumerate(rs):
        x = np.arange(b_lo, b_hi - r)
        if x.size == 0:
            continue
        y = x + r
        a, b, c, d = 2 * x, 2 * x + 3, 2 * y, 2 * y + 3
        cq[k] = float(np.mean(G[a, d] * G[b, c] - G[a, c] * G[b, d]))
    return rs, cq


def fit_delta_B(rs, cq):
    """Delta_B from cq(r) ~ r^{-2 Delta_B} on EVEN r. Returns (Delta_B, R2, odd_null).
    odd_null = rms(cq on odd r)/rms(cq on even r): ~0 if the two-chain decoupling holds."""
    even = rs % 2 == 0
    odd = ~even
    rms = lambda v: float(np.sqrt(np.nanmean(v ** 2))) if np.isfinite(v).any() else np.nan
    odd_null = (rms(cq[odd]) / rms(cq[even])) if rms(cq[even]) else np.nan
    re, ce = rs[even], np.abs(cq[even])              # power-law envelope (sign-robust)
    good = np.isfinite(ce) & (ce > 0)
    re, ce = re[good], ce[good]
    if re.size < 4:
        return np.nan, np.nan, odd_null
    coef = np.polyfit(np.log(re), np.log(ce), 1)
    yhat = np.polyval(coef, np.log(re))
    ss = np.sum((np.log(ce) - yhat) ** 2)
    st = np.sum((np.log(ce) - np.log(ce).mean()) ** 2)
    R2 = 1 - ss / st if st > 0 else np.nan
    return (-coef[0]) / 2.0, float(R2), float(odd_null)


def main():
    print("=" * 66)
    print("GATE 1 (a)+(b): Delta_B at the real zeta=0 no-click anchor")
    print("  expect Delta_B ~ 1.0 (measured 1.009); odd-r null ~ 0 (decoupling);")
    print("  Delta_B~1 across 0<lam<0.8 confirms the SSH Fermi-step reduction.")
    print("=" * 66)
    for L in [128, 256]:
        print(f"\n--- L = {L} ---  ({'lam':>5} {'Delta_B':>9} {'R2':>7} {'odd_null':>9})")
        for lam in [0.20, 0.30, 0.40]:
            G = steady_cov(L, lam)
            rs, cq = cq_connected(G, L)
            dB, R2, odd = fit_delta_B(rs, cq)
            print(f"               {lam:>5.2f} {dB:>9.3f} {R2:>7.3f} {odd:>9.3f}")


if __name__ == "__main__":
    main()
