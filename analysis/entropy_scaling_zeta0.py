#!/usr/bin/env python3
"""
entropy_scaling_zeta0.py -- GATE 1, CORRECTED PROBE: is the zeta=0 no-click anchor
CRITICAL, and if so what is its central charge c?

Why this and not bond correlators: the measured bonds B_x=i g_{2x} g_{2x+3} all
mutually COMMUTE, so the no-click dynamics freezes them to definite values and the
single-state connected bond correlator vanishes identically (cq ~ 1e-34). Criticality
must therefore be read from the ENTANGLEMENT ENTROPY scaling of the steady state, not
from <B_x B_y>_c (which is structurally zero) or the raw two-point (sublattice-mixed).

Calabrese-Cardy, open boundary:  S(ell) = (c/6) ln[ (L/pi) sin(pi ell/L) ] + const.
Fit S vs the chord variable; slope*6 = c.
  c ~ 1     -> critical, free Dirac / SU(2)_1
  c ~ 0.5   -> critical Ising  (the Case-A relocation target)
  c ~ 0     -> GAPPED (S saturates; the anchor is NOT critical)

Block entropy from the Majorana covariance (same formula as worker_clone's _batch_entropy),
in NATS. Run on Mac/Habrok (needs the backend). Deterministic, O(L^3); seconds.
  python analysis/entropy_scaling_zeta0.py
"""
import numpy as np
from delta_B_zeta0 import steady_cov


def block_entropy(G, ell_sites):
    """von Neumann entropy (nats) of the first ell_sites sites = Majorana [0,2 ell)."""
    GA = G[:2 * ell_sites, :2 * ell_sites]
    eig = np.linalg.eigvalsh(1j * GA.astype(np.complex128))
    nu = np.clip(np.abs(eig[ell_sites:]), 0.0, 1.0)         # the +nu_k half
    p = np.clip(0.5 * (1.0 + nu), 1e-15, 1 - 1e-15)
    q = np.clip(0.5 * (1.0 - nu), 1e-15, 1 - 1e-15)
    return float(-np.sum(p * np.log(p) + q * np.log(q)))


def fit_c(L, S_of_ell, ells):
    """Fit S = (c/6) ln[(L/pi) sin(pi ell/L)] + const -> c, R2."""
    chord = (L / np.pi) * np.sin(np.pi * ells / L)
    x = np.log(chord)
    good = np.isfinite(x) & np.isfinite(S_of_ell)
    x, y = x[good], S_of_ell[good]
    if x.size < 4:
        return np.nan, np.nan
    coef = np.polyfit(x, y, 1)
    c = 6.0 * coef[0]
    yhat = np.polyval(coef, x)
    R2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(c), float(R2)


def main():
    print("=" * 64)
    print("GATE 1 (corrected): central charge of the zeta=0 no-click anchor")
    print("  c~1 critical(Dirac/SU(2)_1) | c~0.5 Ising | c~0 GAPPED")
    print("=" * 64)
    for L in [128, 256]:
        print(f"\n--- L={L} ---")
        # use the central window of subsystem sizes (avoid boundary ell)
        ells = np.arange(L // 8, L // 2 + 1, max(1, L // 32))
        for lam in [0.20, 0.30, 0.40, 0.60]:
            G = steady_cov(L, lam)
            S = np.array([block_entropy(G, int(e)) for e in ells])
            c, R2 = fit_c(L, S, ells)
            sat = S.max() - S.min()
            print(f"  lam={lam:.2f}: c={c:.3f}  R2={R2:.3f}  (S range over window={sat:.3f}"
                  f"{'  <- flat => GAPPED' if sat < 0.1 else ''})")


if __name__ == "__main__":
    main()
