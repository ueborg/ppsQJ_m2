#!/usr/bin/env python3
"""
exponent_noclick.py -- pin the no-click correlation-length exponent p in
xi_nc ~ lambda^-p for the Cut B (class DIII) Kitaev subchain at zeta=0.

Builds the REAL backend effective generator (measurement on the same bond as the
hopping: h_eff[measured bond] = w - i*alpha) and constructs the no-click
steady-state Majorana covariance via the O(L^3) dominant-eigenvector projection,
validated to machine precision (cov_diff ~1e-15) against the iterative orbital
loop of delta_B_zeta0.py / the gaussian backend.

Pins what the L<=256 deterministic container check could not: whether the
measured steady-state xi_nc approaches the analytic band-structure lambda^-2 in
the asymptotic small-lambda regime, or stays flatter. Deterministic, no cloning;
one eig of a 2L x 2L complex matrix per (L, lambda).
"""
import numpy as np
import time, json


def bond_jump_pair(bond):
    return 2 * bond, 2 * bond + 3


def covariance_from_orbitals(orb):
    n = orb.shape[0]
    g = 1j * (2.0 * (orb @ orb.conj().T) - np.eye(n))
    g = np.real_if_close(g, tol=1000.0)
    g = np.asarray(g.real)
    return 0.5 * (g - g.T)


def effective_generator(L, w, alpha):
    h = np.zeros((2 * L, 2 * L))
    for b in range(L - 1):
        a, bb = bond_jump_pair(b)
        c, d = 2 * b + 1, 2 * b + 2
        h[a, bb] = w;  h[bb, a] = -w
        h[c, d] = -w;  h[d, c] = w
    h = h.astype(np.complex128)
    for b in range(L - 1):
        a, bb = bond_jump_pair(b)
        h[a, bb] -= 1j * alpha
        h[bb, a] += 1j * alpha
    return h


def noclick_steady_cov(L, lam):
    """No-click steady-state covariance via dominant-eigenvector projection."""
    w, alpha = 1.0 - lam, lam
    h = effective_generator(L, w, alpha)
    mu, V = np.linalg.eig(h)
    idx = np.argsort(mu.real)[::-1][:L]      # L most-amplified right-eigenvectors
    orb = V[:, idx]
    orb, _ = np.linalg.qr(orb, mode="reduced")
    return covariance_from_orbitals(orb)


def xi_from_cov(G, L):
    """Single-particle xi in SSH unit cells: |G[i0, i0+4m]| decay (one sublattice)."""
    i0 = 2 * (L // 2)
    ms, gs = [], []
    for m in range(1, L // 4):
        j = i0 + 4 * m
        if j >= 2 * L - L // 8:
            break
        ms.append(m); gs.append(abs(G[i0, j]))
    ms, gs = np.array(ms), np.array(gs)
    good = gs > gs.max() * 1e-8
    ms, gs = ms[good], gs[good]
    band = gs < gs.max() * 0.6
    if band.sum() < 6:
        band = np.ones_like(gs, bool)
    c = np.polyfit(ms[band], np.log(gs[band]), 1)
    yh = np.polyval(c, ms[band])
    ss = np.sum((np.log(gs[band]) - yh) ** 2)
    st = np.sum((np.log(gs[band]) - np.log(gs[band]).mean()) ** 2)
    R2 = 1 - ss / st if st > 0 else np.nan
    return -1.0 / c[0], R2


def xi_analytic_cells(lam):
    w, k = 1.0 - lam, lam
    return 2.0 / np.log(1.0 + (k / w) ** 2)


L_MAX = 4096   # cap on L; raise (and the wall time) if budget allows smaller lambda


def choose_L(lam):
    xi_sites = 2.0 * xi_analytic_cells(lam)   # ~2 sites per SSH unit cell
    L = int(8 * xi_sites)
    L = max(640, min(L_MAX, L))
    return (L // 4) * 4                        # divisible by 4 for the sublattice sampler


def main():
    lambdas = [0.08, 0.10, 0.13, 0.17, 0.22, 0.28, 0.35]
    out = {"lambda": [], "L": [], "xi_num_cells": [],
           "xi_analytic_cells": [], "R2": []}
    print(f"{'lam':>6} {'L':>6} {'xi_num':>9} {'xi_an':>9} "
          f"{'L/xi_sites':>11} {'R2':>7} {'t[s]':>7}")
    for lam in lambdas:
        L = choose_L(lam)
        t0 = time.time()
        G = noclick_steady_cov(L, lam)
        xi, R2 = xi_from_cov(G, L)
        dt = time.time() - t0
        xa = xi_analytic_cells(lam)
        print(f"{lam:>6.3f} {L:>6d} {xi:>9.3f} {xa:>9.3f} "
              f"{L/(2.0*xa):>11.2f} {R2:>7.3f} {dt:>7.1f}")
        out["lambda"].append(lam); out["L"].append(L)
        out["xi_num_cells"].append(float(xi))
        out["xi_analytic_cells"].append(float(xa)); out["R2"].append(float(R2))

    lam = np.array(out["lambda"]); xi = np.array(out["xi_num_cells"])
    g = np.isfinite(xi) & (xi > 0)
    p_full = -np.polyfit(np.log(lam[g]), np.log(xi[g]), 1)[0]
    sm = g & (lam <= 0.17)
    p_small = -np.polyfit(np.log(lam[sm]), np.log(xi[sm]), 1)[0]
    p_an = -np.polyfit(np.log(lam[g]),
                       np.log(np.array(out["xi_analytic_cells"])[g]), 1)[0]
    print(f"\nmeasured  xi_nc ~ lambda^-p :  full p={p_full:.3f}   "
          f"small-lambda(<=0.17) p={p_small:.3f}")
    print(f"analytic band-structure xi  :  p={p_an:.3f}  (-> 2 as lambda->0)")
    out.update(p_full=float(p_full), p_small=float(p_small), p_analytic=float(p_an))
    with open("exponent_noclick_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote exponent_noclick_result.json")


if __name__ == "__main__":
    main()
