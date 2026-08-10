"""Gates for the Galerkin Doob result.

A) Monte-Carlo check of G phi at a FIXED state: (E[phi(Gamma_h)] - phi(Gamma))/h
   under the ORIGINAL (Born, c=1) process, vs the analytic G phi.
B) dt_nc sensitivity of the no-click drift finite difference.
C) is <G phi>_ss really nonzero, or is the sampled ensemble non-stationary?
"""
import os, sys
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from doob_common import build, KW
from doob_galerkin import generator_phi, phi_from_sigma
from pps_qj.gaussian_backend import gaussian_born_rule_trajectory

L = int(os.environ.get("L", 32))
zeta = float(os.environ.get("ZETA", 0.9))
lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
model, ja, jb = build(L, lam)

# equilibrate one state under the guided process
rng = np.random.default_rng(11)
res = gaussian_born_rule_trajectory(model, 60.0, rng, proposal_c=zeta, **KW)
cov0, orb0 = res.final_covariance, res.final_orbitals
phi0 = phi_from_sigma(cov0[ja, jb])
print("phi(Gamma0) =", np.round(phi0, 4))

print("\n--- B) dt_nc sensitivity of analytic G phi ---")
for dt_nc in (2e-2, 5e-3, 2e-3, 5e-4, 1e-4):
    _, g, _ = generator_phi(model, cov0, orb0, ja, jb, dt_nc=dt_nc)
    print(f"  dt_nc={dt_nc:8.1e}   G phi = {np.round(g, 4)}")

print("\n--- A) Monte-Carlo G phi (Born process, c=1) ---")
_, g_ana, _ = generator_phi(model, cov0, orb0, ja, jb, dt_nc=1e-4)
for h in (0.02, 0.05):
    M = int(os.environ.get("MMC", 4000))
    acc = np.zeros(3)
    r2 = np.zeros(3)
    for m in range(M):
        rr = np.random.default_rng(200000 + m)
        rz = gaussian_born_rule_trajectory(
            model, h, rr, gamma0_override=cov0, orbitals0_override=orb0,
            ja_cached=ja, jb_cached=jb, proposal_c=1.0, **KW)
        p = phi_from_sigma(rz.final_covariance[ja, jb])
        acc += p; r2 += p * p
    mean = acc / M
    sd = np.sqrt(np.maximum(r2 / M - mean ** 2, 0)) / np.sqrt(M)
    est = (mean - phi0) / h
    print(f"  h={h}: G phi MC = {np.round(est,4)}  +- {np.round(sd/h,4)}"
          f"   | analytic = {np.round(g_ana,4)}")
