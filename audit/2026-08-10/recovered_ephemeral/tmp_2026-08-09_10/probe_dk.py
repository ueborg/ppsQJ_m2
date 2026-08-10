"""Feasibility probe for the controlled sampler.

Controlled rates  r_hat_j = zeta * r_j * exp(a * Delta_j K),  Delta_j K = K(J_j Gamma) - K(Gamma).
Thinning from a dominating process needs  M >= max_j exp(a Delta_j K) = exp(a * min_j Delta_j K)
(a < 0).  M is the candidate-overhead factor, so this decides tractability.
"""
import os, sys
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from doob_common import build, KW
from doob_galerkin import phi_after_all_jumps
from pps_qj.gaussian_backend import gaussian_born_rule_trajectory

L = int(os.environ.get("L", 64))
zeta = float(os.environ.get("ZETA", 0.9))
lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
a = float(os.environ.get("A", -3.55))
model, ja, jb = build(L, lam)
rng = np.random.default_rng(31)
cov = np.asarray(model.gamma0, float).copy()
orb = np.asarray(model.orbitals0, complex).copy()

dks, tilt_max, tilt_mean, wsum = [], [], [], []
for k in range(180):
    res = gaussian_born_rule_trajectory(model, 0.5, rng, gamma0_override=cov,
                                        orbitals0_override=orb, ja_cached=ja,
                                        jb_cached=jb, proposal_c=zeta, **KW)
    cov, orb = res.final_covariance, res.final_orbitals
    if k < 60:
        continue
    q = np.clip(0.5 * (1.0 - cov[ja, jb]), 0.0, 1.0)
    K = q.sum()
    dK = phi_after_all_jumps(cov, ja, jb)[:, 0] - K
    t = np.exp(a * dK)
    dks.append(dK)
    tilt_max.append(t.max()); tilt_mean.append((q * t).sum() / max(q.sum(), 1e-12))
    wsum.append(t.max() / ((q * t).sum() / max(q.sum(), 1e-12)))

dks = np.concatenate(dks)
print(f"L={L} zeta={zeta} lam={lam:.4f} a={a}")
print(f"Delta_j K  : min={dks.min():.4f}  p1={np.percentile(dks,1):.4f}  "
      f"median={np.median(dks):.4f}  p99={np.percentile(dks,99):.4f}  max={dks.max():.4f}")
print(f"  frac(Delta_j K < 0) = {(dks<0).mean():.4f}")
print(f"exp(a dK)  : max over states = {max(tilt_max):.4f}   "
      f"mean tilt <t>_q = {np.mean(tilt_mean):.4f}")
print(f"M = max tilt -> candidate overhead vs the r_hat process = "
      f"{max(tilt_max)/np.mean(tilt_mean):.2f}x")
print(f"rate suppression zeta*<t>_q = {zeta*np.mean(tilt_mean):.4f} "
      f"(vs {zeta} for the plain guide)")
