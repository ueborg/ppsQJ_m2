"""Memos 2/3 done CORRECTLY: Galerkin projection of the Doob Poisson equation.

The first-order (in s = -log zeta) residual extensive log-weight is
    R(Gamma) = s * [ G g(Gamma) - delta_r(Gamma) ],
with G the BACKWARD GENERATOR of the original Gaussian trajectory process:
    G phi (Gamma) = (d phi/dt)_no-click + sum_j r_j(Gamma) [ phi(J_j Gamma) - phi(Gamma) ].
If g solves G g = delta_r exactly, R == 0 and log W_res is O(1): zero extensive
variance.  With g-hat = sum_m a_m phi_m the achievable gain is
    Gain = sigma_0^2 / sigma_res^2,
    sigma_0^2   = Var(int delta_r dt)/s_blk       (current c=zeta scheme)
    sigma_res^2 = Var(int [G g-hat - delta_r] dt)/s_blk   (controlled scheme)
minimised over a.  This is memo 2 sec 15-16, evaluated numerically.

NOTE this is NOT the additive control variate Var(Lambda - [g(0)-g(s)]), which
is guaranteed to give ~1 because it leaves the martingale intact.  The Doob
change of measure absorbs the martingale; this residual is the right object.

Bonds are (2j, 2j+3), disjoint in Majorana index, so all L-1 jump-updated bond
entries follow from one rank-2 outer-product formula (apply_projective_jump)
in a single O(L^2) pass.
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from doob_common import build, KW
from pps_qj.gaussian_backend import (gaussian_born_rule_trajectory,
                                     covariance_from_orbitals)

NF = 3   # features: K = sum q, sum q^2, sum q_j q_{j+1}


def bond_entries(cov, ja, jb):
    return cov[ja, jb]


def phi_from_sigma(sig):
    """sig = Gamma[ja,jb] for every bond -> feature vector."""
    q = np.clip(0.5 * (1.0 - sig), 0.0, 1.0)
    return np.array([q.sum(), (q * q).sum(), (q[:-1] * q[1:]).sum()])


def phi_after_all_jumps(cov, ja, jb):
    """phi(J_j Gamma) for every channel j, vectorised.  Returns (nb, NF)."""
    nb = len(ja)
    sig = cov[ja, jb]                       # (nb,)
    denom = 1.0 - sig                       # (nb,)
    # For a jump on bond j=(a,b): Gamma'_{xy} = Gamma_{xy}
    #   + (Gamma_{x a} Gamma_{y b} - Gamma_{x b} Gamma_{y a}) / denom_j
    # for x,y not in {a,b}; and Gamma'_{a,b} = -1.  Bonds are index-disjoint,
    # so for every other bond k=(ka,kb) only the outer-product term applies.
    A = cov[np.ix_(ja, ja)]                 # A[k,j] = Gamma[ka, ja]
    B = cov[np.ix_(jb, jb)]                 # B[k,j] = Gamma[kb, jb]
    Cc = cov[np.ix_(ja, jb)]                # C[k,j] = Gamma[ka, jb]
    D = cov[np.ix_(jb, ja)]                 # D[k,j] = Gamma[kb, ja]
    corr = (A * B - Cc * D) / denom[None, :]          # (nb_k, nb_j)
    sig_new = sig[:, None] + corr                     # (nb_k, nb_j)
    di = np.arange(nb)
    sig_new[di, di] = -1.0                            # jumped bond -> q = 1
    q_new = np.clip(0.5 * (1.0 - sig_new), 0.0, 1.0)  # (nb_k, nb_j)
    out = np.empty((nb, NF))
    out[:, 0] = q_new.sum(axis=0)
    out[:, 1] = (q_new * q_new).sum(axis=0)
    out[:, 2] = (q_new[:-1] * q_new[1:]).sum(axis=0)
    return out


def generator_phi(model, cov, orb, ja, jb, dt_nc=2e-3):
    """G phi = no-click drift + sum_j r_j [phi(J_j) - phi]."""
    sig = cov[ja, jb]
    q = np.clip(0.5 * (1.0 - sig), 0.0, 1.0)
    phi = phi_from_sigma(sig)
    r_j = 2.0 * model.alpha * q                       # per-channel rate
    phi_J = phi_after_all_jumps(cov, ja, jb)          # (nb, NF)
    jump_term = (r_j[:, None] * (phi_J - phi[None, :])).sum(axis=0)
    # no-click propagation via the cached eigendecomposition (expm is ~50x slower)
    coeffs = model.h_eff_V_inv @ orb
    tilde = model.h_eff_V @ (np.exp(model.h_eff_evals * dt_nc)[:, None] * coeffs)
    Q, _ = np.linalg.qr(tilde, mode="reduced")
    cov_nc = covariance_from_orbitals(Q)
    phi_nc = phi_from_sigma(cov_nc[ja, jb])
    drift = (phi_nc - phi) / dt_nc
    return phi, drift + jump_term, float(2.0 * model.alpha * q.sum())


def run(L, lam, zeta, T, delta, seed):
    model, ja, jb = build(L, lam)
    rng = np.random.default_rng(seed)
    cov = np.asarray(model.gamma0, float).copy()
    orb = np.asarray(model.orbitals0, complex).copy()
    n = int(round(T / delta))
    PHI = np.empty((n, NF)); GPHI = np.empty((n, NF)); RR = np.empty(n)
    DL = np.empty(n)
    for k in range(n):
        PHI[k], GPHI[k], RR[k] = generator_phi(model, cov, orb, ja, jb)
        res = gaussian_born_rule_trajectory(
            model, delta, rng, gamma0_override=cov, orbitals0_override=orb,
            ja_cached=ja, jb_cached=jb, proposal_c=zeta, **KW)
        cov, orb = res.final_covariance, res.final_orbitals
        DL[k] = res.Lambda
    return PHI, GPHI, RR, DL


def main():
    L = int(os.environ.get("L", 64))
    zeta = float(os.environ.get("ZETA", 0.9))
    lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
    T = float(os.environ.get("T", 160))
    delta = float(os.environ.get("DELTA", 0.5))
    ntraj = int(os.environ.get("NTRAJ", 16))
    burn = int(os.environ.get("BURN", 64))     # in grid units

    t0 = time.time()
    P, Gp, Rr, Dl = [], [], [], []
    for s in range(ntraj):
        a, b, c, dd = run(L, lam, zeta, T, delta, 50500 + s)
        P.append(a); Gp.append(b); Rr.append(c); Dl.append(dd)
        if s % 4 == 0:
            print(f"  traj {s}, {time.time()-t0:.0f}s", flush=True)
    P = np.array(P); Gp = np.array(Gp); Rr = np.array(Rr); Dl = np.array(Dl)
    np.savez_compressed(f"/tmp/galerkin_L{L}_z{zeta}.npz",
                        PHI=P, GPHI=Gp, R=Rr, DL=Dl, L=L, zeta=zeta,
                        lam=lam, delta=delta, burn=burn)

    rbar = Rr[:, burn:].mean()
    dr = Rr - rbar                       # delta_r time series
    print(f"\nrbar={rbar:.4f}  <G phi> (should be ~0 in steady state) = "
          f"{np.round(Gp[:, burn:].mean(axis=(0,1)), 4)}")

    out = dict(L=L, zeta=zeta, lam=lam, T=T, delta=delta, ntraj=ntraj,
               wall_s=time.time()-t0, rbar=float(rbar))
    for s_blk in (8.0, 16.0, 32.0):
        m = int(round(s_blk / delta))
        nav = Rr.shape[1] - burn
        nblk = nav // m
        if nblk < 2:
            continue
        # block integrals
        def blocks(series):        # series (ntraj, n) or (ntraj, n, NF)
            sl = series[:, burn:burn + nblk * m]
            if sl.ndim == 2:
                return sl.reshape(-1, m).sum(axis=1) * delta
            return sl.reshape(-1, m, sl.shape[-1]).sum(axis=1) * delta
        y = blocks(dr)                       # int delta_r dt
        Gb = blocks(Gp)                      # int G phi dt   (nblk*ntraj, NF)
        v0 = y.var(ddof=1)
        # minimise Var(Gb @ a - y)  ->  OLS with intercept
        Dz = np.column_stack([np.ones(len(y)), Gb])
        beta, *_ = np.linalg.lstsq(Dz, y, rcond=None)
        resid = y - Dz @ beta
        vres = resid.var(ddof=1)
        # single-feature (K only)
        D1 = np.column_stack([np.ones(len(y)), Gb[:, 0]])
        b1, *_ = np.linalg.lstsq(D1, y, rcond=None)
        v1 = (y - D1 @ b1).var(ddof=1)
        out[f"s={s_blk}"] = dict(
            nblocks=int(len(y)),
            sigma0_sq=float(v0 / s_blk),
            sigma_res_sq_K=float(v1 / s_blk),
            sigma_res_sq_all=float(vres / s_blk),
            Gain_K=float(v0 / v1), Gain_all=float(v0 / vres),
            a_K=float(b1[1]), a_all=[float(x) for x in beta[1:]])
        print(f"s={s_blk:5.1f}  nblk={len(y):4d}  sigma0^2={v0/s_blk:7.3f}  "
              f"Gain(K)={v0/v1:6.2f}  Gain(all3)={v0/vres:6.2f}  a_K={b1[1]:+.4f}")
    with open(f"/tmp/galerkin_L{L}_z{zeta}.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
