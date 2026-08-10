"""Round 1 screening for memos 2/3 (approximate-Doob state-dependent control).

Decisive question, per memo 3 'What I would actually calculate now':
  how much of the FUTURE integrated excess activity Y_t(tau) = int_t^{t+tau}(r-rbar)dt'
  is predictable from the CURRENT Gaussian state Gamma_t?

That fraction bounds how much of Var(log W) = (1-zeta)^2 Var(Lambda_T) a
state-dependent guide built from Gaussian features could remove.

This is a DIFFERENT regression from VARIANCE_REDUCTION.md sec 5 (which regressed
the per-window weight on window-start features, R^2 ~ 0.05).  Short horizon vs
long horizon.

Outputs:
  Var(Lambda_T)        -- gate against HANDOFF 2026-07-27 (36.8/(1-zeta)^2 at L=64,z=0.9)
  C_r(t), tau_r        -- activity autocorrelation + integrated time
  2 tau_r Var(r) T     -- Green-Kubo prediction for Var(Lambda_T)
  R^2(tau)             -- cross-fitted, features -> Y_t(tau)
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from pps_qj.gaussian_backend import (build_gaussian_chain_model,
                                     gaussian_born_rule_trajectory,
                                     entanglement_entropy)

KW = dict(jump_update_method="lowrank", solver_method="newton", bisection_tol=1e-6)


def activity_features(model, cov, ja, jb, L):
    """q_j = (1 - Gamma[ja,jb])/2 ; r = 2 alpha sum_j q_j  (HANDOFF rate convention)."""
    q = np.clip(0.5 * (1.0 - cov[ja, jb]), 0.0, 1.0)
    K = q.sum()
    feats = np.array([
        K,
        (q * q).sum(),
        (q[:-1] * q[1:]).sum(),
        (q[:-2] * q[2:]).sum(),
        entanglement_entropy(cov, L // 2),
    ])
    r = 2.0 * model.alpha * K
    return r, feats


def run_one(L, lam, zeta, T, delta, seed):
    """One guided (c=zeta) trajectory, state sampled every `delta` time units."""
    alpha, w = lam, 1.0 - lam
    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)
    jp = model.jump_pairs
    ja = np.array([p[0] for p in jp], dtype=np.intp)
    jb = np.array([p[1] for p in jp], dtype=np.intp)
    rng = np.random.default_rng(seed)

    cov = np.asarray(model.gamma0, dtype=np.float64).copy()
    orb = np.asarray(model.orbitals0, dtype=np.complex128).copy()
    nsteps = int(round(T / delta))
    r_series = np.empty(nsteps)
    X_series = np.empty((nsteps, 5))
    dLam = np.empty(nsteps)
    njump = 0
    for k in range(nsteps):
        r_k, x_k = activity_features(model, cov, ja, jb, L)
        r_series[k] = r_k
        X_series[k] = x_k
        res = gaussian_born_rule_trajectory(
            model, delta, rng, gamma0_override=cov, orbitals0_override=orb,
            ja_cached=ja, jb_cached=jb, proposal_c=zeta, **KW)
        cov, orb = res.final_covariance, res.final_orbitals
        dLam[k] = res.Lambda
        njump += res.n_jumps
    return r_series, X_series, dLam, njump


def acorr(x, maxlag):
    x = x - x.mean()
    n = len(x)
    v = (x * x).mean()
    return np.array([1.0 if k == 0 else (x[:n-k] * x[k:]).mean() / v
                     for k in range(maxlag + 1)])


def cv_r2(X, y, nfold=5):
    """Cross-fitted R^2 of an OLS linear model (avoids in-sample optimism)."""
    n = len(y)
    idx = np.arange(n)
    folds = np.array_split(idx, nfold)
    pred = np.empty(n)
    Xd = np.column_stack([np.ones(n), X])
    for f in folds:
        m = np.ones(n, bool); m[f] = False
        beta, *_ = np.linalg.lstsq(Xd[m], y[m], rcond=None)
        pred[f] = Xd[f] @ beta
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def main():
    L = int(os.environ.get("L", 64))
    zeta = float(os.environ.get("ZETA", 0.9))
    lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
    T = float(os.environ.get("T", 128))
    delta = float(os.environ.get("DELTA", 1.0))
    ntraj = int(os.environ.get("NTRAJ", 12))
    burn = int(os.environ.get("BURN", 32))     # grid points discarded

    t0 = time.time()
    R, X, DL, NJ = [], [], [], []
    for s in range(ntraj):
        r, x, dl, nj = run_one(L, lam, zeta, T, delta, 90000 + s)
        R.append(r); X.append(x); DL.append(dl); NJ.append(nj)
        print(f"  traj {s}: n_jumps={nj}, Lambda_T={dl.sum():.1f}, "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
    R = np.array(R); X = np.array(X); DL = np.array(DL)

    out = dict(L=L, lam=lam, zeta=zeta, T=T, delta=delta, ntraj=ntraj,
               wall_s=time.time() - t0, n_jumps_mean=float(np.mean(NJ)))

    # --- gate 1: Var(Lambda) over a T=L horizon, and Var(log W) ---
    nL = int(round(L / delta))
    LamL = DL[:, burn:burn + nL].sum(axis=1)
    out["Lambda_TL_mean"] = float(LamL.mean())
    out["Var_Lambda_TL"] = float(LamL.var(ddof=1))
    out["Var_logW_TL"] = float((1 - zeta) ** 2 * LamL.var(ddof=1))

    # --- gate 2: activity autocorrelation ---
    Rs = R[:, burn:]
    maxlag = int(round(24 / delta))
    C = np.mean([acorr(Rs[i], maxlag) for i in range(ntraj)], axis=0)
    tau_r = float(np.trapezoid(C, dx=delta))
    var_r = float(np.mean([Rs[i].var(ddof=1) for i in range(ntraj)]))
    out["tau_r"] = tau_r
    out["var_r"] = var_r
    out["GreenKubo_Var_Lambda_TL"] = 2.0 * tau_r * var_r * L
    out["C_r"] = C.tolist()

    # --- gate 3: predictability of FUTURE integrated excess activity ---
    rbar = Rs.mean()
    res_r2 = {}
    for tau in (2.0, 4.0, 8.0, 16.0, 32.0):
        m = int(round(tau / delta))
        Xs, Ys = [], []
        for i in range(ntraj):
            ri = R[i]; xi = X[i]
            hi = len(ri) - m
            csum = np.concatenate([[0.0], np.cumsum(ri - rbar) * delta])
            Y = csum[m:hi + m] - csum[:hi]
            Xs.append(xi[burn:hi]); Ys.append(Y[burn:hi])
        Xs = np.vstack(Xs); Ys = np.concatenate(Ys)
        Xq = np.column_stack([Xs, Xs[:, 0] ** 2, Xs[:, 4] ** 2])
        res_r2[f"tau={tau}"] = dict(
            R2_linear=float(cv_r2(Xs, Ys)),
            R2_quad=float(cv_r2(Xq, Ys)),
            R2_K_only=float(cv_r2(Xs[:, :1], Ys)),
            n=int(len(Ys)),
            sd_Y=float(Ys.std(ddof=1)),
        )
    out["future_activity_R2"] = res_r2

    tag = f"L{L}_z{zeta}"
    with open(f"/tmp/doob_screen_{tag}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "C_r"}, indent=1))
    print("C_r(t) at t=0,1,2,4,8,16:",
          [round(float(C[int(round(t/delta))]), 3) for t in (0, 1, 2, 4, 8, 16)])


if __name__ == "__main__":
    main()
