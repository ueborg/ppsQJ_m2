"""Round 1 screening, v2 -- fixes to /tmp/doob_screen.py.

Changes:
  * leave-one-TRAJECTORY-out CV (random folds leak within-trajectory correlation)
  * Geyer initial-positive-sequence tau_r (raw acorr is biased negative at ~n/2)
  * Var(Lambda over window of length s) vs s  -- the decisive diagnostic:
        linear in s  => diffusive martingale noise, removable only via the
                        Poisson-eq control, gain bounded by the R^2 below
        ~ s^2        => quasi-static per-trajectory activity offset, which a
                        state-dependent guide removes IF it is Gamma-predictable
  * saves the raw (r, X, dLambda) arrays for round 2
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from doob_common import run_one  # noqa


def geyer_tau(C, dx):
    """Initial-positive-sequence integrated time (Geyer 1992), one-sided."""
    s = 0.5 * C[0]
    for k in range(1, len(C) - 1, 2):
        p = C[k] + C[k + 1]
        if p <= 0:
            break
        s += p
    return float(s * dx)


def acorr(x, maxlag):
    x = x - x.mean(); n = len(x); v = (x * x).mean()
    return np.array([1.0 if k == 0 else (x[:n-k] * x[k:]).mean() / v
                     for k in range(maxlag + 1)])


def loo_traj_r2(Xlist, Ylist):
    """Leave-one-trajectory-out cross-fitted R^2 for OLS."""
    n_tr = len(Xlist)
    preds, truth = [], []
    for i in range(n_tr):
        Xtr = np.vstack([Xlist[j] for j in range(n_tr) if j != i])
        Ytr = np.concatenate([Ylist[j] for j in range(n_tr) if j != i])
        Xtr_d = np.column_stack([np.ones(len(Ytr)), Xtr])
        beta, *_ = np.linalg.lstsq(Xtr_d, Ytr, rcond=None)
        Xte = np.column_stack([np.ones(len(Ylist[i])), Xlist[i]])
        preds.append(Xte @ beta); truth.append(Ylist[i])
    p = np.concatenate(preds); y = np.concatenate(truth)
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    L = int(os.environ.get("L", 64))
    zeta = float(os.environ.get("ZETA", 0.9))
    lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
    T = float(os.environ.get("T", 160))
    delta = float(os.environ.get("DELTA", 1.0))
    ntraj = int(os.environ.get("NTRAJ", 24))
    burn = int(os.environ.get("BURN", 32))

    t0 = time.time()
    R, X, DL, NJ = [], [], [], []
    for s in range(ntraj):
        r, x, dl, nj = run_one(L, lam, zeta, T, delta, 90000 + s)
        R.append(r); X.append(x); DL.append(dl); NJ.append(nj)
        if s % 4 == 0:
            print(f"  traj {s}: n_jumps={nj}, elapsed={time.time()-t0:.0f}s", flush=True)
    R = np.array(R); X = np.array(X); DL = np.array(DL)
    tag = f"L{L}_z{zeta}"
    np.savez_compressed(f"/tmp/doob_raw_{tag}.npz", R=R, X=X, DL=DL,
                        L=L, lam=lam, zeta=zeta, T=T, delta=delta, burn=burn)

    out = dict(L=L, lam=lam, zeta=zeta, T=T, delta=delta, ntraj=ntraj,
               wall_s=time.time() - t0, n_jumps_mean=float(np.mean(NJ)))

    # --- Var(Lambda) vs window length ---
    DLs = DL[:, burn:]
    nav = DLs.shape[1]
    vw = {}
    for s_len in (4, 8, 16, 32, 64, 96):
        m = int(round(s_len / delta))
        if m > nav:
            continue
        # non-overlapping windows pooled across trajectories
        nblk = nav // m
        blocks = DLs[:, :nblk * m].reshape(ntraj * nblk, m).sum(axis=1)
        vw[s_len] = dict(var=float(blocks.var(ddof=1)), mean=float(blocks.mean()),
                         n=int(len(blocks)))
    out["Var_Lambda_vs_window"] = vw
    out["Var_logW_perL"] = {k: (1 - zeta) ** 2 * v["var"] for k, v in vw.items()}

    # --- activity autocorrelation ---
    Rs = R[:, burn:]
    maxlag = min(int(round(40 / delta)), Rs.shape[1] // 3)
    C = np.mean([acorr(Rs[i], maxlag) for i in range(ntraj)], axis=0)
    var_r = float(np.mean([Rs[i].var(ddof=1) for i in range(ntraj)]))
    var_r_between = float(Rs.mean(axis=1).var(ddof=1))
    out["tau_r_geyer"] = geyer_tau(C, delta)
    out["var_r_within"] = var_r
    out["var_r_between_traj"] = var_r_between
    out["C_r"] = C.tolist()

    # --- predictability of FUTURE integrated excess activity ---
    rbar = float(Rs.mean())
    res = {}
    for tau in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        m = int(round(tau / delta))
        Xl, Yl = [], []
        for i in range(ntraj):
            ri = R[i]; xi = X[i]
            hi = len(ri) - m
            if hi <= burn + 5:
                continue
            csum = np.concatenate([[0.0], np.cumsum(ri - rbar) * delta])
            Y = csum[m:hi + m] - csum[:hi]
            Xl.append(xi[burn:hi]); Yl.append(Y[burn:hi])
        if len(Xl) < 4:
            continue
        Xq = [np.column_stack([a, a[:, 0] ** 2, a[:, 4] ** 2]) for a in Xl]
        Xk = [a[:, :1] for a in Xl]
        res[f"tau={tau}"] = dict(
            R2_all=loo_traj_r2(Xl, Yl),
            R2_quad=loo_traj_r2(Xq, Yl),
            R2_K_only=loo_traj_r2(Xk, Yl),
            sd_Y=float(np.concatenate(Yl).std(ddof=1)),
            n=int(sum(len(y) for y in Yl)))
    out["future_activity_R2"] = res

    with open(f"/tmp/doob_screen2_{tag}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "C_r"}, indent=1))
    print("C_r:", [round(float(C[int(round(t/delta))]), 3)
                   for t in (0, 1, 2, 4, 8, 16, 32) if int(round(t/delta)) < len(C)])


if __name__ == "__main__":
    main()
