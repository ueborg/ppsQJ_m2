"""Cheap L-scaling of the per-window selection strength D_2 (memo 9).

Per cloning window the incremental weight is G = e^X with
    guided     X_g = -(1-zeta) dLambda
    ctrl/twist X_c = dl_raw + a(u) K_u - a(t) K_t
and D_2 = log(<G^2>/<G>^2) controls coalescence: the chance two offspring pick the
same parent is ~ e^{D_2}/N_c, so to hold genealogical degeneracy fixed
    N_c^ctrl / N_c^guide  ~  SUM_k D_2,k^ctrl / SUM_k D_2,k^guide.
The HORIZON SUM is the right statistic, not the per-window value: per-window ESS is
~1.000 for both arms, and the 1.99-vs-6.23 GESS gap at L=32 is thousands of tiny
events accumulating.

No cloning population is evolved -- M independent trajectories per arm, D_2
estimated ACROSS trajectories at each window index. NOTE this uses the proposal
ensemble rather than the resampled tilted population, so it is a scaling
diagnostic, not an exact prediction.
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from traj_common import build
from controlled_sampler import controlled_trajectory

ZETA = float(os.environ.get("ZETA", 0.9)); LAM = 0.5 * np.sqrt(ZETA)
AK = float(os.environ.get("AK", -3.57)); A_STAR = -np.log(ZETA) * AK
TAU_K = -AK / (2.0 * LAM)
M = int(os.environ.get("M", 40))
LS = [int(x) for x in os.environ.get("LS", "16,24,32,48,64").split(",")]
print(f"D2 SCALING  zeta={ZETA} lam={LAM:.4f} a*={A_STAR:.4f} tau_K={TAU_K:.2f} M={M}",
      flush=True)
print(f"{'L':>4} {'arm':>7} {'windows':>8} {'sum D2':>9} {'mean D2':>10} "
      f"{'Var(totX)':>10} {'tau_X':>7} {'wall':>7}", flush=True)


def acorr_tau(x, maxlag=200):
    x = np.asarray(x) - np.mean(x); v = np.mean(x * x)
    if v <= 0: return 0.0
    s = 0.5
    for k in range(1, maxlag - 1, 2):
        p = (np.mean(x[:len(x)-k]*x[k:]) + np.mean(x[:len(x)-k-1]*x[k+1:])) / v
        if p <= 0: break
        s += p
    return float(s)


out = {}
for L in LS:
    T = float(L)
    model, ja, jb = build(L, LAM)
    dtau = 1.0 / max(2.0 * model.alpha * (L - 1), 1e-6)
    nstep = int(round(T / dtau))
    taper = lambda t: A_STAR * (1.0 - np.exp(-(T - t) / TAU_K))
    Kof = lambda cov: float(np.clip(0.5*(1.0-cov[ja, jb]), 0.0, 1.0).sum())
    for arm in ("guided", "ctrl"):
        t0 = time.time()
        X = np.empty((M, nstep))
        for m in range(M):
            rng = np.random.default_rng(410000 + 131*m + 7*L)
            cov = np.asarray(model.gamma0, float).copy()
            orb = np.asarray(model.orbitals0, complex).copy()
            for k in range(nstep):
                tk, tu = k*dtau, (k+1)*dtau
                ak = 0.0 if arm == "guided" else taper(tk)
                au = 0.0 if arm == "guided" else taper(tu)
                K0 = Kof(cov)
                r = controlled_trajectory(model, dtau, rng, ak, ZETA, ja, jb,
                                          gamma0=cov, orbitals0=orb, simpson=False)
                cov, orb = r["cov"], r["orb"]
                X[m, k] = r["log_w"] + (au*r["K_end"] - ak*K0 if arm == "ctrl" else 0.0)
        half = nstep // 2                      # stationary half
        # D_2 per window across the M trajectories
        D2 = []
        for k in range(half, nstep):
            x = X[:, k] - X[:, k].mean()
            D2.append(float(np.log(np.mean(np.exp(2*x))) - 2*np.log(np.mean(np.exp(x)))))
        D2 = np.array(D2)
        sumD2 = float(D2.sum() * (nstep / len(D2)))    # rescale to full horizon
        totX = X.sum(axis=1)
        tau_X = float(np.mean([acorr_tau(X[m, half:]) for m in range(M)]))
        key = f"L{L}_{arm}"
        out[key] = dict(L=L, arm=arm, nstep=nstep, sumD2=sumD2,
                        meanD2=float(D2.mean()), varTot=float(totX.var(ddof=1)),
                        tau_X=tau_X, wall=time.time()-t0)
        v = out[key]
        print(f"{L:4d} {arm:>7} {nstep:8d} {sumD2:9.3f} {D2.mean():10.3e} "
              f"{v['varTot']:10.3f} {tau_X:7.2f} {v['wall']:6.1f}s", flush=True)
        json.dump(out, open("/tmp/d2scaling.json", "w"), indent=1)

print("\n--- predicted relative population requirement R_N = sumD2_ctrl / sumD2_guided ---")
for L in LS:
    g, c = out.get(f"L{L}_guided"), out.get(f"L{L}_ctrl")
    if g and c:
        print(f"  L={L:3d}  sumD2 guided={g['sumD2']:8.3f}  ctrl={c['sumD2']:8.3f}  "
              f"R_N={c['sumD2']/g['sumD2']:6.3f}   (Var(totX) ratio "
              f"{c['varTot']/g['varTot']:6.3f})", flush=True)
print("DONE", flush=True)
