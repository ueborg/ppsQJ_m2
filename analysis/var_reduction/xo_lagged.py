"""Observable-alignment test (memo 11): does the RESIDUAL selection fluctuation
predict the FUTURE observable?

Particle-filter variance is governed by Cov[selection at t, E(O_T | Gamma_t)], not by
equal-time Corr(K_t, O_t).  Cheap stand-in: lagged cross-correlation with future O.

    C_XO(tau) = Corr[ X(t), O(t+tau) ]      X = per-interval selection increment
    C_KO(tau) = Corr[ K(t), O(t+tau) ]      K = the control feature itself

guided     X_g = -(1-zeta) dLambda
ctrl/twist X_c = dl_raw + a(u)K_u - a(t)K_t   (telescopes within the interval)

SIGNATURE OF "wrong mode": |C_X,BL| small at L=32 but LARGER / longer-lived at L=64,
i.e. total selection variance falls while what remains is increasingly BL-aligned.
If C_X,BL is tiny at both sizes, observable alignment is weak -> go to the real-
population family-persistence diagnostic instead.
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from traj_common import build
from controlled_sampler import controlled_trajectory, observables

ZETA = float(os.environ.get("ZETA", 0.9)); LAM = 0.5*np.sqrt(ZETA)
AK = float(os.environ.get("AK", -3.57)); A_STAR = -np.log(ZETA)*AK
TAU_K = -AK/(2.0*LAM)
M = int(os.environ.get("M", 60)); DT = float(os.environ.get("DT", 1.0))
LS = [int(x) for x in os.environ.get("LS", "32,64").split(",")]
LAGS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
print(f"X-vs-future-O  zeta={ZETA} a*={A_STAR:.4f} tau_K={TAU_K:.2f} M={M} dt={DT}",
      flush=True)
out = {}
for L in LS:
    T = float(L); model, ja, jb = build(L, LAM)
    dtau = 1.0/max(2.0*model.alpha*(L-1), 1e-6)
    nw = max(int(round(DT/dtau)), 1)          # windows per observation interval
    ngrid = int(round(T/(nw*dtau)))
    taper = lambda t: A_STAR*(1.0 - np.exp(-(T-t)/TAU_K))
    Kof = lambda cov: float(np.clip(0.5*(1.0-cov[ja, jb]), 0.0, 1.0).sum())
    for arm in ("guided", "ctrl"):
        t0 = time.time()
        Xs = np.empty((M, ngrid)); Ks = np.empty((M, ngrid))
        Cs = np.empty((M, ngrid)); Bs = np.empty((M, ngrid))
        for m in range(M):
            rng = np.random.default_rng(620000 + 173*m + 11*L)
            cov = np.asarray(model.gamma0, float).copy()
            orb = np.asarray(model.orbitals0, complex).copy()
            for g in range(ngrid):
                t_a = g*nw*dtau
                Ks[m, g] = Kof(cov)
                acc = 0.0
                for k in range(nw):
                    tk, tu = t_a + k*dtau, t_a + (k+1)*dtau
                    ak = 0.0 if arm == "guided" else taper(tk)
                    au = 0.0 if arm == "guided" else taper(tu)
                    K0 = Kof(cov)
                    r = controlled_trajectory(model, dtau, rng, ak, ZETA, ja, jb,
                                              gamma0=cov, orbitals0=orb, simpson=False)
                    cov, orb = r["cov"], r["orb"]
                    acc += r["log_w"] + (au*r["K_end"] - ak*K0 if arm == "ctrl" else 0.0)
                Xs[m, g] = acc
                s_, c_ = observables(cov); Cs[m, g] = c_; Bs[m, g] = s_*c_
        half = ngrid//2
        def lagcorr(A, B, lag):
            n = int(round(lag/DT)); r = []
            for m in range(M):
                a = A[m, half:ngrid-n] if n > 0 else A[m, half:]
                b = B[m, half+n:] if n > 0 else B[m, half:]
                if len(a) < 6: return np.nan, np.nan
                if a.std() < 1e-14 or b.std() < 1e-14: continue
                r.append(np.corrcoef(a, b)[0, 1])
            r = np.array(r)
            return float(r.mean()), float(r.std(ddof=1)/np.sqrt(len(r)))
        row = {}
        for lag in LAGS:
            for nm, B in (("BL", Bs), ("CMI", Cs)):
                row[f"X-{nm}@{lag}"] = lagcorr(Xs, B, lag)
                row[f"K-{nm}@{lag}"] = lagcorr(Ks, B, lag)
        out[f"L{L}_{arm}"] = row
        print(f"\nL={L} {arm}  (wall {time.time()-t0:.0f}s)", flush=True)
        print("  lag   C[X,BL]        C[X,CMI]       C[K,BL]", flush=True)
        for lag in LAGS:
            f = lambda k: (f"{row[k][0]:+.3f}+-{row[k][1]:.3f}"
                           if not np.isnan(row[k][0]) else "   n/a      ")
            print(f"  {lag:4.0f}  {f(f'X-BL@{lag}')}  {f(f'X-CMI@{lag}')}  "
                  f"{f(f'K-BL@{lag}')}", flush=True)
        json.dump({k: {kk: list(vv) for kk, vv in v.items()} for k, v in out.items()},
                  open("/tmp/xo_lagged.json", "w"), indent=1)
print("\nDONE", flush=True)
