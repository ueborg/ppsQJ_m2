"""BANK snapshot averaging: measure the ENSEMBLE-ESTIMATOR autocorrelation C_O(tau)
and the achievable time-averaging gain.  Guided cloning only (= production).

The object that matters is NOT the per-clone autocorrelation but the correlation of
the POPULATION ESTIMATOR at different times, across independent realisations:
    rho_l = Corr[ Ohat(t), Ohat(t + l*DT) ]   pooled over stationary t
    g_snap(K) = K / (1 + 2 sum_{l=1}^{K-1} (1 - l/K) rho_l)
g_snap is the effective number of independent observable samples obtained from ONE
run using K equally spaced snapshots.  This is the last prerequisite for banking
snapshot averaging into production.
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
L = int(os.environ.get("L", 32)); T = float(os.environ.get("T", L))
N_c = int(os.environ.get("NC", 32)); NREP = int(os.environ.get("NREP", 20))
DT = float(os.environ.get("DT", 2.0)); BURN = float(os.environ.get("BURN", 0.5))
model, ja, jb = build(L, LAM)
dtau = 1.0/max(2.0*model.alpha*(L-1), 1e-6)
nw = max(int(round(DT/dtau)), 1); ngrid = int(round(T/(nw*dtau)))
print(f"SNAPSHOT GAIN  L={L} zeta={ZETA} T={T} N_c={N_c} DT={DT} "
      f"grid={ngrid} NREP={NREP}", flush=True)


def sysres(w, rng):
    c = np.cumsum(w); c /= c[-1]
    return np.searchsorted(c, (rng.uniform()+np.arange(len(w)))/len(w))


def one_run(rng):
    cov = [np.asarray(model.gamma0, float).copy() for _ in range(N_c)]
    orb = [np.asarray(model.orbitals0, complex).copy() for _ in range(N_c)]
    S_t, C_t, B_t = [], [], []
    for g in range(ngrid):
        for k in range(nw):
            lw = np.empty(N_c)
            for i in range(N_c):
                r = controlled_trajectory(model, dtau, rng, 0.0, ZETA, ja, jb,
                                          gamma0=cov[i], orbitals0=orb[i],
                                          simpson=False)
                cov[i], orb[i] = r["cov"], r["orb"]; lw[i] = r["log_w"]
            w = np.exp(lw-lw.max()); w /= w.sum()
            idx = sysres(w, rng)
            cov = [cov[j].copy() for j in idx]; orb = [orb[j].copy() for j in idx]
        ob = np.array([observables(c) for c in cov])
        S_t.append(ob[:, 0].mean()); C_t.append(ob[:, 1].mean())
        B_t.append(ob[:, 0].mean()*ob[:, 1].mean())
    return np.array(S_t), np.array(C_t), np.array(B_t)


t0 = time.time(); SS, CC, BB = [], [], []
for rep in range(NREP):
    a, b, c = one_run(np.random.default_rng(770000+523*rep))
    SS.append(a); CC.append(b); BB.append(c)
    if rep % 5 == 0:
        print(f"  rep {rep}  {time.time()-t0:.0f}s", flush=True)
SS, CC, BB = np.array(SS), np.array(CC), np.array(BB)
i0 = int(BURN*ngrid)
out = {}
for nm, A in (("S", SS), ("CMI", CC), ("B_L", BB)):
    n = ngrid - i0
    rho = []
    for l in range(0, min(n, 9)):
        pairs = [(A[:, t], A[:, t+l]) for t in range(i0, ngrid-l)]
        rs = [np.corrcoef(x, y)[0, 1] for x, y in pairs
              if x.std() > 1e-12 and y.std() > 1e-12]
        rho.append(float(np.mean(rs)) if rs else np.nan)
    g = {}
    for K in (2, 4, 8):
        if K <= len(rho):
            den = 1.0 + 2.0*sum((1-l/K)*rho[l] for l in range(1, K))
            g[K] = float(K/max(den, 1e-9))
    out[nm] = dict(rho=[round(float(x), 3) for x in rho], g_snap=g)
    print(f"\n{nm}: rho_l (lag in units of {DT}) = {[round(float(x),3) for x in rho]}",
          flush=True)
    print(f"   g_snap: " + "  ".join(f"K={k}: {v:.2f}x" for k, v in g.items()), flush=True)
json.dump(out, open(f"/tmp/snapgain_L{L}.json", "w"), indent=1)
print(f"\nwall total {time.time()-t0:.0f}s\nDONE", flush=True)
