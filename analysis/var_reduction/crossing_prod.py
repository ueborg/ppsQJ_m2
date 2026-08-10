"""PRODUCTION-MATCHED crossing experiment; simultaneously certifies snapshot averaging.

Same runs give three estimators of B_L per realisation:
    terminal   B_L(T)                      <- what production does now
    dense      mean of K snapshots at dt = L/8
    sparse     mean of K snapshots at dt = L/4  (~2 tau_int(L))
Then F(lambda) = B_{L2} - B_{L1}, root lambda_c = lambda0 - a/b from a local linear
fit, and the WHOLE fit is bootstrapped over realisations for each estimator.

Headline: G_snap^(lc) = Var(lc_terminal) / Var(lc_snap).  Walltime is identical
within the paired comparison, so this IS the efficiency gain -- no multiplying of
separately measured factors.

Config is production: guided, mult=4, T=L, independent lambda runs (coupling closed:
rho=0.083 at delta=0.04, and beating delta=0.04 would need rho>0.77 at 0.02).
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

ZETA = float(os.environ.get("ZETA", 0.9))
L1 = int(os.environ.get("L1", 32)); L2 = int(os.environ.get("L2", 64))
LAMS = [float(x) for x in os.environ.get("LAMS", "0.440,0.474,0.508").split(",")]
N_c = int(os.environ.get("NC", 32)); R = int(os.environ.get("R", 18))
MULT = int(os.environ.get("MULT", 4)); BURNF = float(os.environ.get("BURNF", 0.25))
print(f"CROSSING PROD  L1={L1} L2={L2} zeta={ZETA} lams={LAMS} N_c={N_c} R={R} "
      f"mult={MULT} T=L burn={BURNF}", flush=True)


def sysres(w, rng):
    c = np.cumsum(w); c /= c[-1]
    return np.searchsorted(c, (rng.uniform()+np.arange(len(w)))/len(w))


def one(L, lam, seed):
    """Return (terminal, dense_mean, sparse_mean) of B_L for one realisation."""
    T = float(L)
    model, ja, jb = build(L, lam)
    dtau = MULT/max(2.0*model.alpha*(L-1), 1e-6)
    dt_fine = L/8.0
    nfine = max(int(round(dt_fine/dtau)), 1)
    t0 = BURNF*T
    nburn = max(int(round(t0/dtau)), 1)
    ngrid = int((T-t0)/(nfine*dtau))
    rng = np.random.default_rng(seed)
    cov = [np.asarray(model.gamma0, float).copy() for _ in range(N_c)]
    orb = [np.asarray(model.orbitals0, complex).copy() for _ in range(N_c)]
    def step(n):
        nonlocal cov, orb
        for _ in range(n):
            lw = np.empty(N_c)
            for i in range(N_c):
                r = controlled_trajectory(model, dtau, rng, 0.0, ZETA, ja, jb,
                                          gamma0=cov[i], orbitals0=orb[i],
                                          simpson=False)
                cov[i], orb[i] = r["cov"], r["orb"]; lw[i] = r["log_w"]
            w = np.exp(lw-lw.max()); w /= w.sum()
            idx = sysres(w, rng)
            cov = [cov[j].copy() for j in idx]; orb = [orb[j].copy() for j in idx]
    step(nburn)
    vals = []
    for _ in range(ngrid):
        step(nfine)
        ob = np.array([observables(c) for c in cov])
        vals.append(ob[:, 0].mean()*ob[:, 1].mean())
    vals = np.array(vals)
    return float(vals[-1]), float(vals.mean()), float(vals[::2].mean())


data = {}
t_start = time.time()
for L in (L1, L2):
    for lam in LAMS:
        acc = []
        for rep in range(R):
            acc.append(one(L, lam, 660000 + 733*rep))
        data[f"{L}_{lam}"] = np.array(acc)     # (R, 3)
        m = np.array(acc).mean(axis=0)
        print(f"  L={L} lam={lam:.3f}: term={m[0]:.4f} dense={m[1]:.4f} "
              f"sparse={m[2]:.4f}  ({time.time()-t_start:.0f}s)", flush=True)
        json.dump({k: v.tolist() for k, v in data.items()},
                  open("/tmp/crossing_prod.json", "w"))

lam_arr = np.array(LAMS)
print("\n--- bootstrapped crossing, per estimator ---", flush=True)
rs = np.random.default_rng(5)
names = ["terminal", "dense(K=8)", "sparse(K=4)"]
out = {}
for j, nm in enumerate(names):
    roots = []
    for _ in range(4000):
        F = []
        for lam in LAMS:
            a1 = data[f"{L1}_{lam}"][:, j]; a2 = data[f"{L2}_{lam}"][:, j]
            i1 = rs.integers(0, len(a1), len(a1)); i2 = rs.integers(0, len(a2), len(a2))
            F.append(a2[i2].mean() - a1[i1].mean())
        b, a = np.polyfit(lam_arr, F, 1)
        if abs(b) > 1e-9:
            roots.append(-a/b)
    roots = np.array(roots)
    roots = roots[(roots > lam_arr.min()-0.2) & (roots < lam_arr.max()+0.2)]
    out[nm] = dict(lc=float(np.median(roots)), sd=float(roots.std(ddof=1)),
                   n=int(len(roots)))
    print(f"  {nm:12s}: lambda_c = {np.median(roots):.4f} +- {roots.std(ddof=1):.4f}"
          f"   (kept {len(roots)}/4000)", flush=True)
t = out["terminal"]["sd"]
for nm in names[1:]:
    print(f"  G_snap^(lc) [{nm}] = {(t/out[nm]['sd'])**2:.2f}x", flush=True)
json.dump(out, open("/tmp/crossing_prod_result.json", "w"), indent=1)
print("DONE", flush=True)
