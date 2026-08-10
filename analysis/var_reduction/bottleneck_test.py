"""Is SELECTION the bottleneck for Var(B_L), or is it intrinsic ensemble spread?

TEST 1 -- population vs independent spread.
  Var of B_L ACROSS the N_c clones at T, vs across M INDEPENDENT trajectories.
  If comparable, the population already samples the intrinsic spread and no
  resampling/allocation scheme can help: memo tests A, B and observable-aware
  cloning would all be attacking a non-problem.
  ratio ~1 -> selection is NOT the bottleneck.  ratio <<1 -> it is.

TEST 2 -- sibling decorrelation (memo sec 3 go/no-go for stratified siblings).
  From one equilibrated state launch iid sibling PAIRS, measure Corr(O_1,O_2) at
  dt = 1,2,4,8,16.  Antithetic/stratified sibling RNG can only help if iid siblings
  are still correlated on the timescale that matters; rho_iid ~ 0 => no headroom.
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
L = int(os.environ.get("L", 32)); T = float(os.environ.get("T", L))
N_c = int(os.environ.get("NC", 32)); NREP = int(os.environ.get("NREP", 5))
NPAIR = int(os.environ.get("NPAIR", 150)); MIND = int(os.environ.get("MIND", 120))
model, ja, jb = build(L, LAM)
dtau = 1.0/max(2.0*model.alpha*(L-1), 1e-6); nstep = int(round(T/dtau))
taper = lambda t: A_STAR*(1.0 - np.exp(-(T-t)/TAU_K))
Kof = lambda cov: float(np.clip(0.5*(1.0-cov[ja, jb]), 0.0, 1.0).sum())
print(f"BOTTLENECK  L={L} zeta={ZETA} T={T} N_c={N_c} a*={A_STAR:.4f}", flush=True)


def sysres(w, rng):
    c = np.cumsum(w); c /= c[-1]
    return np.searchsorted(c, (rng.uniform() + np.arange(len(w)))/len(w))


def clone_pop(arm, rng):
    cov = [np.asarray(model.gamma0, float).copy() for _ in range(N_c)]
    orb = [np.asarray(model.orbitals0, complex).copy() for _ in range(N_c)]
    for k in range(nstep):
        tk, tu = k*dtau, (k+1)*dtau
        ak = 0.0 if arm == "guided" else taper(tk)
        au = 0.0 if arm == "guided" else taper(tu)
        lw = np.empty(N_c)
        for i in range(N_c):
            K0 = Kof(cov[i])
            r = controlled_trajectory(model, dtau, rng, ak, ZETA, ja, jb,
                                      gamma0=cov[i], orbitals0=orb[i], simpson=False)
            cov[i], orb[i] = r["cov"], r["orb"]
            lw[i] = r["log_w"] + (au*r["K_end"] - ak*K0 if arm == "ctrl" else 0.0)
        w = np.exp(lw - lw.max()); w /= w.sum()
        idx = sysres(w, rng)
        cov = [cov[j].copy() for j in idx]; orb = [orb[j].copy() for j in idx]
    return np.array([observables(c)[1]*observables(c)[0] for c in cov])


print("\nTEST 1: within-population vs independent spread of B_L", flush=True)
res1 = {}
for arm in ("guided", "ctrl"):
    t0 = time.time(); vp = []
    for rep in range(NREP):
        b = clone_pop(arm, np.random.default_rng(930000 + 311*rep))
        vp.append(b.var(ddof=1))
    # independent trajectories under the SAME arm dynamics
    bi = []
    for m in range(MIND):
        r = controlled_trajectory(model, T, np.random.default_rng(940000+m),
                                  A_STAR if arm == "ctrl" else 0.0, ZETA, ja, jb,
                                  simpson=False)
        s_, c_ = observables(r["cov"]); bi.append(s_*c_)
    vi = float(np.var(bi, ddof=1))
    res1[arm] = dict(var_pop=float(np.mean(vp)), var_ind=vi,
                     ratio=float(np.mean(vp)/vi), wall=time.time()-t0)
    print(f"  {arm:7s}: Var_pop(B_L)={np.mean(vp):.5f}  Var_indep={vi:.5f}  "
          f"ratio={np.mean(vp)/vi:.3f}   ({time.time()-t0:.0f}s)", flush=True)

print("\nTEST 2: iid sibling correlation of B_L / CMI from a common state", flush=True)
r0 = controlled_trajectory(model, T, np.random.default_rng(951), A_STAR, ZETA,
                           ja, jb, simpson=False)
cov0, orb0 = r0["cov"], r0["orb"]
DTS = [1.0, 2.0, 4.0, 8.0, 16.0]
res2 = {}
for dt in DTS:
    B1, B2, C1, C2 = [], [], [], []
    for p in range(NPAIR):
        vals = []
        for child in (0, 1):
            rr = controlled_trajectory(model, dt, np.random.default_rng(960000+2*p+child),
                                       A_STAR, ZETA, ja, jb, gamma0=cov0,
                                       orbitals0=orb0, simpson=False)
            s_, c_ = observables(rr["cov"]); vals.append((s_*c_, c_))
        B1.append(vals[0][0]); B2.append(vals[1][0])
        C1.append(vals[0][1]); C2.append(vals[1][1])
    rb = float(np.corrcoef(B1, B2)[0, 1]); rc = float(np.corrcoef(C1, C2)[0, 1])
    se = (1-rb**2)/np.sqrt(max(NPAIR-3, 1))
    res2[dt] = dict(rho_BL=rb, rho_CMI=rc, se=float(se))
    print(f"  dt={dt:5.1f}: rho(B_L)={rb:+.3f}+-{se:.3f}  rho(CMI)={rc:+.3f}"
          f"   gain if fully decorrelated = {(1+rb)/1.0:.2f}x", flush=True)
json.dump(dict(test1=res1, test2={str(k): v for k, v in res2.items()}),
          open(f"/tmp/bottleneck_L{L}.json", "w"), indent=1)
print("DONE", flush=True)
