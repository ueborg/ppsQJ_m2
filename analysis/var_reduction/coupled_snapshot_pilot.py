"""THE production-design experiment: coupled-lambda x spaced snapshots.

Do the two gains COMPOUND?  Memo 13 is right not to assume 2x * 5x = 10x.
For each realisation, run lambda0-delta, lambda0, lambda0+delta with COMMON RNG and
record B_L at stationary snapshots spaced DTS (>= 2 tau_int ~ 8).  Then

    D(t) = B_L^+(t) - B_L^-(t)          paired difference  (= 2 delta F')
    Dbar = (1/K) sum_m D(t_m)           + time averaging

and compare Var(Dbar) against (i) a single snapshot, (ii) INDEPENDENT RNG.
Also reports the GLS optimal-weight variance 1/(1' Sigma^-1 1) vs equal weights.

Target metric is the one that matters: E_lc = 1/(t_wall Var(lambda_c_hat)),
Var(lambda_c_hat) ~ Var(Dbar) / (dD/dlambda)^2 with dD/dlambda ~ D_mean/(2 delta).
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
LAM0 = float(os.environ.get("LAM0", 0.5*np.sqrt(ZETA)))
DELTA = float(os.environ.get("DELTA", 0.04))
L = int(os.environ.get("L", 32)); T = float(os.environ.get("T", L))
N_c = int(os.environ.get("NC", 32)); R = int(os.environ.get("R", 24))
DTS = float(os.environ.get("DTS", 8.0)); BURN = float(os.environ.get("BURNT", 0.4))
LAMS = [LAM0-DELTA, LAM0, LAM0+DELTA]
print(f"COUPLED+SNAPSHOT  L={L} zeta={ZETA} lam0={LAM0:.4f} delta={DELTA} "
      f"T={T} N_c={N_c} R={R} snap spacing={DTS}", flush=True)


def sysres(w, rng):
    c = np.cumsum(w); c /= c[-1]
    return np.searchsorted(c, (rng.uniform()+np.arange(len(w)))/len(w))


def run(lam, seed):
    model, ja, jb = build(L, lam)
    dtau = 1.0/max(2.0*model.alpha*(L-1), 1e-6)
    nw = max(int(round(DTS/dtau)), 1)
    t_start = BURN*T; ngrid = int((T-t_start)/(nw*dtau))
    nburn = int(round(t_start/dtau))
    rng = np.random.default_rng(seed)
    cov = [np.asarray(model.gamma0, float).copy() for _ in range(N_c)]
    orb = [np.asarray(model.orbitals0, complex).copy() for _ in range(N_c)]
    def step(nsteps):
        nonlocal cov, orb
        for _ in range(nsteps):
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
    out = []
    for g in range(ngrid):
        step(nw)
        ob = np.array([observables(c) for c in cov])
        out.append(ob[:, 0].mean()*ob[:, 1].mean())
    return np.array(out)


res = {}
ARMS = os.environ.get("ARMS", "coupled,independent").split(",")
_all = (("coupled", True), ("independent", False))
for tag, coupled in [x for x in _all if x[0] in ARMS]:
    t0 = time.time(); Bm = []
    for rep in range(R):
        row = []
        for li, lam in enumerate(LAMS):
            seed = (990000+911*rep) if coupled else (990000+911*rep+40000*li)
            row.append(run(lam, seed))
        K = min(len(x) for x in row)
        Bm.append(np.array([x[:K] for x in row]))
        if rep % 6 == 0:
            print(f"  {tag} rep {rep}  {time.time()-t0:.0f}s", flush=True)
    Bm = np.array(Bm)                      # (R, 3, K)
    K = Bm.shape[2]
    D = Bm[:, 2, :] - Bm[:, 0, :]          # paired difference per snapshot
    v1 = float(D[:, -1].var(ddof=1))       # single (terminal) snapshot
    Dbar = D.mean(axis=1)
    vK = float(Dbar.var(ddof=1))
    Sig = np.cov(D.T, ddof=1)
    try:
        iS = np.linalg.pinv(Sig); one = np.ones(K)
        vGLS = float(1.0/(one @ iS @ one))
        a = (iS @ one)/(one @ iS @ one)
    except Exception:
        vGLS, a = np.nan, None
    res[tag] = dict(K=K, var_single=v1, var_Kavg=vK, var_GLS=vGLS,
                    D_mean=float(D.mean()), wall=(time.time()-t0)/R,
                    gain_snapshots=v1/vK if vK > 0 else np.nan,
                    weights=[round(float(x), 3) for x in a] if a is not None else None)
    v = res[tag]
    print(f"{tag:12s}: K={K} <D>={v['D_mean']:+.4f}  Var(single)={v1:.5f}  "
          f"Var(K-avg)={vK:.5f}  Var(GLS)={vGLS:.5f}  snapshot gain={v1/vK:.2f}x  "
          f"wall/rep={v['wall']:.1f}s", flush=True)
    if a is not None:
        print(f"              GLS weights: {res[tag]['weights']}", flush=True)

if "coupled" in res and "independent" in res:
    c, i = res["coupled"], res["independent"]
    print("\n--- compounding check ---", flush=True)
    print(f"  coupling gain, single snapshot : {i['var_single']/c['var_single']:.2f}x", flush=True)
    print(f"  coupling gain, K-averaged      : {i['var_Kavg']/c['var_Kavg']:.2f}x", flush=True)
    print(f"  snapshot gain, coupled arm     : {c['gain_snapshots']:.2f}x", flush=True)
    print(f"  snapshot gain, independent arm : {i['gain_snapshots']:.2f}x", flush=True)
    print(f"  TOTAL vs independent+single    : {i['var_single']/c['var_Kavg']:.2f}x", flush=True)
    print(f"  (product of separate gains would be "
          f"{(i['var_single']/c['var_single'])*(c['gain_snapshots']):.2f}x)", flush=True)
else:
    c = res.get("coupled", list(res.values())[0])
Fp = abs(c["D_mean"])/(2*DELTA)
print(f"\n  dD/dlambda ~ {Fp:.3f};  Var(lambda_c) ~ Var(Dbar)/F'^2 = "
      f"{c['var_Kavg']/Fp**2:.3e}  ->  sd(lambda_c) per realisation = "
      f"{np.sqrt(c['var_Kavg'])/Fp:.4f}", flush=True)
json.dump(res, open(f"/tmp/coupsnap_L{L}.json", "w"), indent=1)
print("DONE", flush=True)
