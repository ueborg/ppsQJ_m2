"""Chunk-lever bias certification (the R>=30 test HANDOFF still owes).

Production `delta_tau` conflates chunk length and resampling period; scaling it by
`mult` is the only remaining low-level wall-time lever (1.6x at zeta=0.9, 2.8x at
zeta=0.5 per HANDOFF 2026-07-27). Those speedups were measured but NEVER certified
unbiased on the GUIDED estimator at production L -- the old dtau test used the
non-guided estimator where the per-window weight was zeta^n rather than the smooth
exp[-(1-zeta) dLambda].

This runs R>=30 realisations at mult in {1,2,4,8} and tests S / CMI / B_L against
mult=1 for a resolved shift, plus the realised wall-time gain.
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
N_c = int(os.environ.get("NC", 32)); R = int(os.environ.get("R", 40))
MULTS = [int(x) for x in os.environ.get("MULTS", "1,2,4,8").split(",")]
model, ja, jb = build(L, LAM)
dtau0 = 1.0/max(2.0*model.alpha*(L-1), 1e-6)
print(f"CHUNK CERT  L={L} zeta={ZETA} T={T} N_c={N_c} R={R} dtau0={dtau0:.5f}",
      flush=True)


def sysres(w, rng):
    c = np.cumsum(w); c /= c[-1]
    return np.searchsorted(c, (rng.uniform()+np.arange(len(w)))/len(w))


def run(mult, rng):
    dtau = dtau0*mult; nstep = int(round(T/dtau))
    cov = [np.asarray(model.gamma0, float).copy() for _ in range(N_c)]
    orb = [np.asarray(model.orbitals0, complex).copy() for _ in range(N_c)]
    anc = np.arange(N_c)
    for k in range(nstep):
        lw = np.empty(N_c)
        for i in range(N_c):
            r = controlled_trajectory(model, dtau, rng, 0.0, ZETA, ja, jb,
                                      gamma0=cov[i], orbitals0=orb[i], simpson=False)
            cov[i], orb[i] = r["cov"], r["orb"]; lw[i] = r["log_w"]
        w = np.exp(lw-lw.max()); w /= w.sum()
        idx = sysres(w, rng)
        cov = [cov[j].copy() for j in idx]; orb = [orb[j].copy() for j in idx]
        anc = anc[idx]
    ob = np.array([observables(c) for c in cov])
    _, ct = np.unique(anc, return_counts=True)
    return (ob[:, 0].mean(), ob[:, 1].mean(), ob[:, 0].mean()*ob[:, 1].mean(),
            N_c**2/(ct**2).sum())


out = {}
for mult in MULTS:
    t0 = time.time(); S, C, B, G = [], [], [], []
    for rep in range(R):
        s_, c_, b_, g_ = run(mult, np.random.default_rng(880000+457*rep))
        S.append(s_); C.append(c_); B.append(b_); G.append(g_)
    S, C, B = np.array(S), np.array(C), np.array(B)
    out[mult] = dict(R=R, wall=(time.time()-t0)/R, GESS=float(np.mean(G)),
                     S=float(S.mean()), seS=float(S.std(ddof=1)/np.sqrt(R)),
                     C=float(C.mean()), seC=float(C.std(ddof=1)/np.sqrt(R)),
                     B=float(B.mean()), seB=float(B.std(ddof=1)/np.sqrt(R)),
                     sdB=float(B.std(ddof=1)))
    v = out[mult]
    print(f"  mult={mult}: wall/rep={v['wall']:6.2f}s GESS={v['GESS']:5.2f} | "
          f"S={v['S']:.4f}+-{v['seS']:.4f} CMI={v['C']:.4f}+-{v['seC']:.4f} "
          f"B_L={v['B']:.4f}+-{v['seB']:.4f}", flush=True)
    json.dump({str(k): v for k, v in out.items()},
              open(f"/tmp/chunkcert_L{L}.json", "w"), indent=1)

b0 = out[MULTS[0]]
print("\n--- bias vs mult=1 (sigma), and realised speedup ---", flush=True)
for mult in MULTS:
    v = out[mult]
    z = lambda k, sk: (v[k]-b0[k])/np.hypot(v[sk], b0[sk]) if mult != MULTS[0] else 0.0
    print(f"  mult={mult}: dS={z('S','seS'):+5.2f}s  dCMI={z('C','seC'):+5.2f}s  "
          f"dB_L={z('B','seB'):+5.2f}s   speedup={b0['wall']/v['wall']:5.2f}x   "
          f"efficiency={(b0['sdB']**2*b0['wall'])/(v['sdB']**2*v['wall']):5.2f}x",
          flush=True)
print("DONE", flush=True)
