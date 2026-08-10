"""Controlled CLONING vs production guided cloning, equal wall time.

Path-IS (previous script) does NOT beat cloning on estimator variance, because
cloning already fixes per-window degeneracy by resampling.  The right use of the
control is INSIDE cloning (memo 2 sec 11: controlled dynamics + residual cloning):
the reduced weight variance should buy genealogical diversity (GESS) and wall time.

a_eff = 0 reproduces the production guided-cloning algorithm exactly (the segment
weight is then exp[-(1-zeta) dLambda], gated to 5e-14).
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from doob_common import build
from controlled_sampler import controlled_trajectory, observables


def systematic_resample(w, rng):
    N = len(w); c = np.cumsum(w); c /= c[-1]
    u = (rng.uniform() + np.arange(N)) / N
    return np.searchsorted(c, u)


def controlled_cloning(model, ja, jb, zeta, T, N_c, dtau, a, rng):
    L = model.gamma0.shape[0] // 2
    cov = [np.asarray(model.gamma0, float).copy() for _ in range(N_c)]
    orb = [np.asarray(model.orbitals0, complex).copy() for _ in range(N_c)]
    anc = np.arange(N_c)
    nsteps = int(round(T / dtau))
    ess_hist = []
    for k in range(nsteps):
        lw = np.empty(N_c)
        for i in range(N_c):
            r = controlled_trajectory(model, dtau, rng, a, zeta, ja, jb,
                                      gamma0=cov[i], orbitals0=orb[i])
            cov[i], orb[i], lw[i] = r["cov"], r["orb"], r["log_w"]
        w = np.exp(lw - lw.max()); w /= w.sum()
        ess_hist.append(1.0 / (w ** 2).sum())
        idx = systematic_resample(w, rng)
        cov = [cov[j].copy() for j in idx]
        orb = [orb[j].copy() for j in idx]
        anc = anc[idx]
    S = np.array([observables(c)[0] for c in cov])
    C = np.array([observables(c)[1] for c in cov])
    _, cnt = np.unique(anc, return_counts=True)
    gess = N_c ** 2 / (cnt ** 2).sum()
    return dict(S=float(S.mean()), CMI=float(C.mean()),
                gess=float(gess), n_anc=int(len(cnt)),
                ess=float(np.mean(ess_hist)),
                coal=float(np.sum(1.0 / np.array(ess_hist))))


L = int(os.environ.get("L", 32)); zeta = float(os.environ.get("ZETA", 0.9))
lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
T = float(os.environ.get("T", 64)); N_c = int(os.environ.get("NC", 32))
aK = float(os.environ.get("AK", -3.46)); NREP = int(os.environ.get("NREP", 8))
a_star = -np.log(zeta) * aK
model, ja, jb = build(L, lam)
dtau = float(os.environ.get("RCHUNK", 1.0)) / max(2.0 * model.alpha * (L - 1), 1e-6)
print(f"L={L} zeta={zeta} T={T} N_c={N_c} dtau={dtau:.5f} a*={a_star:.4f} NREP={NREP}",
      flush=True)
res = {}
for tag, a in (("guided cloning (a=0)", 0.0), ("CONTROLLED cloning  ", a_star)):
    S, C, G, E, W, A = [], [], [], [], [], []
    for rep in range(NREP):
        t0 = time.time()
        r = controlled_cloning(model, ja, jb, zeta, T, N_c, dtau, a,
                               np.random.default_rng(51000 + 811 * rep))
        W.append(time.time() - t0)
        S.append(r["S"]); C.append(r["CMI"]); G.append(r["gess"])
        E.append(r["ess"] / N_c); A.append(r["n_anc"])
    res[tag] = dict(S=float(np.mean(S)), sdS=float(np.std(S, ddof=1)),
                    C=float(np.mean(C)), sdC=float(np.std(C, ddof=1)),
                    gess=float(np.mean(G)), ess=float(np.mean(E)),
                    anc=float(np.mean(A)), wall=float(np.mean(W)))
    res[tag]["sd" + "S"] = float(np.std(S, ddof=1))
    res[tag]["sd" + "CMI"] = float(np.std(C, ddof=1))
    print(f"{tag}: <S>={np.mean(S):.4f}+-{np.std(S,ddof=1)/np.sqrt(NREP):.4f} "
          f"<CMI>={np.mean(C):.4f}+-{np.std(C,ddof=1)/np.sqrt(NREP):.4f} "
          f"GESS={np.mean(G):.2f} n_anc={np.mean(A):.1f} ESS/N_c={np.mean(E):.3f} "
          f"wall={np.mean(W):.1f}s", flush=True)
print("\n--- equal-walltime efficiency 1/(Var * wall) ---")
for k, v in res.items():
    for ob, sd in (("S", v["sdS"]), ("CMI", v["sdC"])):
        print(f"  {k} {ob:3s}: sd={sd:.4f} wall={v['wall']:6.1f}s "
              f"eff={1.0/(sd**2*v['wall']):10.3f}")
b, c = res["guided cloning (a=0)"], res["CONTROLLED cloning  "]
for ob in ("S", "CMI"):
    r = (b["sd" + ob] ** 2 * b["wall"]) / (c["sd" + ob] ** 2 * c["wall"])
    print(f"  SPEEDUP at equal precision, {ob}: {r:.2f}x")
json.dump(res, open(f"/tmp/ccloning_L{L}_z{zeta}.json", "w"), indent=1)
