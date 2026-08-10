"""TWISTED controlled cloning (memo 4 sec 2-4) + Simpson refinement of the compensator.

h(Gamma) = exp(a K),  so  exp(a dK_j) = h(J_j Gamma)/h(Gamma): the control is an
h-twisted proposal.  The raw residual increment splits EXACTLY as
    dl_raw = a K_t - a K_u + int_t^u V_h dt,     V_h = a b_K - r + r_hat
so the twisted cloning potential is obtained with NO new quadrature:
    int V_h = dl_raw + a (K_u - K_t).
The discrete per-click term -a sum dK is gone algebraically.  a=0 gives
V_h = -(1-zeta) r, i.e. EXACTLY the production guided-cloning weight.

Population at T then samples the h-twisted ensemble, so observables are untwisted
    <O>_zeta = sum_i O_i exp(-a K_i) / sum_i exp(-a K_i).
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from doob_common import build
from controlled_sampler import controlled_trajectory, observables


def Kof(cov, ja, jb):
    return float(np.clip(0.5 * (1.0 - cov[ja, jb]), 0.0, 1.0).sum())


def sysresample(w, rng):
    N = len(w); c = np.cumsum(w); c /= c[-1]
    return np.searchsorted(c, (rng.uniform() + np.arange(N)) / N)


def cloning(model, ja, jb, zeta, T, N_c, dtau, a, rng, twisted):
    cov = [np.asarray(model.gamma0, float).copy() for _ in range(N_c)]
    orb = [np.asarray(model.orbitals0, complex).copy() for _ in range(N_c)]
    anc = np.arange(N_c)
    ess_h = []
    for _ in range(int(round(T / dtau))):
        lw = np.empty(N_c)
        for i in range(N_c):
            K0 = Kof(cov[i], ja, jb)
            r = controlled_trajectory(model, dtau, rng, a, zeta, ja, jb,
                                      gamma0=cov[i], orbitals0=orb[i])
            cov[i], orb[i] = r["cov"], r["orb"]
            lw[i] = r["log_w"] + (a * (Kof(cov[i], ja, jb) - K0) if twisted else 0.0)
        w = np.exp(lw - lw.max()); w /= w.sum()
        ess_h.append(1.0 / (w ** 2).sum())
        idx = sysresample(w, rng)
        cov = [cov[j].copy() for j in idx]; orb = [orb[j].copy() for j in idx]
        anc = anc[idx]
    S = np.array([observables(c)[0] for c in cov])
    C = np.array([observables(c)[1] for c in cov])
    if twisted and a != 0.0:                       # untwist:  weight by exp(-a K_T)
        u = np.array([-a * Kof(c, ja, jb) for c in cov])
        wv = np.exp(u - u.max()); wv /= wv.sum()
    else:
        wv = np.full(N_c, 1.0 / N_c)
    _, cnt = np.unique(anc, return_counts=True)
    return dict(S=float((wv * S).sum()), CMI=float((wv * C).sum()),
                gess=float(N_c ** 2 / (cnt ** 2).sum()), n_anc=int(len(cnt)),
                ess=float(np.mean(ess_h)),
                ess_untw=float(1.0 / (wv ** 2).sum() / N_c))


L = int(os.environ.get("L", 32)); zeta = float(os.environ.get("ZETA", 0.9))
lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
T = float(os.environ.get("T", 64)); N_c = int(os.environ.get("NC", 32))
aK = float(os.environ.get("AK", -3.46)); NREP = int(os.environ.get("NREP", 10))
a_star = -np.log(zeta) * aK
model, ja, jb = build(L, lam)
dtau = float(os.environ.get("RCHUNK", 1.0)) / max(2.0 * model.alpha * (L - 1), 1e-6)
print(f"L={L} zeta={zeta} T={T} N_c={N_c} dtau={dtau:.5f} a*={a_star:.4f} "
      f"NREP={NREP}  (production resampling interval)", flush=True)
res = {}
arms = (("production guided   ", 0.0, False),
        ("controlled RAW      ", a_star, False),
        ("controlled TWISTED  ", a_star, True))
for tag, a, tw in arms:
    S, C, G, E, W, U = [], [], [], [], [], []
    for rep in range(NREP):
        t0 = time.time()
        r = cloning(model, ja, jb, zeta, T, N_c, dtau, a,
                    np.random.default_rng(77000 + 613 * rep), tw)
        W.append(time.time() - t0)
        S.append(r["S"]); C.append(r["CMI"]); G.append(r["gess"])
        E.append(r["ess"] / N_c); U.append(r["ess_untw"])
    res[tag] = dict(S=float(np.mean(S)), sdS=float(np.std(S, ddof=1)),
                    C=float(np.mean(C)), sdC=float(np.std(C, ddof=1)),
                    gess=float(np.mean(G)), ess=float(np.mean(E)),
                    ess_untw=float(np.mean(U)), wall=float(np.mean(W)))
    print(f"{tag}: <S>={np.mean(S):.4f}+-{np.std(S,ddof=1)/np.sqrt(NREP):.4f} "
          f"<CMI>={np.mean(C):.4f}+-{np.std(C,ddof=1)/np.sqrt(NREP):.4f} "
          f"GESS={np.mean(G):.2f} ESS/N_c={np.mean(E):.3f} "
          f"ESSuntw={np.mean(U):.3f} wall={np.mean(W):.1f}s", flush=True)
b = res["production guided   "]
print("\n--- efficiency ratio vs production guided (>1 = better), equal wall ---")
for tag, _, _ in arms[1:]:
    v = res[tag]
    for ob in ("S", "C"):
        r = (b["sd" + ob] ** 2 * b["wall"]) / (v["sd" + ob] ** 2 * v["wall"])
        print(f"  {tag} {ob:3s}: {r:6.2f}x   (sd {b['sd'+ob]:.4f} -> {v['sd'+ob]:.4f})")
json.dump(res, open(f"/tmp/twisted_L{L}_z{zeta}.json", "w"), indent=1)
