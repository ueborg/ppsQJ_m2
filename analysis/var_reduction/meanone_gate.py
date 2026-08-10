"""Memo-5 sequence: (1) mean-one gate with Simpson, (2) three-arm tapered-twist test."""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from traj_common import build
from controlled_sampler import controlled_trajectory, observables, tilt_factors

ZETA = float(os.environ.get("ZETA", 0.9))
LAM = float(os.environ.get("LAM", 0.5 * np.sqrt(ZETA)))
AK = float(os.environ.get("AK", -3.46))
A_STAR = -np.log(ZETA) * AK
TAU_K = -AK / (2.0 * LAM)          # one-mode estimate, NOT fitted
MODE = os.environ.get("MODE", "gate")


def Kof(cov, ja, jb):
    return float(np.clip(0.5 * (1.0 - cov[ja, jb]), 0.0, 1.0).sum())


if MODE == "gate":
    L = int(os.environ.get("L", 16)); T = float(os.environ.get("T", 32))
    M = int(os.environ.get("M", 3000))
    model, ja, jb = build(L, LAM)
    print(f"MEAN-ONE GATE  L={L} zeta={ZETA} T={T} M={M} a*={A_STAR:.4f} "
          f"tau_K={TAU_K:.2f}", flush=True)
    for frac in (0.5, 1.0):
        for simp in (False, True):
            a = frac * A_STAR
            lw, Lam, S, C = [], [], [], []
            t0 = time.time()
            for k in range(M):
                r = controlled_trajectory(model, T, np.random.default_rng(64000+k),
                                          a, ZETA, ja, jb, simpson=simp)
                lw.append(r["log_w"]); Lam.append(r["Lambda"])
                s_, c_ = observables(r["cov"]); S.append(s_); C.append(c_)
            lw = np.array(lw); Lam = np.array(Lam)
            logR = lw + (1 - ZETA) * Lam
            mx = logR.max(); R = np.exp(logR - mx)
            ER = float(np.exp(mx) * R.mean())
            se = float(np.exp(mx) * R.std(ddof=1) / np.sqrt(M))
            w = np.exp(lw - lw.max()); w /= w.sum()
            print(f"  a/a*={frac:.1f} simpson={str(simp):5s}: E[R]={ER:.5f}+-{se:.5f} "
                  f"({(ER-1)/max(se,1e-12):+5.2f}s)  sd(logR)={np.std(logR,ddof=1):.3f} "
                  f"<CMI>={(w*np.array(C)).sum():.5f} ESS/M={1/(w**2).sum()/M:.3f} "
                  f"wall={(time.time()-t0)/M:.3f}s", flush=True)
    sys.exit()

# ---------------- three-arm cloning ----------------
L = int(os.environ.get("L", 32)); T = float(os.environ.get("T", 64))
N_c = int(os.environ.get("NC", 32)); NREP = int(os.environ.get("NREP", 30))
model, ja, jb = build(L, LAM)
dtau = 1.0 / max(2.0 * model.alpha * (L - 1), 1e-6)
nstep = int(round(T / dtau))
taper = lambda t: A_STAR * (1.0 - np.exp(-(T - t) / TAU_K))
print(f"THREE-ARM  L={L} zeta={ZETA} T={T} N_c={N_c} dtau={dtau:.5f} "
      f"a*={A_STAR:.4f} tau_K={TAU_K:.2f} NREP={NREP}", flush=True)


def sysres(w, rng):
    c = np.cumsum(w); c /= c[-1]
    return np.searchsorted(c, (rng.uniform() + np.arange(len(w))) / len(w))


def run(afun, rng, twist):
    cov = [np.asarray(model.gamma0, float).copy() for _ in range(N_c)]
    orb = [np.asarray(model.orbitals0, complex).copy() for _ in range(N_c)]
    anc = np.arange(N_c); gess_t = []
    for k in range(nstep):
        tk, tu = k * dtau, (k + 1) * dtau
        ak, au = afun(tk), afun(tu)
        lw = np.empty(N_c)
        for i in range(N_c):
            K0 = Kof(cov[i], ja, jb)
            r = controlled_trajectory(model, dtau, rng, ak, ZETA, ja, jb,
                                      gamma0=cov[i], orbitals0=orb[i])
            cov[i], orb[i] = r["cov"], r["orb"]
            lw[i] = r["log_w"] + (au * r["K_end"] - ak * K0 if twist else 0.0)
        w = np.exp(lw - lw.max()); w /= w.sum()
        idx = sysres(w, rng)
        cov = [cov[j].copy() for j in idx]; orb = [orb[j].copy() for j in idx]
        anc = anc[idx]
        if k % 100 == 0:
            _, ct = np.unique(anc, return_counts=True)
            gess_t.append(float(N_c ** 2 / (ct ** 2).sum()))
    S = np.array([observables(c)[0] for c in cov])
    C = np.array([observables(c)[1] for c in cov])
    _, ct = np.unique(anc, return_counts=True)
    return (float(S.mean()), float(C.mean()),
            float(N_c ** 2 / (ct ** 2).sum()), gess_t)


arms = (("A production guided", lambda t: 0.0, False),
        ("B constant twisted ", lambda t: A_STAR, True),
        ("C tapered twisted  ", taper, True))
res = {}
for tag, afun, tw in arms:
    S, C, G, W, GT = [], [], [], [], []
    for rep in range(NREP):
        t0 = time.time()
        s_, c_, g_, gt = run(afun, np.random.default_rng(90000 + 733 * rep), tw)
        W.append(time.time() - t0); S.append(s_); C.append(c_); G.append(g_); GT.append(gt)
    res[tag] = dict(S=float(np.mean(S)), sdS=float(np.std(S, ddof=1)),
                    C=float(np.mean(C)), sdC=float(np.std(C, ddof=1)),
                    gess=float(np.mean(G)), wall=float(np.mean(W)),
                    gess_t=[float(x) for x in np.mean(GT, axis=0)])
    print(f"{tag}: <S>={np.mean(S):.4f}+-{np.std(S,ddof=1)/np.sqrt(NREP):.4f} "
          f"<CMI>={np.mean(C):.4f}+-{np.std(C,ddof=1)/np.sqrt(NREP):.4f} "
          f"GESS={np.mean(G):.2f} wall={np.mean(W):.1f}s", flush=True)
    print(f"   GESS(t) @ t=0,~5,~10,...: "
          f"{[round(x,2) for x in np.mean(GT,axis=0)[:8]]}", flush=True)
b = res["A production guided"]
print("\n--- efficiency ratio vs production (>1 better), equal wall ---")
for tag, _, _ in arms[1:]:
    v = res[tag]
    for ob in ("S", "C"):
        print(f"  {tag} {ob}: "
              f"{(b['sd'+ob]**2*b['wall'])/(v['sd'+ob]**2*v['wall']):6.2f}x"
              f"   (sd {b['sd'+ob]:.4f} -> {v['sd'+ob]:.4f})")
json.dump(res, open(f"/tmp/threearm_L{L}.json", "w"), indent=1)
