"""Final: is Simpson needed for the ESTIMATOR, or only for the mean-one gate?

Arms (common seeds, paired):
  A  production guided        a=0
  C  tapered twist + Simpson  a(t)=a*[1-exp(-(T-t)/tau_K)]
  D  tapered twist, trapezoid (same weights, cheaper quadrature)
Reports paired-bootstrap efficiency ratios for S, CMI and B_L = <CMI><S>.
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from traj_common import build
from controlled_sampler import controlled_trajectory, observables

ZETA = float(os.environ.get("ZETA", 0.9)); LAM = 0.5 * np.sqrt(ZETA)
AK = float(os.environ.get("AK", -3.46)); A_STAR = -np.log(ZETA) * AK
TAU_K = -AK / (2.0 * LAM)
L = int(os.environ.get("L", 32)); T = float(os.environ.get("T", 64))
N_c = int(os.environ.get("NC", 32)); NREP = int(os.environ.get("NREP", 30))
model, ja, jb = build(L, LAM)
dtau = 1.0 / max(2.0 * model.alpha * (L - 1), 1e-6)
nstep = int(round(T / dtau))
taper = lambda t: A_STAR * (1.0 - np.exp(-(T - t) / TAU_K))
print(f"L={L} T={T} N_c={N_c} NREP={NREP} a*={A_STAR:.4f} tau_K={TAU_K:.2f}", flush=True)


def Kof(cov):
    return float(np.clip(0.5 * (1.0 - cov[ja, jb]), 0.0, 1.0).sum())


def sysres(w, rng):
    c = np.cumsum(w); c /= c[-1]
    return np.searchsorted(c, (rng.uniform() + np.arange(len(w))) / len(w))


def run(afun, rng, twist, simp):
    cov = [np.asarray(model.gamma0, float).copy() for _ in range(N_c)]
    orb = [np.asarray(model.orbitals0, complex).copy() for _ in range(N_c)]
    anc = np.arange(N_c)
    for k in range(nstep):
        ak, au = afun(k * dtau), afun((k + 1) * dtau)
        lw = np.empty(N_c)
        for i in range(N_c):
            K0 = Kof(cov[i])
            r = controlled_trajectory(model, dtau, rng, ak, ZETA, ja, jb,
                                      gamma0=cov[i], orbitals0=orb[i], simpson=simp)
            cov[i], orb[i] = r["cov"], r["orb"]
            lw[i] = r["log_w"] + (au * r["K_end"] - ak * K0 if twist else 0.0)
        w = np.exp(lw - lw.max()); w /= w.sum()
        idx = sysres(w, rng)
        cov = [cov[j].copy() for j in idx]; orb = [orb[j].copy() for j in idx]
        anc = anc[idx]
    S = np.mean([observables(c)[0] for c in cov])
    C = np.mean([observables(c)[1] for c in cov])
    _, ct = np.unique(anc, return_counts=True)
    return float(S), float(C), float(N_c ** 2 / (ct ** 2).sum())


arms = (("A production      ", lambda t: 0.0, False, False),
        ("C taper + Simpson ", taper, True, True),
        ("D taper + trapez. ", taper, True, False))
data = {}
for tag, afun, tw, sp in arms:
    S, C, G, W = [], [], [], []
    for rep in range(NREP):
        t0 = time.time()
        s_, c_, g_ = run(afun, np.random.default_rng(90000 + 733 * rep), tw, sp)
        W.append(time.time() - t0); S.append(s_); C.append(c_); G.append(g_)
    S, C, W = np.array(S), np.array(C), np.array(W)
    data[tag] = dict(S=S.tolist(), C=C.tolist(), BL=(S * C).tolist(),
                     G=float(np.mean(G)), wall=float(np.mean(W)))
    print(f"{tag}: <S>={S.mean():.4f}+-{S.std(ddof=1)/np.sqrt(NREP):.4f} "
          f"<CMI>={C.mean():.4f}+-{C.std(ddof=1)/np.sqrt(NREP):.4f} "
          f"<B_L>={(S*C).mean():.4f}+-{(S*C).std(ddof=1)/np.sqrt(NREP):.4f} "
          f"GESS={np.mean(G):.2f} wall={W.mean():.1f}s", flush=True)

print("\n--- paired bootstrap of E_O = 1/(wall*Var), ratio vs production ---")
rs = np.random.default_rng(7)
base = data["A production      "]
for tag, _, _, _ in arms[1:]:
    v = data[tag]
    for ob in ("S", "C", "BL"):
        b0 = np.array(base[ob]); v0 = np.array(v[ob])
        pt = (b0.var(ddof=1) * base["wall"]) / (v0.var(ddof=1) * v["wall"])
        bs = []
        for _ in range(4000):
            idx = rs.integers(0, NREP, NREP)      # PAIRED resample
            bs.append((b0[idx].var(ddof=1) * base["wall"]) /
                      (v0[idx].var(ddof=1) * v["wall"]))
        lo, hi = np.percentile(bs, [16, 84])
        frac = float(np.mean(np.array(bs) > 1.0))
        print(f"  {tag} {ob:3s}: {pt:5.2f}x  [{lo:.2f},{hi:.2f}]  "
              f"P(>1)={frac:.2f}")
json.dump(data, open(f"/tmp/final_L{L}.json", "w"), indent=1)
