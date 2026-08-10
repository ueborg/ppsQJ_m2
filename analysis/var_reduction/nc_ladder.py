"""N_c ladder for guided vs tapered-twist controlled cloning.

Equal WALL BUDGET per configuration (not equal NREP). Primary bias diagnostics are
CMI and B_L; S is a consistency check only (it does not show the same clean
convergence -- see HANDOFF 2026-08-09).  Results stream to stdout so a partial run
is still usable.

env: L, ZETA, T, AK, CONFIGS ("algo:N_c:budget_s,..."), OUT
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

ZETA = float(os.environ.get("ZETA", 0.9)); LAM = 0.5 * np.sqrt(ZETA)
AK = float(os.environ.get("AK", -3.57)); A_STAR = -np.log(ZETA) * AK
L = int(os.environ.get("L", 64)); T = float(os.environ.get("T", L))
TAU_K = -AK / (2.0 * LAM)
OUT = os.environ.get("OUT", f"/tmp/ladder_L{L}.json")
SPEC = os.environ.get("CONFIGS",
                      "ctrl:32:1500,guided:32:1500,guided:64:1500,"
                      "ctrl:24:1500,guided:128:3000")
model, ja, jb = build(L, LAM)
dtau = 1.0 / max(2.0 * model.alpha * (L - 1), 1e-6)
nstep = int(round(T / dtau))
taper = lambda t: A_STAR * (1.0 - np.exp(-(T - t) / TAU_K))
print(f"LADDER L={L} zeta={ZETA} lam={LAM:.4f} T={T} dtau={dtau:.5f} "
      f"a*={A_STAR:.4f} tau_K={TAU_K:.2f}", flush=True)


def Kof(cov):
    return float(np.clip(0.5 * (1.0 - cov[ja, jb]), 0.0, 1.0).sum())


def sysres(w, rng):
    c = np.cumsum(w); c /= c[-1]
    return np.searchsorted(c, (rng.uniform() + np.arange(len(w))) / len(w))


def run(N_c, afun, rng, twist):
    cov = [np.asarray(model.gamma0, float).copy() for _ in range(N_c)]
    orb = [np.asarray(model.orbitals0, complex).copy() for _ in range(N_c)]
    anc = np.arange(N_c)
    for k in range(nstep):
        ak, au = afun(k * dtau), afun((k + 1) * dtau)
        lw = np.empty(N_c)
        for i in range(N_c):
            K0 = Kof(cov[i])
            r = controlled_trajectory(model, dtau, rng, ak, ZETA, ja, jb,
                                      gamma0=cov[i], orbitals0=orb[i], simpson=False)
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


out = {}
for item in SPEC.split(","):
    algo, N_c, budget = item.split(":"); N_c = int(N_c); budget = float(budget)
    afun = (lambda t: 0.0) if algo == "guided" else taper
    tw = algo == "ctrl"
    S, C, G = [], [], []
    t0 = time.time(); rep = 0
    while time.time() - t0 < budget:
        s_, c_, g_ = run(N_c, afun, np.random.default_rng(310000 + 1013 * rep), tw)
        S.append(s_); C.append(c_); G.append(g_); rep += 1
    wall = (time.time() - t0) / rep
    S, C = np.array(S), np.array(C); BL = S * C; n = len(S)
    key = f"{algo} N_c={N_c}"
    out[key] = dict(n=n, wall_rep=wall, G=float(np.mean(G)),
                    S=float(S.mean()), seS=float(S.std(ddof=1)/np.sqrt(n)),
                    sdS=float(S.std(ddof=1)),
                    C=float(C.mean()), seC=float(C.std(ddof=1)/np.sqrt(n)),
                    sdC=float(C.std(ddof=1)),
                    BL=float(BL.mean()), seBL=float(BL.std(ddof=1)/np.sqrt(n)),
                    sdBL=float(BL.std(ddof=1)))
    v = out[key]
    print(f"{key:16s} n={n:3d} wall/rep={v['wall_rep']:6.1f}s GESS={v['G']:5.2f} | "
          f"S={v['S']:.4f}+-{v['seS']:.4f}  CMI={v['C']:.4f}+-{v['seC']:.4f}  "
          f"B_L={v['BL']:.4f}+-{v['seBL']:.4f}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
print("\nDONE", flush=True)
