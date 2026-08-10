"""N_c ladder: controlled (tapered twist, trapezoid) vs production guided.

Equal WALL BUDGET per configuration, not equal NREP.  The question is how far N_c
can be cut under the control before finite-population bias or variance exceeds
production -- that is where a large gain would come from, not the per-trajectory
20-30%.  MEANS are as important as variances here.
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2/analysis/var_reduction")
sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from traj_common import build
from controlled_sampler import controlled_trajectory, observables

ZETA = float(os.environ.get("ZETA", 0.9)); LAM = 0.5 * np.sqrt(ZETA)
AK = float(os.environ.get("AK", -3.46)); A_STAR = -np.log(ZETA) * AK
TAU_K = -AK / (2.0 * LAM)
L = int(os.environ.get("L", 32)); T = float(os.environ.get("T", 64))
BUDGET = float(os.environ.get("BUDGET", 300))     # wall seconds per configuration
model, ja, jb = build(L, LAM)
dtau = 1.0 / max(2.0 * model.alpha * (L - 1), 1e-6)
nstep = int(round(T / dtau))
taper = lambda t: A_STAR * (1.0 - np.exp(-(T - t) / TAU_K))
print(f"N_c LADDER  L={L} zeta={ZETA} T={T} budget={BUDGET}s/config "
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


configs = [("guided", 64, lambda t: 0.0, False),
           ("guided", 32, lambda t: 0.0, False),
           ("ctrl  ", 32, taper, True),
           ("ctrl  ", 24, taper, True),
           ("ctrl  ", 16, taper, True),
           ("ctrl  ", 12, taper, True)]
out = {}
for tag, N_c, afun, tw in configs:
    S, C, G = [], [], []
    t_start = time.time(); rep = 0
    while time.time() - t_start < BUDGET:
        s_, c_, g_ = run(N_c, afun, np.random.default_rng(120000 + 977 * rep), tw)
        S.append(s_); C.append(c_); G.append(g_); rep += 1
    wall_tot = time.time() - t_start
    S, C = np.array(S), np.array(C); BL = S * C
    n = len(S)
    key = f"{tag} N_c={N_c}"
    out[key] = dict(n=n, wall_rep=wall_tot / n, S=float(S.mean()),
                    seS=float(S.std(ddof=1) / np.sqrt(n)), sdS=float(S.std(ddof=1)),
                    C=float(C.mean()), seC=float(C.std(ddof=1) / np.sqrt(n)),
                    sdC=float(C.std(ddof=1)), BL=float(BL.mean()),
                    seBL=float(BL.std(ddof=1) / np.sqrt(n)),
                    sdBL=float(BL.std(ddof=1)), G=float(np.mean(G)))
    v = out[key]
    print(f"{key:16s} n={n:3d} wall/rep={v['wall_rep']:5.1f}s GESS={v['G']:5.2f} | "
          f"S={v['S']:.4f}+-{v['seS']:.4f}  CMI={v['C']:.4f}+-{v['seC']:.4f}  "
          f"B_L={v['BL']:.4f}+-{v['seBL']:.4f}", flush=True)

print("\n--- efficiency E_O = 1/(wall_rep * Var_rep), ratio vs guided N_c=32 ---")
b = out["guided N_c=32"]
for k, v in out.items():
    r = [(b["sd" + o] ** 2 * b["wall_rep"]) / (v["sd" + o] ** 2 * v["wall_rep"])
         for o in ("S", "C", "BL")]
    print(f"  {k:16s}  S={r[0]:5.2f}x  CMI={r[1]:5.2f}x  B_L={r[2]:5.2f}x")
print("\n--- finite-population bias check: means vs guided N_c=64 ---")
ref = out["guided N_c=64"]
for k, v in out.items():
    print("  %-16s " % k + "  ".join(
        f"{o}:{(v[o]-ref[o])/np.sqrt(v['se'+o]**2+ref['se'+o]**2):+5.2f}s"
        for o in ("S", "C", "BL")))
json.dump(out, open(f"/tmp/ncladder_L{L}.json", "w"), indent=1)
