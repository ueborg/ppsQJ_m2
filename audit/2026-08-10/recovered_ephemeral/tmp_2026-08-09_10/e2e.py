"""End-to-end exactness + cost comparison: controlled path-IS vs production guided cloning.

Both estimate the SAME tilted expectations <S_{L/2}>_zeta and <CMI>_zeta.
Equal-walltime comparison with error bars from independent repeats.
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from doob_common import build
from controlled_sampler import controlled_trajectory, observables
from pps_qj.cloning import run_cloning
from pps_qj.gaussian_backend import topological_entanglement_entropy

L = int(os.environ.get("L", 32)); zeta = float(os.environ.get("ZETA", 0.9))
lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
T = float(os.environ.get("T", 64)); aK = float(os.environ.get("AK", -3.46))
NREP = int(os.environ.get("NREP", 8))
M = int(os.environ.get("M", 64))       # paths per controlled repeat
NC = int(os.environ.get("NC", 32))     # clones per cloning repeat
a_star = -np.log(zeta) * aK
model, ja, jb = build(L, lam)
print(f"L={L} zeta={zeta} lam={lam:.4f} T={T} a*={a_star:.4f} "
      f"NREP={NREP} M={M} N_c={NC}", flush=True)


def ctrl_repeat(a, seed0):
    lw, S, C = [], [], []
    t0 = time.time()
    for k in range(M):
        r = controlled_trajectory(model, T, np.random.default_rng(seed0 + k), a,
                                  zeta, ja, jb)
        lw.append(r["log_w"]); sS, cC = observables(r["cov"])
        S.append(sS); C.append(cC)
    lw = np.array(lw); w = np.exp(lw - lw.max()); w /= w.sum()
    return ((w*np.array(S)).sum(), (w*np.array(C)).sum(),
            1/(w**2).sum(), time.time()-t0)


def clone_repeat(seed):
    t0 = time.time()
    res = run_cloning(model, zeta, T, NC, np.random.default_rng(seed),
                      proposal_c=zeta, jump_update_method="lowrank",
                      solver_method="newton", record_entropy=True)
    cmi = float(np.mean([topological_entanglement_entropy(c) for c in res.final_covs]))
    return res.S_mean, cmi, res.eff_sample_size, time.time()-t0


rows = {}
for tag, fn, args in (("guided path-IS (a=0)", ctrl_repeat, 0.0),
                      ("CONTROLLED path-IS ", ctrl_repeat, a_star)):
    S, C, E, W = [], [], [], []
    for rep in range(NREP):
        s, c, e, wl = fn(args, 20000 + 1000*rep)
        S.append(s); C.append(c); E.append(e); W.append(wl)
    rows[tag] = dict(S=np.mean(S), sdS=np.std(S, ddof=1), C=np.mean(C),
                     sdC=np.std(C, ddof=1), ESS=np.mean(E), wall=np.mean(W))
    print(f"{tag}: <S>={np.mean(S):.4f}+-{np.std(S,ddof=1)/np.sqrt(NREP):.4f} "
          f"<CMI>={np.mean(C):.4f}+-{np.std(C,ddof=1)/np.sqrt(NREP):.4f} "
          f"ESS={np.mean(E):.1f}/{M}  wall/rep={np.mean(W):.1f}s", flush=True)

S, C, E, W = [], [], [], []
for rep in range(NREP):
    s, c, e, wl = clone_repeat(31000 + 97*rep)
    S.append(s); C.append(c); E.append(e); W.append(wl)
rows["guided CLONING      "] = dict(S=np.mean(S), sdS=np.std(S, ddof=1),
                                    C=np.mean(C), sdC=np.std(C, ddof=1),
                                    ESS=np.mean(E), wall=np.mean(W))
print(f"guided CLONING      : <S>={np.mean(S):.4f}+-{np.std(S,ddof=1)/np.sqrt(NREP):.4f} "
      f"<CMI>={np.mean(C):.4f}+-{np.std(C,ddof=1)/np.sqrt(NREP):.4f} "
      f"ESS={np.mean(E):.1f}/{NC}  wall/rep={np.mean(W):.1f}s", flush=True)

print("\n--- equal-walltime efficiency:  1 / (Var(estimator) * wall) ---")
base = None
for k, v in rows.items():
    for ob, sd in (("S", v["sdS"]), ("CMI", v["sdC"])):
        eff = 1.0 / (sd**2 * v["wall"])
        print(f"  {k}  {ob:3s}: sd={sd:.4f} wall={v['wall']:6.1f}s  eff={eff:10.4f}")
json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in rows.items()},
          open(f"/tmp/e2e_L{L}_z{zeta}.json", "w"), indent=1)
