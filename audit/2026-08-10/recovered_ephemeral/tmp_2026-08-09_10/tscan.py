"""(a) T-scaling: baseline Var(logW) ~ T, controlled should be O(1) -> gain grows with T.
   (b) quadrature gate for I at the CORRECT a_eff.
   (c) end-to-end exactness: controlled path-IS <S>,<CMI> vs production guided cloning.
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

L = int(os.environ.get("L", 32)); zeta = float(os.environ.get("ZETA", 0.9))
lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
aK = float(os.environ.get("AK", -3.46))
a_star = -np.log(zeta) * aK
model, ja, jb = build(L, lam)


def batch(T, M, a, dt_max=np.inf, seed0=6000):
    lw, S, C, na, nc = [], [], [], [], []
    t0 = time.time()
    for k in range(M):
        r = controlled_trajectory(model, T, np.random.default_rng(seed0 + k), a,
                                  zeta, ja, jb, dt_max=dt_max)
        lw.append(r["log_w"]); na.append(r["n_acc"]); nc.append(r["n_cand"])
        sS, cC = observables(r["cov"]); S.append(sS); C.append(cC)
    lw = np.array(lw); w = np.exp(lw - lw.max()); w /= w.sum()
    return dict(var=float(np.var(lw, ddof=1)), ess=float(1/(w**2).sum()/M),
                S=float((w*np.array(S)).sum()), CMI=float((w*np.array(C)).sum()),
                n_acc=float(np.mean(na)), n_cand=float(np.mean(nc)),
                wall=(time.time()-t0)/M)


print("=== (a) T-scaling of Var(log W), L=%d zeta=%.2f ===" % (L, zeta))
print(f"{'T':>5} {'Var base':>9} {'Var ctrl':>9} {'gain':>7} "
      f"{'ESS base':>9} {'ESS ctrl':>9} {'wall base':>10} {'wall ctrl':>10}")
rowsT = []
for T in (16.0, 32.0, 64.0, 128.0):
    b = batch(T, 48, 0.0); c = batch(T, 48, a_star)
    rowsT.append(dict(T=T, base=b, ctrl=c))
    print(f"{T:5.0f} {b['var']:9.3f} {c['var']:9.3f} {b['var']/c['var']:7.2f} "
          f"{b['ess']:9.3f} {c['ess']:9.3f} {b['wall']:10.3f} {c['wall']:10.3f}")

print("\n=== (b) quadrature gate for I at a_eff=%.4f (T=64) ===" % a_star)
for dtm in (np.inf, 0.2, 0.05):
    r = batch(64.0, 24, a_star, dt_max=dtm, seed0=7100)
    print(f"  dt_max={str(dtm):>5s}: Var(logW)={r['var']:8.4f}  <S>={r['S']:8.4f} "
          f" <CMI>={r['CMI']:8.4f}  wall={r['wall']:6.3f}s")

print("\n=== (c) end-to-end exactness vs production guided cloning (T=64) ===")
M = int(os.environ.get("MBIG", 400))
c = batch(64.0, M, a_star, seed0=8100)
print(f"  controlled path-IS  M={M}:  <S>={c['S']:.4f}  <CMI>={c['CMI']:.4f} "
      f" ESS={c['ess']*M:.1f}  wall_total={c['wall']*M:.1f}s")
t0 = time.time()
Ss, Cs = [], []
for k in range(int(os.environ.get("NREAL", 6))):
    res = run_cloning(model, zeta, 64.0, 64, np.random.default_rng(8800+k),
                      proposal_c=zeta, jump_update_method="lowrank",
                      solver_method="newton", record_entropy=True)
    Ss.append(np.mean(res.entropy_history[-1]) if hasattr(res, "entropy_history") else np.nan)
print("  cloning wall_total=%.1fs" % (time.time()-t0))
json.dump(dict(L=L, zeta=zeta, lam=lam, aK=aK, a_star=a_star, T_scan=rowsT),
          open(f"/tmp/tscan_L{L}_z{zeta}.json", "w"), indent=1, default=str)
