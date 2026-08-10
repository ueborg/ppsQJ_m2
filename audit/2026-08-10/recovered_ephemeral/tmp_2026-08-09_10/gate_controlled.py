"""Gates for the controlled sampler."""
import os, sys, time
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from doob_common import build, KW
from controlled_sampler import controlled_trajectory
from pps_qj.gaussian_backend import gaussian_born_rule_trajectory

L = int(os.environ.get("L", 32)); zeta = float(os.environ.get("ZETA", 0.9))
lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
T = float(os.environ.get("T", 40)); M = int(os.environ.get("M", 24))
a = float(os.environ.get("A", -3.55))
model, ja, jb = build(L, lam)

print("=== GATE 1: a=0 must reproduce the production guided sampler ===")
NT_c, LAM_c, W_c = [], [], []
for s in range(M):
    r = controlled_trajectory(model, T, np.random.default_rng(1000 + s), 0.0,
                              zeta, ja, jb)
    NT_c.append(r["n_acc"]); LAM_c.append(r["Lambda"])
    W_c.append(r["log_w"] + (1 - zeta) * r["Lambda"])   # must be 0 identically
print(f"  max |log_w + (1-zeta)Lambda| at a=0 : {np.max(np.abs(W_c)):.3e}   (exact identity)")
NT_p, LAM_p = [], []
for s in range(M):
    r = gaussian_born_rule_trajectory(model, T, np.random.default_rng(9000 + s),
                                      ja_cached=ja, jb_cached=jb,
                                      proposal_c=zeta, **KW)
    NT_p.append(r.n_jumps); LAM_p.append(r.Lambda)
for nm, c, p in (("N_T", NT_c, NT_p), ("Lambda_T", LAM_c, LAM_p)):
    mc, mp = np.mean(c), np.mean(p)
    se = np.sqrt(np.var(c, ddof=1) / M + np.var(p, ddof=1) / M)
    print(f"  {nm:9s} controlled(a=0) {mc:10.3f} +- {np.std(c,ddof=1)/np.sqrt(M):7.3f} | "
          f"production {mp:10.3f} +- {np.std(p,ddof=1)/np.sqrt(M):7.3f} | "
          f"diff {mc-mp:+8.3f} = {(mc-mp)/max(se,1e-12):+5.2f} sigma")

print("\n=== GATE 2: quadrature convergence of I under dt_max refinement ===")
for dtm in (np.inf, 0.25, 0.1, 0.04):
    lw, ii = [], []
    for s in range(12):
        r = controlled_trajectory(model, T, np.random.default_rng(2000 + s), a,
                                  zeta, ja, jb, dt_max=dtm)
        lw.append(r["log_w"]); ii.append(r["I"])
    print(f"  dt_max={str(dtm):>6s}: <log W_res>={np.mean(lw):9.4f} "
          f"sd={np.std(lw,ddof=1):7.4f}  <I>={np.mean(ii):10.4f}")

print("\n=== GATE 3: same-seed pairing, controlled vs guided baseline ===")
for tag, aa in (("baseline a=0 ", 0.0), ("controlled   ", a)):
    t0 = time.time(); lw, na, nc = [], [], []
    for s in range(M):
        r = controlled_trajectory(model, T, np.random.default_rng(3000 + s), aa,
                                  zeta, ja, jb)
        lw.append(r["log_w"]); na.append(r["n_acc"]); nc.append(r["n_cand"])
    w = np.exp(np.array(lw) - np.max(lw))
    print(f"  {tag} Var(logW)={np.var(lw,ddof=1):9.4f}  ESS/M={ (w.sum()**2/(w**2).sum())/M:6.3f}"
          f"  N_acc={np.mean(na):8.1f}  N_cand={np.mean(nc):8.1f}"
          f"  wall/traj={(time.time()-t0)/M:6.3f}s")
