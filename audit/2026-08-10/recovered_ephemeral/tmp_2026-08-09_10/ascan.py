"""Direct a-scan of the controlled sampler: does it deliver the predicted variance win?

tilt exponent is  a_eff = s * a_K,  s = -log(zeta).  a_eff = 0 is EXACTLY the
production guided scheme (gated to 5e-14).  We scan a_eff rather than assume the
first-order value, since s is finite.
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from doob_common import build
from controlled_sampler import controlled_trajectory, observables

L = int(os.environ.get("L", 64)); zeta = float(os.environ.get("ZETA", 0.9))
lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
T = float(os.environ.get("T", L)); M = int(os.environ.get("M", 40))
aK = float(os.environ.get("AK", -3.55))
s = -np.log(zeta)
a_star = s * aK
grid = [0.0, 0.25 * a_star, 0.5 * a_star, 0.75 * a_star, a_star,
        1.25 * a_star, 1.5 * a_star, 2.0 * a_star]
model, ja, jb = build(L, lam)
print(f"L={L} zeta={zeta} lam={lam:.4f} T={T} M={M}  s={s:.4f}  a_K={aK}  a*=s*a_K={a_star:.4f}")
print(f"{'a_eff':>9} {'Var(logW)':>10} {'ESS/M':>7} {'sd(logW)':>9} "
      f"{'N_acc':>8} {'N_cand':>8} {'wall/traj':>10} {'<S>_SNIS':>9} {'<CMI>_SNIS':>10}")
out = []
for a in grid:
    t0 = time.time(); lw, na, nc, S, C = [], [], [], [], []
    for k in range(M):
        r = controlled_trajectory(model, T, np.random.default_rng(4400 + k), a,
                                  zeta, ja, jb)
        lw.append(r["log_w"]); na.append(r["n_acc"]); nc.append(r["n_cand"])
        sS, cC = observables(r["cov"]); S.append(sS); C.append(cC)
    wall = (time.time() - t0) / M
    lw = np.array(lw); S = np.array(S); C = np.array(C)
    w = np.exp(lw - lw.max()); w /= w.sum()
    ess = 1.0 / (w ** 2).sum()
    out.append(dict(a=a, var=float(np.var(lw, ddof=1)), ess=float(ess / M),
                    n_acc=float(np.mean(na)), n_cand=float(np.mean(nc)),
                    wall=wall, S=float((w * S).sum()), CMI=float((w * C).sum())))
    print(f"{a:9.4f} {np.var(lw,ddof=1):10.4f} {ess/M:7.3f} {np.std(lw,ddof=1):9.4f} "
          f"{np.mean(na):8.1f} {np.mean(nc):8.1f} {wall:10.3f} "
          f"{(w*S).sum():9.4f} {(w*C).sum():10.4f}")
json.dump(dict(L=L, zeta=zeta, lam=lam, T=T, M=M, aK=aK, rows=out),
          open(f"/tmp/ascan_L{L}_z{zeta}.json", "w"), indent=1)
