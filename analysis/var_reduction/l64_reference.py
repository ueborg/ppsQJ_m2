"""L=64 arbitration (memo 7): cloning-free reference + likelihood gates.

Gate A : a=0 identity, log W_res == -(1-zeta) Lambda_T, must hold to ~1e-14.
REF    : controlled full-path IS at CONSTANT a=a* with Simpson J-quadrature.
         No cloning population, no twist, no taper -> an independent estimate of
         <S>, <CMI>, <B_L> in the tilted ensemble to compare BOTH cloning arms against.
Gate B : mean-one martingale <R_t> = <dQ_0/dQ_a> at t = 8,16,32,64 (a = 0.5 a*).
         A progressive drift 1.00 -> 0.98 -> 0.94 -> 0.88 indicts accumulated
         compensator error; flat 1.00 clears the likelihood machinery.
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
L = int(os.environ.get("L", 64)); T = float(os.environ.get("T", 64))
M_REF = int(os.environ.get("M_REF", 400)); M_GATE = int(os.environ.get("M_GATE", 250))
model, ja, jb = build(L, LAM)
print(f"L={L} zeta={ZETA} lam={LAM:.4f} T={T} a*={A_STAR:.4f} "
      f"alpha={model.alpha:.4f} M_ref={M_REF}", flush=True)

# ---- Gate A ----
d = []
for k in range(6):
    r = controlled_trajectory(model, T, np.random.default_rng(500 + k), 0.0,
                              ZETA, ja, jb, simpson=False)
    d.append(abs(r["log_w"] + (1 - ZETA) * r["Lambda"]))
print(f"GATE A (a=0 identity, L={L}): max |log W + (1-z)Lambda| = {max(d):.3e}"
      f"   {'PASS' if max(d) < 1e-9 else '*** FAIL ***'}", flush=True)

# ---- controlled path-IS reference ----
t0 = time.time(); lw, S, C = [], [], []
for k in range(M_REF):
    r = controlled_trajectory(model, T, np.random.default_rng(770000 + k), A_STAR,
                              ZETA, ja, jb, simpson=True)
    lw.append(r["log_w"]); s_, c_ = observables(r["cov"]); S.append(s_); C.append(c_)
    if (k + 1) % 100 == 0:
        print(f"   ref {k+1}/{M_REF}  {time.time()-t0:.0f}s", flush=True)
lw = np.array(lw); S = np.array(S); C = np.array(C)
w = np.exp(lw - lw.max()); w /= w.sum()
ess = 1.0 / (w ** 2).sum()
nb = 40; blk = np.array_split(np.arange(M_REF), nb)
jS, jC, jB = [], [], []
for b in blk:
    m = np.ones(M_REF, bool); m[b] = False
    ww = np.exp(lw[m] - lw[m].max()); ww /= ww.sum()
    a_, b_ = (ww * S[m]).sum(), (ww * C[m]).sum()
    jS.append(a_); jC.append(b_); jB.append(a_ * b_)
se = lambda j: float(np.sqrt((nb - 1) / nb * np.sum((np.array(j) - np.mean(j)) ** 2)))
mS, mC = float((w * S).sum()), float((w * C).sum())
print(f"\nREFERENCE controlled path-IS (no cloning), M={M_REF}, ESS={ess:.1f} "
      f"({ess/M_REF:.3f}), wall={time.time()-t0:.0f}s", flush=True)
print(f"  <S>   = {mS:.4f} +- {se(jS):.4f}", flush=True)
print(f"  <CMI> = {mC:.4f} +- {se(jC):.4f}", flush=True)
print(f"  <B_L> = {mS*mC:.4f} +- {se(jB):.4f}", flush=True)
print("  COMPARE: ctrl-clone N_c=32 S=2.2519 CMI=0.4973 B_L=1.1212 | "
      "guided-clone N_c=32 S=2.3379 CMI=0.5370 B_L=1.2586", flush=True)
json.dump(dict(M=M_REF, ess=float(ess), S=mS, seS=se(jS), C=mC, seC=se(jC),
               BL=mS*mC, seBL=se(jB)), open(f"/tmp/l64ref.json", "w"), indent=1)

# ---- Gate B ----
print(f"\nGATE B  mean-one <R_t> at a=0.5a*, Simpson", flush=True)
for tt in (8.0, 16.0, 32.0, 64.0):
    lw2, Lam2 = [], []
    for k in range(M_GATE):
        r = controlled_trajectory(model, tt, np.random.default_rng(880000 + k),
                                  0.5 * A_STAR, ZETA, ja, jb, simpson=True)
        lw2.append(r["log_w"]); Lam2.append(r["Lambda"])
    lg = np.array(lw2) + (1 - ZETA) * np.array(Lam2)
    mx = lg.max(); R = np.exp(lg - mx)
    ER = float(np.exp(mx) * R.mean()); seR = float(np.exp(mx) * R.std(ddof=1) / np.sqrt(M_GATE))
    print(f"  t={tt:5.1f}: <R_t>={ER:.4f} +- {seR:.4f}  ({(ER-1)/max(seR,1e-12):+5.2f}s)"
          f"  sd(logR)={np.std(lg,ddof=1):.3f}", flush=True)
print("DONE", flush=True)
