"""BLOCKER gates (memo 4, sections 7-8).

(1) Mean-one likelihood-ratio martingale.  Between the two NORMALISED proposal
    processes Q_0 (rates zeta*r_j) and Q_a (rates zeta*r_j*exp(a dK)),
        log dQ_0/dQ_a = -a sum dK - zeta*Lambda + int r_hat  =  log_w + (1-zeta)*Lambda.
    E_{Q_a}[R_t] = 1 EXACTLY at every t, with no tilted partition function.
    Fails  -> bug in the control / compensator / quadrature.
    Passes -> the 9% CMI shift is finite-sample SNIS bias, not a correctness bug.

(2) a-invariance of the estimator.  All a target the SAME zeta-ensemble, so
    <CMI>_zeta and <S>_zeta must be independent of a.  Run at small L with huge M
    to push the statistical error well below the observed 9%.
"""
import os, sys, time, json
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/Users/catlover1337/Documents/ppsQJ_m2")
from doob_common import build
from controlled_sampler import controlled_trajectory, observables

L = int(os.environ.get("L", 16)); zeta = float(os.environ.get("ZETA", 0.9))
lam = float(os.environ.get("LAM", 0.5 * np.sqrt(zeta)))
T = float(os.environ.get("T", 32)); M = int(os.environ.get("M", 3000))
aK = float(os.environ.get("AK", -3.46))
a_star = -np.log(zeta) * aK
model, ja, jb = build(L, lam)
print(f"L={L} zeta={zeta} lam={lam:.4f} T={T} M={M} a*={a_star:.4f}", flush=True)
print(f"{'a/a*':>6} {'E[R_T]':>10} {'+-':>8} {'sigma':>7} {'<S>':>9} {'+-':>7} "
      f"{'<CMI>':>9} {'+-':>7} {'ESS/M':>7} {'N_acc':>8}", flush=True)

rows = []
for frac in (0.0, 0.5, 1.0, 1.5):
    a = frac * a_star
    lw, Lam, S, C, NA = [], [], [], [], []
    t0 = time.time()
    for k in range(M):
        r = controlled_trajectory(model, T, np.random.default_rng(64000 + k), a,
                                  zeta, ja, jb)
        lw.append(r["log_w"]); Lam.append(r["Lambda"]); NA.append(r["n_acc"])
        s_, c_ = observables(r["cov"]); S.append(s_); C.append(c_)
    lw = np.array(lw); Lam = np.array(Lam); S = np.array(S); C = np.array(C)
    logR = lw + (1 - zeta) * Lam
    R = np.exp(logR - logR.max())
    ER = float(np.exp(logR.max()) * R.mean())
    seR = float(np.exp(logR.max()) * R.std(ddof=1) / np.sqrt(M))
    w = np.exp(lw - lw.max()); w /= w.sum()
    ess = 1.0 / (w ** 2).sum()
    mS = float((w * S).sum()); mC = float((w * C).sum())
    # jackknife-over-blocks SE for the SNIS means
    nb = 40; blk = np.array_split(np.arange(M), nb)
    jS, jC = [], []
    for b in blk:
        m = np.ones(M, bool); m[b] = False
        ww = np.exp(lw[m] - lw[m].max()); ww /= ww.sum()
        jS.append((ww * S[m]).sum()); jC.append((ww * C[m]).sum())
    seS = float(np.sqrt((nb - 1) / nb * np.sum((np.array(jS) - np.mean(jS)) ** 2)))
    seC = float(np.sqrt((nb - 1) / nb * np.sum((np.array(jC) - np.mean(jC)) ** 2)))
    rows.append(dict(frac=frac, a=a, ER=ER, seR=seR, S=mS, seS=seS, C=mC, seC=seC,
                     ess=float(ess / M), n_acc=float(np.mean(NA)),
                     var_logw=float(np.var(lw, ddof=1)), wall=time.time() - t0))
    print(f"{frac:6.2f} {ER:10.5f} {seR:8.5f} {np.std(logR,ddof=1):7.3f} "
          f"{mS:9.5f} {seS:7.5f} {mC:9.5f} {seC:7.5f} {ess/M:7.3f} "
          f"{np.mean(NA):8.1f}", flush=True)

print("\n--- a-invariance test (all rows must agree) ---")
b = rows[0]
for r in rows[1:]:
    for ob in ("S", "C"):
        d = r[ob] - b[ob]
        se = np.sqrt(r["se" + ob] ** 2 + b["se" + ob] ** 2)
        print(f"  a/a*={r['frac']:.1f}  {ob}: diff={d:+9.5f}  "
              f"{d/max(se,1e-12):+6.2f} sigma")
print("\n--- mean-one test (all rows must be 1.000) ---")
for r in rows:
    print(f"  a/a*={r['frac']:.1f}  E[R_T]={r['ER']:.5f} +- {r['seR']:.5f}  "
          f"-> {(r['ER']-1)/max(r['seR'],1e-12):+6.2f} sigma")
json.dump(rows, open(f"/tmp/blocker_L{L}_z{zeta}.json", "w"), indent=1)
