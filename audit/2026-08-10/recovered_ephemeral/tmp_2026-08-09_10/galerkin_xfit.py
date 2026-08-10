"""Leave-one-trajectory-out cross-fitted Gain from a saved galerkin npz."""
import sys, numpy as np

d = np.load(sys.argv[1])
Gp, Rr = d["GPHI"], d["R"]
L = int(d["L"]); zeta = float(d["zeta"]); delta = float(d["delta"]); burn = int(d["burn"])
ntraj = Rr.shape[0]
rbar = Rr[:, burn:].mean()
dr = Rr - rbar
s_val = -np.log(zeta)
print(f"L={L} zeta={zeta} ntraj={ntraj} delta={delta} s=-log(zeta)={s_val:.4f}")

for s_blk in (8.0, 16.0, 32.0):
    m = int(round(s_blk / delta))
    nav = Rr.shape[1] - burn
    nblk = nav // m
    if nblk < 2:
        continue
    yl, Gl = [], []
    for i in range(ntraj):
        sl = slice(burn, burn + nblk * m)
        yl.append(dr[i, sl].reshape(nblk, m).sum(1) * delta)
        Gl.append(Gp[i, sl].reshape(nblk, m, Gp.shape[-1]).sum(1) * delta)
    y_all = np.concatenate(yl)
    v0 = y_all.var(ddof=1)

    def xfit(cols):
        res = []
        for i in range(ntraj):
            Xtr = np.vstack([Gl[j][:, cols] for j in range(ntraj) if j != i])
            ytr = np.concatenate([yl[j] for j in range(ntraj) if j != i])
            Dtr = np.column_stack([np.ones(len(ytr)), Xtr])
            b, *_ = np.linalg.lstsq(Dtr, ytr, rcond=None)
            Dte = np.column_stack([np.ones(len(yl[i])), Gl[i][:, cols]])
            res.append(yl[i] - Dte @ b)
        r = np.concatenate(res)
        return v0 / r.var(ddof=1), b

    gK, bK = xfit([0])
    gA, bA = xfit([0, 1, 2])
    # bootstrap over trajectories for an error bar on Gain(K)
    rs = np.random.default_rng(3)
    boots = []
    for _ in range(200):
        idx = rs.integers(0, ntraj, ntraj)
        yb = np.concatenate([yl[i] for i in idx])
        Gb = np.vstack([Gl[i][:, [0]] for i in idx])
        D = np.column_stack([np.ones(len(yb)), Gb])
        b, *_ = np.linalg.lstsq(D, yb, rcond=None)
        boots.append(yb.var(ddof=1) / (yb - D @ b).var(ddof=1))
    lo, hi = np.percentile(boots, [16, 84])
    print(f"  s_blk={s_blk:5.1f} nblk={len(y_all):4d}  sigma0^2={v0/s_blk:7.3f}"
          f"  Gain_K(xfit)={gK:6.2f} [{lo:.1f},{hi:.1f}]"
          f"  Gain_all3(xfit)={gA:6.2f}  a_K={bK[1]:+.3f}")
    if s_blk == 16.0:
        varlogW_T_L = s_val ** 2 * (v0 / s_blk) * L
        print(f"     -> predicted Var(logW) over T=L: {varlogW_T_L:6.2f} (c=zeta)"
              f"  ->  {varlogW_T_L/gK:6.3f} (controlled, 1st order in s)")
