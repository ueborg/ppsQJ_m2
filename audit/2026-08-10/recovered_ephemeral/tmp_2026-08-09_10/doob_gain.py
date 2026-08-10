"""Round 2 for memos 2/3: the actual figure of merit G = sigma_0^2 / sigma_res^2.

Poisson-equation / Doob control variate.  If g solves -Gg = delta_r then
    int_0^s delta_r dt = g(Gamma_0) - g(Gamma_s) + M_s
with M_s a martingale, so subtracting the boundary term of an APPROXIMATE
g-hat removes exactly the part of the Lambda fluctuation that g-hat captures.
This is the O(s) equivalent of the first-order Doob change of measure
(memo 2 sec 7-8, sec 16), and it needs no new simulation.

    G = Var(dLambda_block) / Var(dLambda_block - a.[X(Gamma_end) - X(Gamma_start)])

Coefficients a are CROSS-FITTED (leave-one-trajectory-out) so the reported G is
not in-sample optimism.
"""
import sys, json
import numpy as np

path = sys.argv[1]
d = np.load(path)
R, X, DL = d["R"], d["X"], d["DL"]
L = int(d["L"]); zeta = float(d["zeta"]); delta = float(d["delta"])
burn = int(d["burn"]); ntraj = R.shape[0]

print(f"L={L} zeta={zeta} delta={delta} ntraj={ntraj} burn={burn}")
rbar = R[:, burn:].mean()

feat_sets = {
    "K":        [0],
    "K,K2":     None,          # handled specially
    "K,q2,qq1": [0, 1, 2],
    "all5":     [0, 1, 2, 3, 4],
}

for s_len in (16, 32, 64):
    m = int(round(s_len / delta))
    nav = DL.shape[1] - burn
    nblk = nav // m
    if nblk < 2:
        continue
    print(f"\n--- block length s = {s_len} ({nblk} blocks/traj, {ntraj*nblk} total) ---")
    # y = block Lambda minus its mean;  dX = X(end) - X(start) of the block
    y_t, dX_t = [], []
    for i in range(ntraj):
        ys, dxs = [], []
        for b in range(nblk):
            a0 = burn + b * m
            a1 = a0 + m
            ys.append(DL[i, a0:a1].sum())
            dxs.append(X[i, min(a1, X.shape[1]-1)] - X[i, a0])
        y_t.append(np.array(ys)); dX_t.append(np.array(dxs))
    y_all = np.concatenate(y_t)
    v0 = y_all.var(ddof=1)
    print(f"  Var(dLambda) = {v0:8.2f}   Var/s = {v0/s_len:7.3f}   "
          f"Var(logW) over T=L: {(1-zeta)**2 * v0 * L/s_len:7.2f}")

    for name in ("K", "K,K2", "K,q2,qq1", "all5"):
        def design(dx, xs):
            if name == "K":
                return dx[:, [0]]
            if name == "K,K2":
                return np.column_stack([dx[:, 0], xs[:, 0] ** 2 - 0.0])
            if name == "K,q2,qq1":
                return dx[:, [0, 1, 2]]
            return dx
        # cross-fitted residuals
        res = []
        for i in range(ntraj):
            Dtr = np.vstack([design(dX_t[j], dX_t[j]) for j in range(ntraj) if j != i])
            ytr = np.concatenate([y_t[j] for j in range(ntraj) if j != i])
            Dtr = np.column_stack([np.ones(len(ytr)), Dtr])
            beta, *_ = np.linalg.lstsq(Dtr, ytr, rcond=None)
            Dte = design(dX_t[i], dX_t[i])
            Dte = np.column_stack([np.ones(len(y_t[i])), Dte])
            res.append(y_t[i] - Dte @ beta)
        res = np.concatenate(res)
        vr = res.var(ddof=1)
        print(f"  control [{name:9s}]  Var_res = {vr:8.2f}   G = {v0/vr:6.3f}")

    # ---- upper bound: what would a PERFECT control give? ----
    # perfect g removes the whole predictable drift, leaving the martingale.
    # lower-bound the achievable by regressing on the FULL future instead:
    print("  (for reference) sd(dLambda) = %.2f, (1-zeta)*sd = %.3f"
          % (np.sqrt(v0), (1 - zeta) * np.sqrt(v0)))
