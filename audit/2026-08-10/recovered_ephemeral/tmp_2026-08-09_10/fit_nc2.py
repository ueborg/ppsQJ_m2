import numpy as np
D = {
 "S":   [("g",128,2.0204,0.0111),("g",64,1.9916,0.0165),("g",32,1.9803,0.0248),
         ("c",32,2.0074,0.0191),("c",24,2.0020,0.0161),("c",16,2.0020,0.0231),
         ("c",12,1.9437,0.0227)],
 "CMI": [("g",128,0.5488,0.0088),("g",64,0.5377,0.0154),("g",32,0.5312,0.0168),
         ("c",32,0.5680,0.0173),("c",24,0.5535,0.0133),("c",16,0.5554,0.0192),
         ("c",12,0.5489,0.0203)],
 "B_L": [("g",128,1.1095,0.0222),("g",64,1.0720,0.0364),("g",32,1.0567,0.0438),
         ("c",32,1.1414,0.0389),("c",24,1.1102,0.0316),("c",16,1.1206,0.0488),
         ("c",12,1.0778,0.0513)],
}
rs = np.random.default_rng(11)
def fit(rows, ys):
    X = np.array([[1.0, 1.0/N if a=="g" else 0.0, 1.0/N if a=="c" else 0.0]
                  for a,N,_,_ in rows])
    W = np.diag([1.0/s**2 for *_,s in rows])
    return np.linalg.solve(X.T@W@X, X.T@W@ys)
for drop12 in (False, True):
    print(f"\n===== joint fit, shared O_inf {'(ctrl N_c=12 EXCLUDED)' if drop12 else '(all points)'} =====")
    print(f"{'obs':5s} {'O_inf':>22s} {'b_guided':>20s} {'b_ctrl':>20s} {'P(|bc|<|bg|)':>12s}")
    for ob, rows0 in D.items():
        rows = [r for r in rows0 if not (drop12 and r[0]=="c" and r[1]==12)]
        y0 = np.array([m for _,_,m,_ in rows]); se = np.array([s for *_,s in rows])
        pt = fit(rows, y0)
        B = np.array([fit(rows, y0 + rs.normal(0, se)) for _ in range(20000)])
        q = lambda c: np.percentile(B[:,c],[16,84])
        p = float(np.mean(np.abs(B[:,2]) < np.abs(B[:,1])))
        print(f"{ob:5s} {pt[0]:8.4f}[{q(0)[0]:.4f},{q(0)[1]:.4f}] "
              f"{pt[1]:+7.3f}[{q(1)[0]:+.2f},{q(1)[1]:+.2f}] "
              f"{pt[2]:+7.3f}[{q(2)[0]:+.2f},{q(2)[1]:+.2f}] {p:11.2f}")
print("\n===== guided sequence vs controlled plateau (N_c=16,24,32) =====")
for ob, rows in D.items():
    c = [(m,s) for a,N,m,s in rows if a=="c" and N>=16]
    mu = np.sum([m/s**2 for m,s in c])/np.sum([1/s**2 for m,s in c])
    su = np.sqrt(1/np.sum([1/s**2 for m,s in c]))
    print(f"  {ob:5s} ctrl plateau = {mu:.4f}+-{su:.4f} | guided " + "  ".join(
        f"{N}:{m:.4f}({(m-mu)/np.hypot(s,su):+.2f}s)"
        for a,N,m,s in rows if a=="g"))
