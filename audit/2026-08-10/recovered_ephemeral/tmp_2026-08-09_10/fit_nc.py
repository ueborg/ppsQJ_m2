import numpy as np
# (algo, N_c, mean, se) for S, CMI, B_L  -- from /tmp/ncladder.log
D = {
 "S":   [("g",64,1.9916,0.0165),("g",32,1.9803,0.0248),
         ("c",32,2.0074,0.0191),("c",24,2.0020,0.0161),("c",16,2.0020,0.0231)],
 "CMI": [("g",64,0.5377,0.0154),("g",32,0.5312,0.0168),
         ("c",32,0.5680,0.0173),("c",24,0.5535,0.0133),("c",16,0.5554,0.0192)],
 "B_L": [("g",64,1.0720,0.0364),("g",32,1.0567,0.0438),
         ("c",32,1.1414,0.0389),("c",24,1.1102,0.0316),("c",16,1.1206,0.0488)],
}
rs = np.random.default_rng(11)
def fit(rows, ys):
    X = np.array([[1.0, 1.0/N if a=="g" else 0.0, 1.0/N if a=="c" else 0.0]
                  for a,N,_,_ in rows])
    w = np.array([1.0/s**2 for *_ ,s in rows])
    W = np.diag(w)
    beta = np.linalg.solve(X.T@W@X, X.T@W@ys)
    return beta          # [O_inf, b_g, b_c]
print("Joint fit  O(N_c) = O_inf + b/N_c  with SHARED O_inf")
print(f"{'obs':5s} {'O_inf':>16s} {'b_guided':>16s} {'b_ctrl':>16s} {'P(|b_c|<|b_g|)':>15s}")
for ob, rows in D.items():
    y0 = np.array([m for *_ , m, _ in [(a,N,m,s) for a,N,m,s in rows]])
    se = np.array([s for a,N,m,s in rows])
    pt = fit(rows, y0)
    B = np.array([fit(rows, y0 + rs.normal(0, se)) for _ in range(20000)])
    q = lambda c: np.percentile(B[:,c],[16,84])
    p = float(np.mean(np.abs(B[:,2]) < np.abs(B[:,1])))
    print(f"{ob:5s} {pt[0]:8.4f}[{q(0)[0]:.4f},{q(0)[1]:.4f}] "
          f"{pt[1]:+7.3f}[{q(1)[0]:+.2f},{q(1)[1]:+.2f}] "
          f"{pt[2]:+7.3f}[{q(2)[0]:+.2f},{q(2)[1]:+.2f}] {p:14.2f}")
print("\nFlatness of the CONTROLLED arm across N_c=16,24,32 (chi2/dof vs constant):")
for ob, rows in D.items():
    c = [(N,m,s) for a,N,m,s in rows if a=="c"]
    m = np.array([x[1] for x in c]); s = np.array([x[2] for x in c])
    mu = np.sum(m/s**2)/np.sum(1/s**2)
    print(f"  {ob:5s} mean={mu:.4f}  chi2/dof={np.sum(((m-mu)/s)**2)/2:.2f}")
print("\nGuided N_c=32 -> 64 shift, in sigma:")
for ob, rows in D.items():
    g = [(N,m,s) for a,N,m,s in rows if a=="g"]
    d = g[0][1]-g[1][1]; sd = np.hypot(g[0][2],g[1][2])
    print(f"  {ob:5s} {d:+.4f} +- {sd:.4f}  ({d/sd:+.2f} sigma)")
print("\nControlled pooled vs guided N_c=64:")
for ob, rows in D.items():
    c = [(m,s) for a,N,m,s in rows if a=="c"]
    mu = np.sum([m/s**2 for m,s in c])/np.sum([1/s**2 for m,s in c])
    su = np.sqrt(1/np.sum([1/s**2 for m,s in c]))
    g = [(m,s) for a,N,m,s in rows if a=="g" and N==64][0]
    d = mu-g[0]; sd = np.hypot(su,g[1])
    print(f"  {ob:5s} {d:+.4f} +- {sd:.4f}  ({d/sd:+.2f} sigma)")
