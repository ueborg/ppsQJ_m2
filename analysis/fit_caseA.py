#!/usr/bin/env python
"""
fit_caseA.py -- preliminary FSS analysis of the Case A campaign (two competing
measurements, H=0, self-dual class D).  Tests the sharp predictions:
  (1) lambda_c = 1/2 PINNED for all zeta  [the clean test; crossing LOCATION,
      forced by self-duality, robust even at small L]
  (2) Ising universality: nu = 1 (vs KMR/QSD nu~5/3), c = 1/2  [SUGGESTIVE only
      at L<=64 -- 3 small sizes give a soft nu/c]

Reads a directory of caseA_*.npz (worker_caseA output) OR an aggregate pkl.
B_L = CMI*S_AB (same observable as Case B); lam = lambda_A; zeta, L as stored.

NOTE on duality: B_L(lam) need NOT equal B_L(1-lam) -- the c<->d duality is
non-local and scrambles the spatial cut (S(L/2) is already known asymmetric).
Self-duality manifests as the crossing pinned at 0.5, NOT as B_L symmetry. The
lam<->1-lam asymmetry is printed as a diagnostic, not a pass/fail.

Usage:
    python analysis/fit_caseA.py /path/to/pps_caseA            # dir of npz
    python analysis/fit_caseA.py --selftest
"""
import argparse, glob, os, sys
import numpy as np


def load(path):
    """-> by_zL {zeta: {L: [(lam, B_L, B_L_err, S_mean), ...]}} ."""
    recs = {}
    if os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, "caseA_*.npz"))):
            z = np.load(f, allow_pickle=True)
            zeta = float(z["zeta"]); L = int(z["L"]); lam = float(z["lam"])
            bm = float(z["B_L_mean"]); be = float(z["B_L_err"])
            S = float(z["S_mean"]) if "S_mean" in z.files else np.nan
            if np.isfinite(bm):
                recs.setdefault(round(zeta, 4), {}).setdefault(L, []).append((lam, bm, be, S))
    else:
        import pickle
        d = pickle.load(open(path, "rb"))
        it = d.items() if isinstance(d, dict) else ((None, r) for r in d)
        for _k, r in it:
            zeta = float(r["zeta"]); L = int(r["L"]); lam = float(r["lam"])
            bm = r.get("B_L_mean", np.nan); be = r.get("B_L_err", np.nan)
            S = r.get("S_mean", np.nan)
            if np.isfinite(bm):
                recs.setdefault(round(zeta, 4), {}).setdefault(L, []).append(
                    (lam, float(bm), float(be), float(S)))
    for zk in recs:
        for L in recs[zk]:
            recs[zk][L] = sorted(recs[zk][L])
    return recs


def crossing(cz, La, Lb):
    if La not in cz or Lb not in cz:
        return float("nan")
    a = np.array([(p[0], p[1]) for p in cz[La]]); b = np.array([(p[0], p[1]) for p in cz[Lb]])
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    lo, hi = max(a[:, 0].min(), b[:, 0].min()), min(a[:, 0].max(), b[:, 0].max())
    if hi <= lo:
        return float("nan")
    g = np.linspace(lo, hi, 800)
    d = np.interp(g, a[:, 0], a[:, 1]) - np.interp(g, b[:, 0], b[:, 1])
    s = np.where(np.diff(np.sign(d)) != 0)[0]
    if not len(s):
        return float("nan")
    # pick the STEEPEST crossing (physical), not the first: avoids spurious
    # zero-crossings where both curves are flat + noisy (saturated tails).
    k = int(s[int(np.argmax(np.abs(d[s + 1] - d[s])))])
    x0, x1, y0, y1 = g[k], g[k + 1], d[k], d[k + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def duality_asym(cz, L):
    """mean |B_L(lam) - B_L(1-lam)| / B_L over the symmetric grid (diagnostic)."""
    if L not in cz:
        return float("nan")
    arr = {round(p[0], 4): p[1] for p in cz[L]}
    res = []
    for lam, B in arr.items():
        m = round(1.0 - lam, 4)
        if m in arr and lam < 0.5:
            denom = 0.5 * (abs(B) + abs(arr[m])) + 1e-9
            res.append(abs(B - arr[m]) / denom)
    return float(np.mean(res)) if res else float("nan")


def slope_at_half(cz, L):
    """dB_L/dlam at 0.5 from the central refinement points."""
    arr = {round(p[0], 4): p[1] for p in cz[L]}
    for (a, b, h) in ((0.495, 0.505, 0.01), (0.49, 0.51, 0.02), (0.475, 0.525, 0.05)):
        if a in arr and b in arr:
            return (arr[b] - arr[a]) / h
    return float("nan")


def nu_from_slopes(cz):
    """|dB_L/dlam|_0.5 ~ L^{1/nu}; fit ln|slope| vs lnL over available L."""
    Ls, sl = [], []
    for L in sorted(cz):
        s = slope_at_half(cz, L)
        if np.isfinite(s) and abs(s) > 1e-6:
            Ls.append(L); sl.append(abs(s))
    if len(Ls) < 2:
        return None
    Ls = np.array(Ls, float); sl = np.array(sl)
    p = np.polyfit(np.log(Ls), np.log(sl), 1)
    return 1.0 / p[0], Ls.astype(int).tolist()   # nu, sizes


def c_ent(cz):
    """S_mean(L) ~ (c/6) lnL at lam=0.5 (OBC half-chain) -> c."""
    Ls, S = [], []
    for L in sorted(cz):
        arr = {round(p[0], 4): p[3] for p in cz[L]}
        v = arr.get(0.5, arr.get(0.495, arr.get(0.505, np.nan)))
        if np.isfinite(v):
            Ls.append(L); S.append(v)
    if len(Ls) < 3:
        return None, Ls
    Ls = np.array(Ls, float); S = np.array(S)
    slope = np.polyfit(np.log(Ls), S, 1)[0]
    return 6.0 * slope, [int(x) for x in Ls]


def report(recs):
    zetas = sorted(recs)
    print(f"Case A preliminary FSS -- zeta present: {zetas}")
    print("PREDICTION: lambda_c = 0.5 for ALL zeta (pinned by self-duality);"
          " Ising nu=1, c=1/2.\n")
    print("=== (1) lambda_c crossings  [the clean test: expect ~0.50] ===")
    for z in zetas:
        cz = recs[z]; Ls = sorted(cz)
        xs = []
        for i in range(len(Ls)):
            for j in range(i + 1, len(Ls)):
                lc = crossing(cz, Ls[i], Ls[j])
                if np.isfinite(lc):
                    xs.append((Ls[i], Ls[j], lc))
        s = "  ".join(f"({a},{b})={lc:.3f}" for a, b, lc in xs) or "(<2 usable L)"
        med = np.median([lc for _, _, lc in xs]) if xs else float("nan")
        print(f"  zeta={z:.2f}  L={Ls}:  {s}   median={med:.3f}")
    print("\n=== (2) preliminary universality [SUGGESTIVE at L<=64] ===")
    for z in zetas:
        cz = recs[z]
        nu = nu_from_slopes(cz)
        c, cls = c_ent(cz)
        nu_s = f"nu~{nu[0]:.2f} (slopes L={nu[1]})" if nu else "nu: <2 L"
        c_s = f"c~{c:.2f} (S(L) L={cls})" if c is not None else f"c: <3 L ({cls})"
        print(f"  zeta={z:.2f}:  {nu_s}   {c_s}    [Ising: nu=1, c=0.5; KMR/QSD nu~1.67]")
    print("\n=== diagnostic: lam<->1-lam asymmetry of B_L (expected NONZERO) ===")
    for z in zetas:
        cz = recs[z]
        a = {L: duality_asym(cz, L) for L in sorted(cz)}
        print(f"  zeta={z:.2f}:  " + "  ".join(f"L{L}={v:.2f}" for L, v in a.items()
                                                if np.isfinite(v)))


def selftest():
    # antisymmetric-about-0.5 B_L: crossing pinned at 0.5; |slope|~L^{1/nu}, nu=1
    nu, A, B = 1.0, 0.8, 0.6
    lams = sorted(set([round(x, 4) for x in np.linspace(0.35, 0.65, 13)]
                      + [0.49, 0.495, 0.505, 0.51]))
    recs = {}
    rng = np.random.default_rng(0)
    for z in (0.10, 0.30, 0.50, 1.00):
        Ls = [16, 32, 64] if z == 1.0 else [32, 64]
        for L in Ls:
            pts = []
            for lm in lams:
                x = (lm - 0.5) * L ** (1.0 / nu)
                Bv = A + 0.1 * L ** (-1.0) - B * np.tanh(x) + rng.normal(0, 0.01)
                S = (0.5 / 6) * np.log(L) + rng.normal(0, 0.01)   # c=0.5 synthetic
                pts.append((lm, Bv, 0.02, S))
            recs.setdefault(z, {})[L] = sorted(pts)
    print("[selftest] truth: lambda_c=0.5 all zeta, nu=1, c=0.5\n")
    report(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.path:
        ap.error("path to pps_caseA dir or aggregate pkl required (or --selftest)")
    report(load(a.path))


if __name__ == "__main__":
    main()
