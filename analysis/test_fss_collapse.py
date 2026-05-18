#!/usr/bin/env python
"""
test_fss_collapse.py

Test the click-recycling scaling-collapse prediction:
    lambda_c(L, zeta) * sqrt(L) = F(zeta * sqrt(L))

If this collapse holds, the curves at different L should overlay onto a
single function F(x), with F(x << 1) ~ const and F(x >> 1) ~ x.

This distinguishes:
  Scenario A: lambda_c -> 0.5 for all zeta > 0  (LMR-Ising)
  Scenario B: separatrix at zeta_c ~ 0.143 (finite RG fixed point)
  Scenario C: lambda_c -> 0 as zeta -> 0 linearly (click-recycling, NEW)

Scenario C is the only one predicting the collapse.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Edit if your aggregate lives elsewhere
AGG_PATH = Path.home() / "Documents/ppsQJ_m2/data/clone_aggregate_1_.pkl"
if not AGG_PATH.exists():
    for c in [Path.home() / "Downloads/clone_aggregate_1_.pkl",
              Path.home() / "Documents/ppsQJ_m2/clone_aggregate_1_.pkl"]:
        if c.exists():
            AGG_PATH = c
            break
    else:
        raise FileNotFoundError("Set AGG_PATH to your aggregate pickle")

with open(AGG_PATH, "rb") as f:
    agg = pickle.load(f)

Ls = sorted(set(k[0] for k in agg))
zetas = sorted(set(k[2] for k in agg))
lams = sorted(set(k[1] for k in agg))
print(f"Ls={Ls}, zetas={zetas}, n_lambda={len(lams)}")


def c_eff(L, lam, zeta):
    key = (L, lam, zeta)
    if key not in agg:
        return None
    return 6.0 * agg[key]["S_mean"] / np.log(L / 2)


def lambda_c_for(L, zeta, threshold=1.0):
    """Linear interp of lambda where c_eff crosses threshold."""
    pts = [(lam, c_eff(L, lam, zeta)) for lam in lams]
    pts = [(l, c) for l, c in pts if c is not None]
    if len(pts) < 2:
        return None
    arr_l = np.array([l for l, _ in pts])
    arr_c = np.array([c for _, c in pts])
    diff = arr_c - threshold
    sc = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sc) == 0:
        return None
    i = sc[0]
    return float(arr_l[i] - diff[i] * (arr_l[i+1] - arr_l[i]) / (diff[i+1] - diff[i]))


# Print table of zeta*sqrt(L), lambda_c*sqrt(L) for all (L, zeta)
print()
print(f"{'L':>4} | {'zeta':>6} | {'lam_c':>8} | {'z*sqrt(L)':>10} | {'lc*sqrt(L)':>11}")
print("-" * 55)
rows = []
for L in Ls:
    for z in zetas:
        lc = lambda_c_for(L, z)
        if lc is not None:
            x = z * np.sqrt(L)
            y = lc * np.sqrt(L)
            rows.append((L, z, lc, x, y))
            print(f"{L:>4} | {z:>6.3f} | {lc:>8.4f} | {x:>10.3f} | {y:>11.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(Ls)))

ax = axes[0]
for L, c in zip(Ls, colors):
    pts = [(z, lc) for (LL, z, lc, _, _) in rows if LL == L]
    if pts:
        zs, lcs = zip(*pts)
        ax.plot(zs, lcs, "o-", color=c, label=f"L={L}")
ax.set_xlabel(r"$\zeta$")
ax.set_ylabel(r"$\lambda_c(L,\zeta)$")
ax.set_title("Raw critical line")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
for L, c in zip(Ls, colors):
    pts = [(x, y) for (LL, _, _, x, y) in rows if LL == L]
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=c, label=f"L={L}")
ax.set_xlabel(r"$\zeta \sqrt{L}$")
ax.set_ylabel(r"$\lambda_c \sqrt{L}$")
ax.set_title("Scaling collapse (Scenario C test)")
ax.set_xscale("log")
ax.set_yscale("log")
xs_ref = np.logspace(-1, 1.5, 50)
ax.plot(xs_ref, np.ones_like(xs_ref), "k--", alpha=0.3, label=r"$F\sim$const (no-click)")
ax.plot(xs_ref, xs_ref, "k:", alpha=0.3, label=r"$F\sim x$ (linear $\lambda_c$)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
outp = Path.home() / "Documents/ppsQJ_m2/analysis/fss_collapse_test.png"
plt.savefig(outp, dpi=120)
print(f"\nSaved plot to {outp}")
print("\nVerdict:")
print("  If the right panel shows curves OVERLAYING onto a single F(x):")
print("    -> Scenario C (click-recycling) supported")
print("    -> lambda_c -> 0 as zeta -> 0 linearly, no finite-zeta_c separatrix")
print("  If curves do NOT collapse:")
print("    -> Some other mechanism applies; L=192,256 needed")
