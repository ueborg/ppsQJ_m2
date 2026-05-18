"""
Decisive test for the QJ-PPS FSS framework with predicted y_zeta=1.

Usage:
    python analysis/test_yzeta1_collapse.py [--add-fst PATH]

Prediction (from extensivity of click vertex over bonds + locality at
the no-click fixed point):

    lambda_c(L, zeta) * sqrt(L) = F(zeta * L)

where F(x) is a single universal function: F(x->0) -> C0 (no-click
crossover) and F(x->infty) -> A*sqrt(x) (TD power law).

When L=192,256 FST data arrives, run this with --add-fst pointing to
the aggregated pickle. If the new points fall on the L<=128 curve in
the (zeta*L, lambda_c*sqrt(L)) plane, the prediction is confirmed
and the thesis-level result is:

    lambda_c(zeta) ~ A*sqrt(zeta) in the thermodynamic limit
"""
import argparse, pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--main-agg',
                    default='/Users/catlover1337/Downloads/clone_aggregate(1).pkl',
                    help='Path to main L<=128 aggregate')
parser.add_argument('--add-fst', default=None,
                    help='Optional path to L=192,256 FST aggregate')
parser.add_argument('--outdir', default='/Users/catlover1337/Documents/ppsQJ_m2/analysis')
args = parser.parse_args()

# Load main aggregate
agg = pickle.load(open(args.main_agg, 'rb'))
if args.add_fst is not None and Path(args.add_fst).exists():
    fst = pickle.load(open(args.add_fst, 'rb'))
    agg = {**agg, **fst}
    print(f"Loaded {len(agg)} entries (including FST data)")
else:
    print(f"Loaded {len(agg)} entries (L<=128 only)")

Ls = sorted(set(k[0] for k in agg))
zetas = sorted(set(k[2] for k in agg))
lams = sorted(set(k[1] for k in agg))

def BL(L, lam, zeta):
    k = (L, lam, zeta)
    if k not in agg: return None
    v = agg[k].get('B_L_mean')
    if v is None or not np.isfinite(v): return None
    return float(v)

def lambda_c_BL(L1, L2, zeta):
    diffs = []
    for lam in lams:
        v1 = BL(L1, lam, zeta); v2 = BL(L2, lam, zeta)
        if v1 is not None and v2 is not None:
            diffs.append((lam, v1-v2))
    if len(diffs) < 2: return None
    diffs.sort()
    arr_l = np.array([p[0] for p in diffs]); arr_d = np.array([p[1] for p in diffs])
    for i in range(len(arr_d)-1):
        if arr_d[i]*arr_d[i+1] < 0:
            return float(arr_l[i] - arr_d[i]*(arr_l[i+1]-arr_l[i])/(arr_d[i+1]-arr_d[i]))
    return None

L_pairs = [(Ls[i], Ls[i+1]) for i in range(len(Ls)-1)]
clean_pairs = [(L1,L2) for L1,L2 in L_pairs if L1>=16]

# Collect data with predicted scaling
data = []
for L1,L2 in clean_pairs:
    Lg = np.sqrt(L1*L2)
    for z in zetas:
        lc = lambda_c_BL(L1, L2, z)
        if lc is not None and lc > 0.01:
            if (L1, L2) == (96, 128) and z <= 0.10:
                continue  # known noisy N_c=100 regime
            data.append({
                'L1': L1, 'L2': L2, 'Lg': Lg, 'zeta': z, 'lc': lc,
                'x': z * Lg,           # PREDICTED scaling variable: zeta*L
                'y': lc * np.sqrt(Lg), # lc*sqrt(L)
                'is_fst': L1 >= 192,
            })

# Plot the decisive test
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: raw (zeta, lambda_c) showing critical line
ax = axes[0]
n_pairs = len(clean_pairs)
pair_colors = plt.cm.plasma(np.linspace(0, 0.85, n_pairs))
for (L1,L2), c in zip(clean_pairs, pair_colors):
    pts = sorted([(d['zeta'], d['lc']) for d in data if (d['L1'],d['L2'])==(L1,L2)])
    if pts:
        zs, lcs = zip(*pts)
        marker = 's' if L1>=192 else 'o'
        size = 9 if L1>=192 else 6
        ax.plot(zs, lcs, '-', color=c, markersize=size, marker=marker,
                label=f'({L1},{L2})' + (' [FST]' if L1>=192 else ''))

# Reference: lambda_c ~ sqrt(zeta) with A=0.5
zs_th = np.linspace(0.01, 1.0, 200)
ax.plot(zs_th, 0.5*np.sqrt(zs_th), 'k--', lw=1.5, alpha=0.7,
        label=r'$\lambda_c = 0.5\sqrt{\zeta}$ (predicted TD)')
ax.axhline(0.5, color='red', ls=':', alpha=0.5, label='Born rule ≈ 0.5')

ax.set_xlabel(r'$\zeta$')
ax.set_ylabel(r'$\lambda_c(L_g,\zeta)$')
ax.set_title('Critical line: data vs $\\lambda_c \\sim \\sqrt{\\zeta}$ prediction')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
ax.set_xscale('log')

# Right: THE DECISIVE COLLAPSE TEST
ax = axes[1]
for (L1,L2), c in zip(clean_pairs, pair_colors):
    pts = sorted([(d['x'], d['y']) for d in data if (d['L1'],d['L2'])==(L1,L2)])
    if pts:
        xs, ys = zip(*pts)
        marker = 's' if L1>=192 else 'o'
        size = 9 if L1>=192 else 6
        ax.plot(xs, ys, '-', color=c, markersize=size, marker=marker,
                label=f'({L1},{L2})' + (' [FST]' if L1>=192 else ''))

# Reference function F(x)
xs_ref = np.logspace(-1.5, 2.5, 200)
# Small-x: F = C0 ~ 0.7
# Large-x: F ~ A*sqrt(x) with A=0.5
F_pred = np.where(xs_ref < 1.0, 0.7, 0.5*np.sqrt(xs_ref))
ax.plot(xs_ref, F_pred, 'k--', lw=1.5, alpha=0.6,
        label=r'$F(x)$: const at small $x$, $\sim\sqrt{x}$ at large $x$')
ax.axhline(0.7, color='gray', ls=':', alpha=0.4)
ax.plot(xs_ref, 0.5*np.sqrt(xs_ref), 'gray', ls=':', alpha=0.4)

ax.set_xlabel(r'$\zeta L_g$  (predicted FSS variable)')
ax.set_ylabel(r'$\lambda_c \sqrt{L_g}$')
ax.set_title('THE DECISIVE TEST: $\\lambda_c\\sqrt{L} = F(\\zeta L)$\n'
             '(if FST points fall on L<=128 curve, framework confirmed)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
outpath = Path(args.outdir) / 'yzeta1_collapse_test.png'
plt.savefig(outpath, dpi=120, bbox_inches='tight')
print(f"Saved {outpath}")

# Print collapse-quality metric
print("\nCollapse quality check (within-bin variance):")
xs = np.array([d['x'] for d in data])
ys = np.array([d['y'] for d in data])
log_xs = np.log(xs); log_ys = np.log(np.abs(ys))
order = np.argsort(log_xs)
log_xs = log_xs[order]; log_ys = log_ys[order]
dx = 0.3
residuals = []
for x_center in np.arange(log_xs.min()+dx, log_xs.max()-dx, dx/2):
    mask = np.abs(log_xs - x_center) < dx
    if np.sum(mask) >= 2:
        residuals.append(np.std(log_ys[mask]))
print(f"  Mean log-y std within log-x bins of width {dx}: {np.mean(residuals):.4f}")
print(f"  (lower = better collapse; comparable to ~0.20 at L<=128)")

# Final report
fst_data = [d for d in data if d['is_fst']]
if fst_data:
    print("\n=== FST data summary ===")
    print(f"  Number of FST points: {len(fst_data)}")
    for d in fst_data:
        print(f"  L=({d['L1']},{d['L2']}) ζ={d['zeta']:.3f}: λ_c={d['lc']:.3f}, "
              f"x={d['x']:.2f}, y={d['y']:.3f}")
    print("\nVisual check: do the FST [square] markers lie on the L<=128 [circle] curve?")
    print("  YES -> y_zeta=1 framework confirmed, lambda_c~sqrt(zeta) is the result.")
    print("  NO  -> need to reassess.")
else:
    print("\n(No FST data loaded -- pass --add-fst <path> when L=192,256 results arrive)")
