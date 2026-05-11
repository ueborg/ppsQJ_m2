"""Test the analytical prediction c_eff(λ; ζ) ≈ ζ · c_eff(λ; ζ=1).

This is the leading-order linear-in-ζ scaling derived in
theory/qj_bosonization_calculation.md (Section 9.5).

Usage:
    python scripts/check_linear_zeta_prediction.py <aggregate.pkl>

Plots and prints:
  - For each λ, the ratio R(λ, ζ) = c_eff(λ; ζ) / (ζ · c_eff(λ; ζ=1)).
  - If R ≈ 1 across all (λ, ζ), the linear-in-ζ prediction is good.
  - Deviations identify where higher-order ζ corrections matter.
"""
from __future__ import annotations
import sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def main(agg_path):
    with open(agg_path, 'rb') as f:
        agg = pickle.load(f)

    # Fit c_eff(λ, ζ) = slope of S vs ln L over L ∈ {32, 48, 64, 96, 128}
    Ls = sorted({k[0] for k in agg})
    lams = sorted({k[1] for k in agg})
    zetas = sorted({k[2] for k in agg})
    Ls_fit = [L for L in Ls if L >= 32]

    def c_eff(lam, zeta):
        S = np.array([agg[(L, lam, zeta)]['S_mean'] for L in Ls_fit])
        sig = np.array([agg[(L, lam, zeta)]['S_err'] for L in Ls_fit])
        ln_L = np.log(Ls_fit)
        w = 1.0 / sig ** 2
        A = np.column_stack([np.ones_like(ln_L), ln_L])
        W = np.diag(w)
        coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ S)
        return coef[1]

    # For each λ, compute c_eff(ζ=1) and the ratio R(ζ) for all other ζ
    print(f"{'λ':>6} {'c(ζ=1)':>10}  " +
          ''.join(f"{f'R(ζ={z})':>11}" for z in zetas if z < 1.0))
    print('-' * 90)
    R_data = defaultdict(list)
    for lam in lams:
        c_1 = c_eff(lam, 1.0)
        if c_1 < 0.01:
            continue  # area-law, no meaningful crossover
        row = []
        for zeta in zetas:
            if zeta >= 1.0:
                continue
            c_z = c_eff(lam, zeta)
            R = c_z / (zeta * c_1)
            row.append(R)
            R_data[zeta].append((lam, c_z, R))
        print(f"{lam:>6.3f} {c_1:>10.3f}  " + ''.join(f"{r:>11.3f}" for r in row))

    # Plot R(λ, ζ) heatmap
    R_arr = np.full((len(zetas), len(lams)), np.nan)
    for i, zeta in enumerate(zetas):
        if zeta >= 1.0:
            continue
        for j, lam in enumerate(lams):
            c_z = c_eff(lam, zeta)
            c_1 = c_eff(lam, 1.0)
            if c_1 > 0.1:
                R_arr[i, j] = c_z / (zeta * c_1)
    fig, ax = plt.subplots(figsize=(11, 5))
    R_clip = np.clip(R_arr, 0, 3)
    im = ax.imshow(R_clip, origin='lower', aspect='auto',
                   extent=[lams[0], lams[-1], zetas[0], zetas[-1]],
                   cmap='RdBu_r', vmin=0, vmax=3)
    ax.axhline(1.0, color='k', alpha=0.3, ls='--')
    ax.set_xlabel('λ')
    ax.set_ylabel('ζ')
    ax.set_title(r'Test of $c_{\rm eff}(\lambda;\zeta) \approx \zeta \cdot c_{\rm eff}(\lambda; \zeta=1)$' + '\n'
                 r'Ratio $R(\lambda,\zeta) = c_{\rm eff}(\lambda;\zeta) / [\zeta\cdot c_{\rm eff}(\lambda; 1)]$;'
                 r' R=1 means perfect linear scaling')
    cb = fig.colorbar(im, ax=ax, label='R')
    fig.tight_layout()
    fig.savefig('linear_zeta_test.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('\nSaved linear_zeta_test.png')

    # Plot R(ζ) for several λ
    fig, ax = plt.subplots(figsize=(9, 5))
    lams_check = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lams_check)))
    for lam, col in zip(lams_check, colors):
        zarr = np.array([z for z in zetas if z < 1.0])
        Rs = []
        for zeta in zarr:
            c_z = c_eff(lam, zeta)
            c_1 = c_eff(lam, 1.0)
            if c_1 > 0.05:
                Rs.append(c_z / (zeta * c_1))
            else:
                Rs.append(np.nan)
        ax.plot(zarr, Rs, 'o-', color=col, ms=7, lw=1.5, label=f'λ={lam}')
    ax.axhline(1.0, color='k', alpha=0.5, ls='--', label='linear prediction')
    ax.set_xlabel('ζ')
    ax.set_ylabel(r'$R(\lambda, \zeta) = c_{\rm eff}(\zeta) / [\zeta\cdot c_{\rm eff}(\zeta=1)]$')
    ax.set_title(r'Linear-in-$\zeta$ prediction test at several $\lambda$')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 3.5)
    fig.tight_layout()
    fig.savefig('linear_zeta_curves.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('Saved linear_zeta_curves.png')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python check_linear_zeta_prediction.py <aggregate.pkl>')
        sys.exit(1)
    main(sys.argv[1])
