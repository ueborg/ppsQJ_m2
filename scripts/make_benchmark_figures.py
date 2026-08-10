#!/usr/bin/env python
"""Generate the exact-Q_ζ benchmark figures (Figs 1–4 in the spec) from an
aggregated CSV produced by ``scripts/aggregate.py``.

Figures:
  1. ⟨S_{L/2}⟩ vs λ for each (L, ζ): subplot per L, line per ζ.
  2. ⟨S_{L/2}⟩ vs ζ at fixed λ ∈ {0.3, 0.5, 0.7}: subplot per λ, line per L.
  3. Heatmap of ⟨S⟩ over (λ, ζ) at the largest L present.
  4. ⟨N_T⟩ vs ζ for each (L, λ): subplot per L, one line per λ.

Also prints a table of argmax_λ |∂S/∂λ| for each (L, ζ) — the apparent
crossover — and flags whether it shifts systematically with ζ.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["L"] = int(r["L"])
            r["lambda"] = float(r["lambda"])
            r["zeta"] = float(r["zeta"])
            r["S_mean"] = float(r["S_mean"])
            r["S_sem"] = float(r["S_sem"])
            r["n_clicks_mean"] = float(r["n_clicks_mean"])
            rows.append(r)
    return rows


def _group(rows: list[dict], key) -> dict:
    out: dict = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def figure_S_vs_lambda(rows: list[dict], outdir: Path) -> None:
    by_L = _group(rows, lambda r: r["L"])
    Ls = sorted(by_L)
    fig, axes = plt.subplots(1, len(Ls), figsize=(4.5 * len(Ls), 4.0), sharey=True, squeeze=False)
    for ax, L in zip(axes[0], Ls):
        by_z = _group(by_L[L], lambda r: r["zeta"])
        for zeta in sorted(by_z):
            sub = sorted(by_z[zeta], key=lambda r: r["lambda"])
            xs = [r["lambda"] for r in sub]
            ys = [r["S_mean"] for r in sub]
            err = [r["S_sem"] for r in sub]
            ax.errorbar(xs, ys, yerr=err, marker="o", ms=3, lw=1,
                        label=f"ζ={zeta:.1f}")
        ax.set_title(f"L={L}")
        ax.set_xlabel("λ")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(r"$\langle S_{L/2}\rangle_{Q_\zeta}$")
    axes[0, -1].legend(fontsize=8, loc="best", ncol=2)
    fig.suptitle("Figure 1: half-cut entanglement vs λ, coloured by ζ")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_S_vs_lambda.png", dpi=140)
    plt.close(fig)


def figure_S_vs_zeta(rows: list[dict], outdir: Path, lambdas=(0.3, 0.5, 0.7)) -> None:
    fig, axes = plt.subplots(1, len(lambdas), figsize=(4.5 * len(lambdas), 4.0), sharey=True, squeeze=False)
    for ax, target_lam in zip(axes[0], lambdas):
        # Select rows whose λ is closest to target within a small tolerance.
        relevant = [r for r in rows if abs(r["lambda"] - target_lam) < 0.02]
        by_L = _group(relevant, lambda r: r["L"])
        for L in sorted(by_L):
            sub = sorted(by_L[L], key=lambda r: r["zeta"])
            xs = [r["zeta"] for r in sub]
            ys = [r["S_mean"] for r in sub]
            err = [r["S_sem"] for r in sub]
            ax.errorbar(xs, ys, yerr=err, marker="o", ms=3, lw=1, label=f"L={L}")
        ax.set_title(f"λ = {target_lam:.1f}")
        ax.set_xlabel("ζ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0, 0].set_ylabel(r"$\langle S_{L/2}\rangle_{Q_\zeta}$")
    fig.suptitle("Figure 2: half-cut entanglement vs ζ at fixed λ")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_S_vs_zeta.png", dpi=140)
    plt.close(fig)


def figure_heatmap(rows: list[dict], outdir: Path) -> None:
    by_L = _group(rows, lambda r: r["L"])
    L_max = max(by_L)
    subset = by_L[L_max]
    lambdas = sorted({r["lambda"] for r in subset})
    zetas = sorted({r["zeta"] for r in subset})
    grid = np.full((len(zetas), len(lambdas)), np.nan)
    lam_idx = {lam: i for i, lam in enumerate(lambdas)}
    zeta_idx = {z: j for j, z in enumerate(zetas)}
    for r in subset:
        grid[zeta_idx[r["zeta"]], lam_idx[r["lambda"]]] = r["S_mean"]
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    im = ax.imshow(
        grid, aspect="auto", origin="lower",
        extent=[min(lambdas) - 0.05, max(lambdas) + 0.05,
                min(zetas) - 0.05, max(zetas) + 0.05],
        cmap="viridis",
    )
    ax.set_xlabel("λ")
    ax.set_ylabel("ζ")
    ax.set_title(f"Figure 3: ⟨S⟩(λ, ζ) at L = {L_max}")
    fig.colorbar(im, ax=ax, label=r"$\langle S_{L/2}\rangle_{Q_\zeta}$")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_heatmap.png", dpi=140)
    plt.close(fig)


def figure_clicks(rows: list[dict], outdir: Path) -> None:
    by_L = _group(rows, lambda r: r["L"])
    Ls = sorted(by_L)
    fig, axes = plt.subplots(1, len(Ls), figsize=(4.5 * len(Ls), 4.0), sharey=False, squeeze=False)
    for ax, L in zip(axes[0], Ls):
        by_lam = _group(by_L[L], lambda r: r["lambda"])
        for lam in sorted(by_lam):
            sub = sorted(by_lam[lam], key=lambda r: r["zeta"])
            xs = [r["zeta"] for r in sub]
            ys = [r["n_clicks_mean"] for r in sub]
            ax.plot(xs, ys, marker="o", ms=3, lw=1, label=f"λ={lam:.1f}")
        ax.set_title(f"L={L}")
        ax.set_xlabel("ζ")
        ax.set_ylabel(r"$\langle N_T\rangle_{Q_\zeta}$")
        ax.grid(alpha=0.3)
    axes[0, -1].legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle("Figure 4: mean click count vs ζ — tilting suppresses clicks")
    fig.tight_layout()
    fig.savefig(outdir / "fig4_clicks.png", dpi=140)
    plt.close(fig)


def print_crossover_table(rows: list[dict]) -> None:
    """For each (L, ζ): locate argmax_λ |dS/dλ| on the sampled grid."""
    from itertools import groupby
    keyed = sorted(rows, key=lambda r: (r["L"], r["zeta"], r["lambda"]))
    header = f"{'L':>4} {'zeta':>6} {'λ*':>8} {'|dS/dλ|':>12}"
    print(header)
    print("-" * len(header))
    for (L, zeta), grp in groupby(keyed, key=lambda r: (r["L"], r["zeta"])):
        sub = list(grp)
        lams = np.array([r["lambda"] for r in sub])
        Ss = np.array([r["S_mean"] for r in sub])
        if lams.size < 3:
            continue
        # Central differences on the sorted grid.
        order = np.argsort(lams)
        lams, Ss = lams[order], Ss[order]
        d = np.abs(np.gradient(Ss, lams))
        idx = int(np.argmax(d))
        print(f"{L:>4} {zeta:>6.2f} {lams[idx]:>8.3f} {d[idx]:>12.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()

    rows = _load_rows(args.csv)
    if not rows:
        raise SystemExit("no rows in CSV")
    args.outdir.mkdir(parents=True, exist_ok=True)
    figure_S_vs_lambda(rows, args.outdir)
    figure_S_vs_zeta(rows, args.outdir)
    figure_heatmap(rows, args.outdir)
    figure_clicks(rows, args.outdir)
    print_crossover_table(rows)


if __name__ == "__main__":
    main()
