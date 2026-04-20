"""Shared plotting helpers for PPS analysis scripts."""
from __future__ import annotations

from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PLOT_RCPARAMS = {
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_style() -> None:
    plt.style.use("default")
    mpl.rcParams.update(PLOT_RCPARAMS)


def color_for_L(L: int, L_range: Iterable[int]):
    L_list = list(L_range)
    vmin, vmax = min(L_list), max(L_list)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    return plt.cm.viridis(norm(L))


def color_for_zeta(zeta: float):
    return plt.cm.plasma(float(zeta))


def save_fig(fig, path_base: str) -> None:
    """Save both .pdf and .png at 150 dpi."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=150, bbox_inches="tight")


def indicator(rec: dict) -> tuple[float, float]:
    """Return (B_L_mean, B_L_err) — the c-basis indicator used throughout.

    B'_L (dual-basis) is NaN for every L in L_DOOB because L_d = L - 1 is
    never divisible by 4 for L in {16, 24, 32, 48, 64, 96, 128, 192, 256}
    (all satisfy L-1 ≡ 3 mod 4). We therefore use B_L exclusively.
    """
    return (
        float(rec.get("B_L_mean", float("nan"))),
        float(rec.get("B_L_err", float("nan"))),
    )
