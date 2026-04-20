"""Computational engine for the PPS phase-diagram analysis.

Consumes the aggregated pickles produced by
``pps_qj.parallel.aggregate_pps`` and produces:

  * ``figures/fig{1..7}.{pdf,png}``    — all final figures
  * ``results/lambda_c.npz``            — λ_c(ζ) and uncertainties
  * ``results/nu_vs_zeta.npz``          — ν(ζ), δν, resolution status
  * ``results/nu_quality_curves.npz``   — full ν-quality scan per ζ
  * ``results/crossings.npz``           — raw (L, L') λ-crossings per ζ

B_L (c-basis) is used EXCLUSIVELY; B'_L (dual) is NaN across the whole grid
because L-1 is never divisible by 4 for L ∈ L_DOOB.

Usage::

    python analysis/pps_phase_diagram.py <doob_pkl> <clone_pkl|none> <output_dir>

The notebook in ``notebooks/pps_results_analysis.ipynb`` *loads* the figures
this script produces — it does NOT recompute.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar

from analysis._analysis_common import (
    apply_style,
    color_for_L,
    color_for_zeta,
    indicator,
    save_fig,
)


# Grid constants — duplicated here to keep analysis self-contained.
L_DOOB = [16, 24, 32, 48, 64, 96, 128, 192, 256]
ZETA_VALS_DOOB = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LAMBDA_GRID_LIMITS = (0.1, 0.9)


# =======================================================================
# Data assembly
# =======================================================================

def build_B_curves(doob_data: dict) -> dict:
    """Return ``{zeta: {L: (lam_arr, B_mean_arr, B_err_arr)}}`` sorted by λ.

    Only points with finite ``B_L_mean`` are included.
    """
    out: dict = {}
    # Collate per (zeta, L).
    bucket: dict[tuple[float, int], list[tuple[float, float, float]]] = {}
    for key, rec in doob_data.items():
        L, lam, zeta = key
        mean, err = indicator(rec)
        if np.isnan(mean):
            continue
        bucket.setdefault((round(zeta, 3), int(L)), []).append((float(lam), mean, err))
    for (zeta, L), pts in bucket.items():
        pts.sort()
        lam_arr = np.array([p[0] for p in pts])
        mean_arr = np.array([p[1] for p in pts])
        err_arr = np.array([p[2] for p in pts])
        out.setdefault(zeta, {})[L] = (lam_arr, mean_arr, err_arr)
    return out


# =======================================================================
# PART A — finite-size crossings + λ_c extrapolation
# =======================================================================

def _spline_or_none(lam: np.ndarray, B: np.ndarray) -> Optional[CubicSpline]:
    # Need at least 4 unique points for a cubic spline.
    if lam.size < 4:
        return None
    try:
        return CubicSpline(lam, B)
    except Exception:
        return None


def find_lambda_crossings(curves: dict) -> list[dict]:
    """For every ζ and every adjacent L pair, find λ where B_L(λ) = B_{L'}(λ).

    Returns a list of dicts {zeta, L1, L2, lam_cross, d_spread, err}.
    """
    found: list[dict] = []
    for zeta, per_L in curves.items():
        Ls = sorted(set(per_L.keys()) & set(L_DOOB))
        if len(Ls) < 2:
            continue
        splines: dict[int, CubicSpline] = {}
        for L in Ls:
            lam_arr, B_arr, _ = per_L[L]
            sp = _spline_or_none(lam_arr, B_arr)
            if sp is not None:
                splines[L] = sp

        for i in range(len(Ls) - 1):
            L1, L2 = Ls[i], Ls[i + 1]
            if L1 not in splines or L2 not in splines:
                continue
            # Shared interval.
            lam1 = curves[zeta][L1][0]
            lam2 = curves[zeta][L2][0]
            lo = max(lam1.min(), lam2.min(), LAMBDA_GRID_LIMITS[0])
            hi = min(lam1.max(), lam2.max(), LAMBDA_GRID_LIMITS[1])
            if hi - lo < 0.02:
                continue

            f = lambda x: float(splines[L1](x) - splines[L2](x))
            # Scan for a sign change on a fine grid.
            grid = np.linspace(lo, hi, 201)
            vals = np.array([f(x) for x in grid])
            sign_changes = np.where(np.diff(np.signbit(vals)))[0]
            if len(sign_changes) == 0:
                continue
            # Use the sign change nearest the interval midpoint.
            mid = 0.5 * (lo + hi)
            k = sign_changes[np.argmin(np.abs(grid[sign_changes] - mid))]
            a, b = grid[k], grid[k + 1]
            try:
                lam_cross = brentq(f, a, b, xtol=1e-6)
            except Exception:
                continue

            # Propagate error from B_L error bars.
            e1 = float(np.interp(lam_cross, curves[zeta][L1][0], curves[zeta][L1][2]))
            e2 = float(np.interp(lam_cross, curves[zeta][L2][0], curves[zeta][L2][2]))
            # |d/dλ (B1 - B2)| via finite difference on the spline.
            h = 1e-3
            slope = (f(lam_cross + h) - f(lam_cross - h)) / (2 * h)
            if abs(slope) < 1e-8:
                err_lam = float("nan")
            else:
                err_lam = float(np.sqrt(e1**2 + e2**2) / abs(slope))

            found.append(dict(
                zeta=zeta, L1=L1, L2=L2,
                lam_cross=float(lam_cross), err=err_lam,
            ))
    return found


def extrapolate_lambda_c(crossings: list[dict]) -> dict:
    """For each ζ: linear fit λ_cross vs 1/sqrt(L1*L2), extrapolate to 0."""
    by_zeta: dict[float, list[dict]] = {}
    for c in crossings:
        by_zeta.setdefault(c["zeta"], []).append(c)

    out: dict = {}
    for zeta, entries in by_zeta.items():
        if len(entries) == 0:
            continue
        x = np.array([1.0 / np.sqrt(c["L1"] * c["L2"]) for c in entries])
        y = np.array([c["lam_cross"] for c in entries])
        w = np.array([c["err"] for c in entries])
        w_safe = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        if len(entries) >= 2:
            # Weighted least squares: y ≈ a + b x.
            wts = 1.0 / (w_safe ** 2)
            X = np.column_stack([np.ones_like(x), x])
            W = np.diag(wts)
            try:
                beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
                cov = np.linalg.inv(X.T @ W @ X)
                lam_c = float(beta[0])
                err_lam_c = float(np.sqrt(max(cov[0, 0], 0.0)))
            except np.linalg.LinAlgError:
                lam_c = float(np.mean(y))
                err_lam_c = float(np.std(y) / np.sqrt(len(y)))
        else:
            lam_c = float(y[0])
            err_lam_c = float(w_safe[0])
        out[zeta] = dict(
            lam_c=lam_c, err_lam_c=err_lam_c,
            n_crossings=len(entries),
            crossings=entries,
        )
    return out


# =======================================================================
# PART B — ν extraction by data collapse
# =======================================================================

def _collapse_quality(nu: float, lam_c: float,
                      points_by_L: dict[int, tuple[np.ndarray, np.ndarray]],
                      window: float = 0.3,
                      n_x_grid: int = 50) -> float:
    """Sum of squared pairwise differences between B_L(x) curves on a shared
    x-grid, where x = (λ - λ_c) L^{1/ν}.

    Returns +inf if fewer than 2 L values have data inside the window.
    """
    usable: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L, (lam, B) in points_by_L.items():
        mask = np.abs(lam - lam_c) <= window
        if mask.sum() >= 3:
            usable[L] = (lam[mask], B[mask])
    if len(usable) < 2:
        return float("inf")

    # Rescale λ -> x per L.
    rescaled: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for L, (lam, B) in usable.items():
        x = (lam - lam_c) * (L ** (1.0 / nu))
        order = np.argsort(x)
        rescaled[L] = (x[order], B[order])

    # Shared x-range: intersection of per-L ranges.
    x_lo = max(r[0][0] for r in rescaled.values())
    x_hi = min(r[0][-1] for r in rescaled.values())
    if x_hi - x_lo < 1e-6:
        return float("inf")

    xg = np.linspace(x_lo, x_hi, n_x_grid)
    interps: dict[int, np.ndarray] = {}
    for L, (x, B) in rescaled.items():
        interps[L] = np.interp(xg, x, B)

    Ls_sorted = sorted(interps.keys())
    q = 0.0
    n_pairs = 0
    for i in range(len(Ls_sorted)):
        for j in range(i + 1, len(Ls_sorted)):
            q += float(np.mean((interps[Ls_sorted[i]] - interps[Ls_sorted[j]]) ** 2))
            n_pairs += 1
    return q / max(n_pairs, 1)


def extract_nu(zeta: float, curves: dict, lam_c: float,
               nu_min: float = 0.5, nu_max: float = 4.0,
               n_grid: int = 200) -> dict:
    """Extract ν(ζ) via dense grid scan + minimize_scalar refinement.

    Returns a dict with keys::

        resolved : bool
        nu, nu_err : float (NaN if unresolved)
        nu_grid, quality : 1-D arrays (always saved)
        lam_c    : echoed
        reason   : why unresolved, if applicable
    """
    per_L = curves.get(zeta, {})
    points_by_L = {L: (arr[0], arr[1]) for L, arr in per_L.items()
                   if L in L_DOOB and len(arr[0]) >= 3}

    nu_grid = np.linspace(nu_min, nu_max, n_grid)
    quality = np.array([
        _collapse_quality(float(nu), lam_c, points_by_L) for nu in nu_grid
    ])
    base = dict(
        nu_grid=nu_grid, quality=quality, lam_c=float(lam_c),
    )

    if len(points_by_L) < 3:
        return {**base, "resolved": False, "nu": float("nan"),
                "nu_err": float("nan"),
                "reason": f"only {len(points_by_L)} L values available (need >= 3)"}

    finite = np.isfinite(quality)
    if finite.sum() < 5:
        return {**base, "resolved": False, "nu": float("nan"),
                "nu_err": float("nan"),
                "reason": "quality function is infinite almost everywhere"}

    # Dense-grid argmin.
    q_masked = np.where(finite, quality, np.inf)
    k = int(np.argmin(q_masked))
    nu0 = float(nu_grid[k])

    # Refine via minimize_scalar in a bracket around k.
    k_lo = max(0, k - 2)
    k_hi = min(n_grid - 1, k + 2)
    try:
        res = minimize_scalar(
            lambda x: _collapse_quality(float(x), lam_c, points_by_L),
            bracket=(nu_grid[k_lo], nu0, nu_grid[k_hi]),
            method="brent",
            options=dict(xtol=1e-4),
        )
        if res.success and nu_min <= res.x <= nu_max:
            nu_best = float(res.x)
            q_min = float(res.fun)
        else:
            nu_best = nu0
            q_min = float(quality[k])
    except Exception:
        nu_best = nu0
        q_min = float(quality[k])

    # Uncertainty: half-width at 2*q_min on the dense grid.
    threshold = 2.0 * q_min
    below = quality <= threshold
    if below.sum() < 2:
        reason = "quality minimum too narrow to resolve uncertainty"
        return {**base, "resolved": False, "nu": nu_best,
                "nu_err": float("nan"), "reason": reason}
    nu_band_lo = float(nu_grid[below][0])
    nu_band_hi = float(nu_grid[below][-1])
    # Check for a CLEAR minimum — band must not span most of the scan.
    if nu_band_hi - nu_band_lo > 0.5 * (nu_max - nu_min):
        return {**base, "resolved": False, "nu": nu_best,
                "nu_err": float("nan"),
                "reason": "no clear minimum (band spans > half the scan)"}
    nu_err = 0.5 * (nu_band_hi - nu_band_lo)

    return {**base, "resolved": True, "nu": float(nu_best),
            "nu_err": float(nu_err), "reason": "ok"}


# =======================================================================
# Figures
# =======================================================================

def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    fig_dir = root / "figures"
    res_dir = root / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, res_dir


def make_figure_1(curves: dict, lam_c_dict: dict, fig_dir: Path) -> None:
    zetas = sorted(curves.keys())
    n = len(zetas)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             sharex=True, squeeze=False)
    for idx, zeta in enumerate(zetas):
        ax = axes[idx // ncol, idx % ncol]
        per_L = curves[zeta]
        for L in sorted(per_L.keys()):
            lam, B, err = per_L[L]
            ax.errorbar(lam, B, yerr=err, marker="o", ms=3, lw=1,
                        capsize=2, color=color_for_L(L, L_DOOB),
                        label=f"L={L}")
        if zeta in lam_c_dict:
            lc = lam_c_dict[zeta]
            ax.axvline(lc["lam_c"], ls="--", color="k", lw=1, alpha=0.6)
            if np.isfinite(lc["err_lam_c"]):
                ax.axvspan(lc["lam_c"] - lc["err_lam_c"],
                           lc["lam_c"] + lc["err_lam_c"],
                           color="grey", alpha=0.15)
        ax.set_title(rf"$\zeta = {zeta:.2f}$")
        ax.set_yscale("log")
        if idx // ncol == nrow - 1:
            ax.set_xlabel(r"$\lambda = \alpha/(\alpha+w)$")
        if idx % ncol == 0:
            ax.set_ylabel(r"$B_L = S^{\rm top}_L \cdot \bar S_L$")
    # Remove unused axes.
    for k in range(n, nrow * ncol):
        fig.delaxes(axes[k // ncol, k % ncol])
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(r"$B_L$ vs $\lambda$ (one panel per $\zeta$)", y=1.00)
    fig.tight_layout()
    save_fig(fig, str(fig_dir / "fig1_BL_vs_lambda"))
    plt.close(fig)


def _qj_theoretical(zeta: float, lam_c1: float) -> float:
    return lam_c1 / (zeta * (1 - lam_c1) + lam_c1)


def make_figure_2(lam_c_dict: dict, fig_dir: Path) -> None:
    zetas = sorted(lam_c_dict.keys())
    if not zetas:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    zs = np.array(zetas)
    lc = np.array([lam_c_dict[z]["lam_c"] for z in zetas])
    err = np.array([lam_c_dict[z]["err_lam_c"] for z in zetas])
    ax.errorbar(zs, lc, yerr=err, fmt="o", color="C0", capsize=3,
                label="data")

    # Theoretical curve, anchored at ζ=1 if available.
    if 1.0 in lam_c_dict:
        lam_c1 = lam_c_dict[1.0]["lam_c"]
        zg = np.linspace(0.05, 1.05, 201)
        ax.plot(zg, [_qj_theoretical(z, lam_c1) for z in zg],
                "--", color="C3",
                label=r"QJ effective-rate conjecture: "
                      r"$\lambda_c(\zeta)=\lambda_c(1)/[\zeta(1-\lambda_c(1))+\lambda_c(1)]$")
        # Uncertainty band via perturbing lam_c1.
        e1 = lam_c_dict[1.0]["err_lam_c"]
        if np.isfinite(e1):
            upper = [_qj_theoretical(z, lam_c1 + e1) for z in zg]
            lower = [_qj_theoretical(z, lam_c1 - e1) for z in zg]
            ax.fill_between(zg, lower, upper, color="C3", alpha=0.15)

    ax.set_xlabel(r"$\zeta$ (partial post-selection)")
    ax.set_ylabel(r"$\lambda_c(\zeta)$")
    ax.set_title("Phase boundary: area-law to critical transition")
    ax.legend(loc="best")
    fig.tight_layout()
    save_fig(fig, str(fig_dir / "fig2_lambda_c"))
    plt.close(fig)


def make_figure_3_primary(nu_results: dict, curves: dict, lam_c_dict: dict,
                          fig_dir: Path) -> None:
    """THE primary result. Styled per spec."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Larger fonts per spec.
    ax.tick_params(axis="both", labelsize=13)

    zetas = sorted(nu_results.keys())
    for z in zetas:
        r = nu_results[z]
        if r["resolved"]:
            ax.errorbar([z], [r["nu"]], yerr=[r["nu_err"]],
                        fmt="s", color="C0", markersize=9, capsize=4,
                        markerfacecolor="C0")
        else:
            # Open circle at the bare nu estimate (even if spurious), clearly
            # flagged as unresolved.
            y = r["nu"] if np.isfinite(r.get("nu", np.nan)) else 1.5
            ax.plot([z], [y], "o", markersize=10,
                    markerfacecolor="white", markeredgecolor="grey",
                    markeredgewidth=1.5)

    # Reference lines.
    ax.axhline(5 / 3, ls="--", color="C2", lw=1.2)
    ax.text(0.14, 5 / 3 + 0.05, r"$\nu=5/3$ (free fermion)",
            color="C2", fontsize=11)
    ax.axhline(1.0, ls="--", color="C1", lw=1.2)
    ax.text(0.14, 1.0 + 0.05, r"$\nu=1$ (Ising)",
            color="C1", fontsize=11)

    ax.set_xlabel(r"$\zeta$", fontsize=16)
    ax.set_ylabel(r"$\nu(\zeta)$", fontsize=16)
    ax.set_title(r"Critical exponent $\nu(\zeta)$ vs partial post-selection",
                 fontsize=14)
    ax.set_ylim(0.5, 3.0)
    ax.set_xlim(0.05, 1.05)

    # Legend explaining markers.
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="s", color="C0", linestyle="",
               markersize=8, label="resolved"),
        Line2D([0], [0], marker="o", color="grey", linestyle="",
               markersize=8, markerfacecolor="white", label="unresolved"),
    ]
    ax.legend(handles=legend_items, loc="upper right")

    # Inset: data collapse at ζ = 1.0 as validation.
    if 1.0 in nu_results and nu_results[1.0]["resolved"] and 1.0 in lam_c_dict:
        inset = ax.inset_axes([0.08, 0.08, 0.35, 0.35])
        nu_val = nu_results[1.0]["nu"]
        lam_c = lam_c_dict[1.0]["lam_c"]
        per_L = curves.get(1.0, {})
        for L in sorted(per_L.keys()):
            lam, B, _ = per_L[L]
            mask = np.abs(lam - lam_c) <= 0.3
            if mask.sum() < 2:
                continue
            x = (lam[mask] - lam_c) * (L ** (1.0 / nu_val))
            order = np.argsort(x)
            inset.plot(x[order], B[mask][order], "o-", ms=3, lw=1,
                       color=color_for_L(L, L_DOOB), label=f"L={L}")
        inset.set_title(rf"Collapse at $\zeta=1$, $\nu={nu_val:.2f}$",
                        fontsize=9)
        inset.set_xlabel(r"$(\lambda-\lambda_c) L^{1/\nu}$", fontsize=8)
        inset.set_ylabel(r"$B_L$", fontsize=8)
        inset.tick_params(labelsize=7)

    fig.tight_layout()
    save_fig(fig, str(fig_dir / "fig3_nu_vs_zeta"))
    plt.close(fig)


def make_figure_4(nu_results: dict, curves: dict, lam_c_dict: dict,
                  fig_dir: Path) -> None:
    show_zetas = [z for z in (0.4, 0.7, 1.0)
                  if z in nu_results and nu_results[z]["resolved"]
                  and z in lam_c_dict]
    if not show_zetas:
        return
    fig, axes = plt.subplots(1, len(show_zetas),
                             figsize=(5 * len(show_zetas), 4),
                             squeeze=False)
    for ax, z in zip(axes[0], show_zetas):
        nu_val = nu_results[z]["nu"]
        lam_c = lam_c_dict[z]["lam_c"]
        per_L = curves[z]
        for L in sorted(per_L.keys()):
            lam, B, _ = per_L[L]
            mask = np.abs(lam - lam_c) <= 0.3
            if mask.sum() < 2:
                continue
            x = (lam[mask] - lam_c) * (L ** (1.0 / nu_val))
            order = np.argsort(x)
            ax.plot(x[order], B[mask][order], "o-", ms=4,
                    color=color_for_L(L, L_DOOB), label=f"L={L}")
        ax.set_title(rf"$\zeta={z:.1f}$, $\nu={nu_val:.2f}$")
        ax.set_xlabel(r"$(\lambda-\lambda_c)L^{1/\nu}$")
        ax.set_ylabel(r"$B_L$")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, str(fig_dir / "fig4_collapse"))
    plt.close(fig)


def make_figure_5(doob_data: dict, clone_data: Optional[dict],
                  fig_dir: Path) -> None:
    # <S_half>(ζ) at λ ∈ {0.2, 0.5, 0.8}.
    pick_lams = [0.2, 0.5, 0.8]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, lam_t in zip(axes, pick_lams):
        for L in L_DOOB:
            zs, ss, es = [], [], []
            for z in ZETA_VALS_DOOB:
                key = (L, round(lam_t, 4), round(z, 3))
                rec = doob_data.get(key)
                if rec is None:
                    continue
                zs.append(z)
                ss.append(float(rec["S_half_mean"]))
                es.append(float(rec["S_half_err"]))
            if zs:
                ax.errorbar(zs, ss, yerr=es, marker="o", ms=3,
                            lw=1, color=color_for_L(L, L_DOOB),
                            label=f"L={L}")
        if clone_data is not None:
            for L in (8, 12, 16):
                zs, ss, es = [], [], []
                for z in (0.3, 0.5, 0.7, 1.0):
                    key = (L, round(lam_t, 4), round(z, 3))
                    rec = clone_data.get(key)
                    if rec is None:
                        continue
                    zs.append(z)
                    ss.append(float(rec["S_mean"]))
                    es.append(float(rec["S_err"]))
                if zs:
                    ax.errorbar(zs, ss, yerr=es, marker="s", ms=4, ls="--",
                                color=color_for_L(L, L_DOOB) if L in L_DOOB
                                else "grey",
                                label=f"clone L={L}")
        ax.set_title(rf"$\lambda = {lam_t}$")
        ax.set_xlabel(r"$\zeta$")
    axes[0].set_ylabel(r"$\bar S_{L/2}$")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, str(fig_dir / "fig5_S_vs_zeta"))
    plt.close(fig)


def make_figure_6(curves: dict, lam_c_dict: dict, fig_dir: Path) -> None:
    if 1.0 not in lam_c_dict:
        return
    lam_c1 = lam_c_dict[1.0]["lam_c"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for z in sorted(curves.keys()):
        per_L = curves[z]
        Ls, Bs = [], []
        for L in sorted(per_L.keys()):
            lam, B, _ = per_L[L]
            idx = int(np.argmin(np.abs(lam - lam_c1)))
            if abs(lam[idx] - lam_c1) > 0.1:
                continue
            Ls.append(L)
            Bs.append(float(B[idx]))
        if len(Ls) >= 2:
            ax.plot(Ls, Bs, "o-", ms=5, color=color_for_zeta(z),
                    label=rf"$\zeta={z:.1f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(rf"$B_L$ at $\lambda \approx \lambda_c(\zeta=1) = {lam_c1:.3f}$")
    ax.set_title(r"Critical-point scaling $B_L$ vs $L$")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    save_fig(fig, str(fig_dir / "fig6_BL_vs_L"))
    plt.close(fig)


def make_figure_7(doob_data: dict, clone_data: Optional[dict],
                  fig_dir: Path) -> None:
    if clone_data is None:
        return
    fig, (ax_L, ax_th) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: |Δ<S>| vs L at λ=0.5 for whichever zetas overlap both datasets.
    common_L = sorted(set(k[0] for k in doob_data.keys())
                      & set(k[0] for k in clone_data.keys()))
    for z in (0.3, 0.5, 0.7, 1.0):
        xs, ys = [], []
        for L in common_L:
            key = (L, round(0.5, 4), round(z, 3))
            rd = doob_data.get(key)
            rc = clone_data.get(key)
            if rd is None or rc is None:
                continue
            sd = float(rd["S_half_mean"])
            sc = float(rc["S_mean"])
            if np.isnan(sd) or sd <= 0 or np.isnan(sc):
                continue
            xs.append(L)
            ys.append(abs(sd - sc) / sd)
        if xs:
            ax_L.plot(xs, ys, "o-", color=color_for_zeta(z),
                      label=rf"$\zeta={z:.1f}$")
    ax_L.set_xlabel(r"$L$")
    ax_L.set_ylabel(r"$|\langle S\rangle_{\rm Doob}-\langle S\rangle_{\rm Clone}|/\langle S\rangle_{\rm Doob}$")
    ax_L.set_title(r"Doob vs cloning: relative $\langle S\rangle$ difference at $\lambda=0.5$")
    ax_L.legend(fontsize=9)

    # Right: θ(ζ) from Doob vs Clone at L=16, λ=0.5.
    for z in sorted(ZETA_VALS_DOOB):
        key_d = (16, round(0.5, 4), round(z, 3))
        rd = doob_data.get(key_d)
        if rd is not None:
            ax_th.plot(z, float(rd["theta_doob"]), "o", color="C0",
                       label=r"$\theta_{\rm Doob}$" if z == ZETA_VALS_DOOB[0] else None)
    for z in (0.3, 0.5, 0.7, 1.0):
        key_c = (16, round(0.5, 4), round(z, 3))
        rc = clone_data.get(key_c)
        if rc is not None:
            ax_th.errorbar(z, float(rc["theta_mean"]),
                           yerr=float(rc["theta_err"]),
                           fmt="s", color="C1", capsize=3,
                           label=r"$\theta_{\rm Clone}$" if z == 0.3 else None)
    ax_th.set_xlabel(r"$\zeta$")
    ax_th.set_ylabel(r"$\theta(\zeta)$")
    ax_th.set_title(r"SCGF $\theta(\zeta)$ at $L=16$, $\lambda=0.5$")
    ax_th.legend(fontsize=9)

    fig.tight_layout()
    save_fig(fig, str(fig_dir / "fig7_doob_vs_clone"))
    plt.close(fig)


# =======================================================================
# Entry point
# =======================================================================

def _load_pkl(path: str) -> Optional[dict]:
    if path.lower() == "none" or not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: python analysis/pps_phase_diagram.py "
              "<doob_pkl> <clone_pkl|none> <output_dir>")
        return 1
    doob_pkl = argv[0]
    clone_pkl = argv[1]
    output_dir = Path(argv[2])

    apply_style()
    fig_dir, res_dir = _ensure_dirs(output_dir)

    doob_data = _load_pkl(doob_pkl)
    if doob_data is None:
        print(f"Doob pickle not found or 'none': {doob_pkl}")
        return 1
    clone_data = _load_pkl(clone_pkl)

    print(f"[analysis] doob entries: {len(doob_data)}")
    print(f"[analysis] clone entries: "
          f"{0 if clone_data is None else len(clone_data)}")

    curves = build_B_curves(doob_data)
    crossings = find_lambda_crossings(curves)
    lam_c_dict = extrapolate_lambda_c(crossings)

    print(f"[analysis] crossings found: {len(crossings)}")
    print(f"[analysis] lambda_c resolved for zetas: "
          f"{sorted(lam_c_dict.keys())}")

    # Save λ_c results.
    if lam_c_dict:
        zs = np.array(sorted(lam_c_dict.keys()))
        np.savez(
            res_dir / "lambda_c.npz",
            zetas=zs,
            lam_c=np.array([lam_c_dict[z]["lam_c"] for z in zs]),
            err_lam_c=np.array([lam_c_dict[z]["err_lam_c"] for z in zs]),
            n_crossings=np.array([lam_c_dict[z]["n_crossings"] for z in zs]),
        )
    np.savez(
        res_dir / "crossings.npz",
        crossings=np.array([
            (c["zeta"], c["L1"], c["L2"], c["lam_cross"], c["err"])
            for c in crossings
        ], dtype=np.float64) if crossings else np.zeros((0, 5)),
    )

    # ν extraction per ζ.
    nu_results: dict = {}
    q_curves: dict = {}
    for z in sorted(curves.keys()):
        if z not in lam_c_dict:
            continue
        r = extract_nu(z, curves, lam_c_dict[z]["lam_c"])
        nu_results[z] = r
        q_curves[z] = (r["nu_grid"], r["quality"])

    # Save ν quality curves.
    if q_curves:
        z_sorted = sorted(q_curves.keys())
        np.savez(
            res_dir / "nu_quality_curves.npz",
            zetas=np.array(z_sorted),
            nu_grid=q_curves[z_sorted[0]][0],
            quality=np.vstack([q_curves[z][1] for z in z_sorted]),
        )
    if nu_results:
        zs = np.array(sorted(nu_results.keys()))
        np.savez(
            res_dir / "nu_vs_zeta.npz",
            zetas=zs,
            nu=np.array([nu_results[z]["nu"] for z in zs]),
            nu_err=np.array([nu_results[z]["nu_err"] for z in zs]),
            resolved=np.array([nu_results[z]["resolved"] for z in zs]),
            reasons=np.array([nu_results[z]["reason"] for z in zs],
                             dtype=object),
        )

    # Figures.
    make_figure_1(curves, lam_c_dict, fig_dir)
    make_figure_2(lam_c_dict, fig_dir)
    make_figure_3_primary(nu_results, curves, lam_c_dict, fig_dir)
    make_figure_4(nu_results, curves, lam_c_dict, fig_dir)
    make_figure_5(doob_data, clone_data, fig_dir)
    make_figure_6(curves, lam_c_dict, fig_dir)
    make_figure_7(doob_data, clone_data, fig_dir)

    # Text table: ζ | λ_c ± δ | ν ± δν.
    with open(res_dir / "summary_table.txt", "w") as f:
        f.write("zeta\tlam_c\t±\tn_cross\tnu\t±\tresolved\treason\n")
        for z in sorted(set(lam_c_dict.keys()) | set(nu_results.keys())):
            lc = lam_c_dict.get(z, {})
            nr = nu_results.get(z, {})
            f.write(
                f"{z:.2f}\t"
                f"{lc.get('lam_c', float('nan')):.4f}\t"
                f"{lc.get('err_lam_c', float('nan')):.4f}\t"
                f"{lc.get('n_crossings', 0)}\t"
                f"{nr.get('nu', float('nan')):.3f}\t"
                f"{nr.get('nu_err', float('nan')):.3f}\t"
                f"{nr.get('resolved', False)}\t"
                f"{nr.get('reason', '-')}\n"
            )
    print(f"[analysis] wrote figures to {fig_dir}")
    print(f"[analysis] wrote results to {res_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
