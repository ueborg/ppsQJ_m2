"""Live progress monitor + partial-data figure renderer.

Reads ``summary_*.json`` files (atomic; safe to read at any time) from the
Doob scan output directory, prints a progress table, and regenerates every
figure from ``pps_phase_diagram.py`` on whatever data is currently
available — silently dropping (L, λ, ζ) points that have not yet completed.

Also emits a Figure M — completion heatmap across (λ, ζ) coloured by the
fraction of L values completed — as a visual status overview.

Usage::

    python analysis/monitor_and_plot.py <output_dir> [--output-figures <dir>]

Safe to run whether the scan is 0% or 100% complete.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from analysis._analysis_common import apply_style, save_fig
from analysis.pps_phase_diagram import (
    L_DOOB,
    ZETA_VALS_DOOB,
    build_B_curves,
    extract_nu,
    extrapolate_lambda_c,
    find_lambda_crossings,
    make_figure_1,
    make_figure_2,
    make_figure_3_primary,
    make_figure_4,
    make_figure_5,
    make_figure_6,
    make_figure_7,
)
from pps_qj.parallel.grid_pps import (
    LAMBDA_VALS,
    make_doob_grid,
    n_tasks_doob,
)


# --------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------

def _load_summaries(output_dir: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted(output_dir.glob("summary_*.json")):
        # Exclude summary_clone_*.json (cloning has its own prefix).
        if path.name.startswith("summary_clone_"):
            continue
        try:
            with open(path) as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _progress_table(summaries: list[dict]) -> None:
    by_L: dict[int, list[dict]] = {}
    for s in summaries:
        if s.get("status") != "complete":
            continue
        by_L.setdefault(int(s["L"]), []).append(s)

    tasks_per_L = len(LAMBDA_VALS) * len(ZETA_VALS_DOOB)
    n_done = sum(len(v) for v in by_L.values())
    n_total = n_tasks_doob()

    print("\n" + "=" * 70)
    print(f"  PPS Doob scan progress   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"  {'L':>4} | {'done':>6} / {'total':>6} | {'%':>5} | {'avg wall':>10} | ETA")
    print("  " + "-" * 64)
    for L in L_DOOB:
        done = len(by_L.get(L, []))
        pct = 100.0 * done / tasks_per_L if tasks_per_L else 0.0
        walls = [float(s.get("wall_time", 0.0)) for s in by_L.get(L, [])]
        if walls and done < tasks_per_L:
            avg = float(np.mean(walls))
            remaining = tasks_per_L - done
            eta_sec = avg * remaining
            eta = f"~{eta_sec/3600:.1f} h" if eta_sec > 3600 else f"~{eta_sec/60:.0f} m"
        else:
            avg = float(np.mean(walls)) if walls else 0.0
            eta = "done" if done == tasks_per_L else "unknown"
        print(f"  {L:>4} | {done:>6} / {tasks_per_L:>6} | "
              f"{pct:>4.1f}% | {avg:>8.1f}s | {eta}")
    print("  " + "-" * 64)
    print(f"  total: {n_done}/{n_total} = {100.0*n_done/n_total:.1f}%")
    print("=" * 70 + "\n")


# --------------------------------------------------------------------------
# Build a partial data dict in the same format as aggregate_doob()
# --------------------------------------------------------------------------

def _summaries_to_data(summaries: list[dict]) -> dict:
    data: dict = {}
    for s in summaries:
        if s.get("status") != "complete":
            continue
        key = (int(s["L"]), round(float(s["lam"]), 4), round(float(s["zeta"]), 3))
        # Fill only the fields downstream code needs; pad with NaN for the rest.
        rec = {
            "L": int(s["L"]),
            "lam": float(s["lam"]),
            "zeta": float(s["zeta"]),
            "T": float(s.get("T", np.nan)),
            "n_traj": int(s.get("n_traj", 0)),
            "S_half_mean": float(s.get("S_half_mean", np.nan)),
            "S_half_err": float(s.get("S_half_err", np.nan)),
            "S_top_mean": float(s.get("S_top_mean", np.nan)),
            "B_L_mean": float(s.get("B_L_mean", np.nan)),
            "B_L_err": float(s.get("B_L_err", np.nan)),
            "B_L_prime_mean": float(s.get("B_L_prime_mean", np.nan)),
            "theta_doob": float(s.get("theta_doob", np.nan)),
            "Z_T": float(s.get("Z_T", np.nan)),
        }
        data[key] = rec
    return data


# --------------------------------------------------------------------------
# Figure M: completion heatmap
# --------------------------------------------------------------------------

def make_completion_heatmap(summaries: list[dict], fig_dir: Path) -> None:
    lams = list(LAMBDA_VALS)
    zetas = list(ZETA_VALS_DOOB)
    frac = np.zeros((len(zetas), len(lams)), dtype=np.float64)
    denom = float(len(L_DOOB))
    for zi, z in enumerate(zetas):
        for li, lam in enumerate(lams):
            n_done = 0
            for L in L_DOOB:
                for s in summaries:
                    if s.get("status") != "complete":
                        continue
                    if (int(s["L"]) == L
                            and round(float(s["lam"]), 4) == round(lam, 4)
                            and round(float(s["zeta"]), 3) == round(z, 3)):
                        n_done += 1
                        break
            frac[zi, li] = n_done / denom

    n_done = int(sum(1 for s in summaries if s.get("status") == "complete"))
    n_total = n_tasks_doob()
    pct = 100.0 * n_done / n_total if n_total else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(frac, aspect="auto", origin="lower", cmap="Blues",
                   vmin=0, vmax=1,
                   extent=(min(lams) - 0.025, max(lams) + 0.025,
                           min(zetas) - 0.05, max(zetas) + 0.05))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\zeta$")
    ax.set_title(f"Job completion status: {n_done}/{n_total} = {pct:.0f}%")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("fraction of L values complete")
    fig.tight_layout()
    save_fig(fig, str(fig_dir / "figM_completion_heatmap"))
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print("usage: python analysis/monitor_and_plot.py <output_dir> "
              "[--output-figures <fig_dir>]")
        return 1
    output_dir = Path(argv[0])
    fig_root: Optional[Path] = None
    i = 1
    while i < len(argv):
        if argv[i] == "--output-figures" and i + 1 < len(argv):
            fig_root = Path(argv[i + 1])
            i += 2
        else:
            i += 1
    if fig_root is None:
        fig_root = output_dir / "figures_monitor"

    apply_style()
    fig_dir = fig_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summaries = _load_summaries(output_dir)
    _progress_table(summaries)

    make_completion_heatmap(summaries, fig_dir)

    # Build a partial data dict and try every figure. Each figure gracefully
    # omits L values with no data.
    partial = _summaries_to_data(summaries)
    curves = build_B_curves(partial)

    crossings: list[dict] = []
    lam_c_dict: dict = {}
    nu_results: dict = {}
    try:
        crossings = find_lambda_crossings(curves)
        lam_c_dict = extrapolate_lambda_c(crossings)
        for z in sorted(curves.keys()):
            if z not in lam_c_dict:
                continue
            nu_results[z] = extract_nu(z, curves, lam_c_dict[z]["lam_c"])
    except Exception as exc:
        print(f"[monitor] crossing/ν analysis skipped: {exc}")

    # Figures. Wrap each in try/except so one failure doesn't kill the rest.
    for name, fn, args in [
        ("fig1", make_figure_1, (curves, lam_c_dict, fig_dir)),
        ("fig2", make_figure_2, (lam_c_dict, fig_dir)),
        ("fig3", make_figure_3_primary,
         (nu_results, curves, lam_c_dict, fig_dir)),
        ("fig4", make_figure_4, (nu_results, curves, lam_c_dict, fig_dir)),
        ("fig5", make_figure_5, (partial, None, fig_dir)),
        ("fig6", make_figure_6, (curves, lam_c_dict, fig_dir)),
        ("fig7", make_figure_7, (partial, None, fig_dir)),
    ]:
        try:
            fn(*args)
        except Exception as exc:
            print(f"[monitor] {name} skipped: {exc}")

    n_done = int(sum(1 for s in summaries if s.get("status") == "complete"))
    n_total = n_tasks_doob()
    pct = 100.0 * n_done / n_total if n_total else 0.0
    print(f"Figures written to: {fig_dir}")
    print(f"Progress: {n_done}/{n_total} jobs complete ({pct:.1f}%)")
    print("Run again at any time to refresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
