#!/usr/bin/env python
"""Plot the saturation curves so we can EYEBALL whether T is right, rather
than trusting the recommended_T heuristic.

Reads t_saturation_raw.json (saved by t_saturation.py) and produces, per point:
  - S(t): the entanglement-entropy trajectory, with the last-20% plateau band
    shaded; if the curve is still rising into that band, T is too short.
  - B_L(T), CMI(T): the same-seed ladder (mean +/- sem over seeds) vs horizon.

It also prints a quantitative flatness table so the verdict isn't purely
visual: the fraction of the TOTAL rise that happens in the last quarter of T
(near 0 = plateaued; large = still climbing = T too short).

Usage:
    python analysis/plot_saturation.py
    python analysis/plot_saturation.py --raw outputs/diagnostics/tsat/t_saturation_raw.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _interp(t, S, frac):
    t = np.asarray(t); S = np.asarray(S)
    return float(np.interp(frac * t[-1], t, S))


def _flatness(t, S):
    """Fraction of total rise occurring in the last quarter of the trajectory."""
    t = np.asarray(t, float); S = np.asarray(S, float)
    S0, Shalf, S75, ST = S[0], _interp(t, S, 0.5), _interp(t, S, 0.75), S[-1]
    total = ST - S0
    if abs(total) < 1e-12:
        return 0.0, S0, Shalf, S75, ST
    return (ST - S75) / total, S0, Shalf, S75, ST


def main(argv=None):
    ap = argparse.ArgumentParser(description="plot saturation curves")
    ap.add_argument("--raw", type=str,
                    default="outputs/diagnostics/tsat/t_saturation_raw.json")
    ap.add_argument("--out", type=str, default="outputs/diagnostics/tsat/saturation.png")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    raw = json.loads(Path(args.raw).read_text())

    scurves = {r["key"]: r for r in raw if r.get("kind") == "scurve" and r.get("ok")}
    ladders = defaultdict(lambda: defaultdict(lambda: {"B_L": [], "CMI": []}))
    for r in raw:
        if r.get("kind") == "ladder" and r.get("ok"):
            ladders[r["key"]][r["frac"]]["B_L"].append(r["B_L"])
            ladders[r["key"]][r["frac"]]["CMI"].append(r["CMI"])

    keys = sorted(scurves)
    n = len(keys)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3 * max(n, 1)), squeeze=False)

    print("\n" + "=" * 78)
    print("FLATNESS: fraction of total S(t) rise in the LAST QUARTER of T")
    print("  (~0 = plateaued/overkill possible;  large = still climbing => T TOO SHORT)")
    print("=" * 78)
    print(f"{'point':>22} {'T':>6} {'lastQ_rise':>11} {'S(.5T)':>8} {'S(.75T)':>8} {'S(T)':>8}")

    summary = {}
    for i, k in enumerate(keys):
        sc = scurves[k]; t = sc["t"]; S = sc["S"]; T = sc["T"]
        lastq, S0, Shalf, S75, ST = _flatness(t, S)
        verdict = "TOO SHORT" if lastq > 0.10 else ("borderline" if lastq > 0.03 else "plateaued")
        print(f"{k:>22} {T:>6.0f} {lastq:>10.1%} {Shalf:>8.3f} {S75:>8.3f} {ST:>8.3f}"
              f"   {verdict}")
        summary[k] = dict(T=T, lastquarter_rise_frac=lastq, verdict=verdict)

        ax = axes[i][0]
        ax.plot(t, S, lw=1.4)
        plateau = float(np.mean(np.asarray(S)[int(0.8 * len(S)):]))
        band = max(0.02 * abs(plateau), float(np.std(np.asarray(S)[int(0.8 * len(S)):])))
        ax.axhspan(plateau - band, plateau + band, color="orange", alpha=0.2,
                   label="last-20% band")
        ax.axvline(0.75 * t[-1], color="grey", ls=":", lw=0.8)
        ax.set_title(f"{k}  S(t)   [lastQ rise {lastq:.0%}]", fontsize=9)
        ax.set_xlabel("t"); ax.set_ylabel("S"); ax.legend(fontsize=7)

        axl = axes[i][1]
        fr = sorted(ladders[k])
        for obs, mk in (("B_L", "o-"), ("CMI", "s-")):
            xs, ys, es = [], [], []
            for f in fr:
                a = np.asarray([v for v in ladders[k][f][obs]
                                if v is not None and np.isfinite(v)], float)
                if a.size:
                    xs.append(f * T); ys.append(a.mean())
                    es.append(a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0)
            if xs:
                axl.errorbar(xs, ys, yerr=es, fmt=mk, capsize=3, label=obs, ms=4)
        axl.set_title(f"{k}  observable vs horizon T", fontsize=9)
        axl.set_xlabel("T (= frac x T_prod)"); axl.set_ylabel("value"); axl.legend(fontsize=7)

    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print("\n" + "-" * 78)
    n_short = sum(1 for v in summary.values() if v["verdict"] == "TOO SHORT")
    if n_short:
        print(f"WARNING: {n_short} point(s) still climbing at T -> T may be TOO SHORT,")
        print("         not overkill. The recommended_T heuristic cannot see this.")
    else:
        print("All curves flatten before T: the T choice is supported (not too short).")
    print(f"plot -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
