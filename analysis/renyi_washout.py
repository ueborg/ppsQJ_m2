#!/usr/bin/env python
"""B4 / item 3: does the entanglement-transition signature wash out as the
Renyi index n -> 1 (von Neumann)?  This is the von Neumann-vs-crossover test.

For each (lambda, zeta) it fits the log-L coefficient of the half-cut Renyi
entropy S_n(L/2) ~ a_n * ln(L) + const, for n = 1 (von Neumann), 2, 3, over a
chosen clean-L set. a_n is the effective-central-charge slope: ~0 in the area
law, finite in the log phase. The transition is where a_n(lambda) rises.

Read the result two ways:
  - amplitude: max a_n over lambda in the log phase, per n.
  - shape: a_n(lambda) curves and the ratio a_1/a_2 near the transition.
A CFT (no washout) predicts a_n ∝ (1 + 1/n), i.e. a_1/a_2 = (2)/(1.5) ≈ 1.33,
with von Neumann the LARGEST. Washout shows up as a_1 SUPPRESSED/SMEARED
relative to that near lambda_c -- the n->1 signal weakening while n=2,3 stay sharp.

NOTE: large-L points are finite-N_c biased (see nc_bias_pairs); default L set
stays <=64 to limit that. This is exploratory scoping on existing data; the
decisive version reruns unchanged on the clean higher-N_c campaign.

Usage:
    python analysis/renyi_washout.py \
        --data /scratch/$USER/pps_qj/pps_clone_dense /scratch/$USER/pps_qj/pps_clone_rescue \
        --Ls 8,16,24,32,64 --out outputs/diagnostics/renyi_washout
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SF = ("L", "lam", "zeta", "N_c", "S_mean", "S_renyi_2_mean", "S_renyi_3_mean")


def _from_npz(path):
    try:
        with np.load(path, allow_pickle=False) as d:
            r = {}
            for k in SF:
                if k in d.files:
                    v = d[k]
                    r[k] = float(v) if v.ndim == 0 else float(np.asarray(v).ravel()[0])
            return r if "L" in r and "S_renyi_2_mean" in r else None
    except Exception:
        return None


def load(paths):
    recs = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.glob("clone_*.npz")):
                r = _from_npz(f)
                if r:
                    recs.append(r)
        elif p.suffix == ".pkl":
            for key, rec in pickle.load(open(p, "rb")).items():
                if isinstance(rec, dict) and "S_renyi_2_mean" in rec:
                    r = {k: float(rec[k]) for k in SF if k in rec}
                    if "L" not in r and isinstance(key, (tuple, list)):
                        r["L"], r["lam"], r["zeta"] = map(float, key[:3])
                    if "L" in r:
                        recs.append(r)
    return recs


# S_n field per Renyi index
NFIELD = {1: "S_mean", 2: "S_renyi_2_mean", 3: "S_renyi_3_mean"}


def slope_lnL(Ls, Sn):
    """Fit S_n = a*ln(L)+b; return a (log-coefficient) or nan if <3 valid."""
    Ls = np.asarray(Ls, float); Sn = np.asarray(Sn, float)
    m = np.isfinite(Ls) & np.isfinite(Sn) & (Ls > 0)
    if m.sum() < 3:
        return float("nan")
    p = np.polyfit(np.log(Ls[m]), Sn[m], 1)
    return float(p[0])


def build(recs, Ls):
    """-> {zeta: {lam: {n: a_n}}} of log-L slopes per Renyi index."""
    # index: (zeta, lam) -> {L: rec}
    idx = defaultdict(dict)
    for r in recs:
        if int(round(r["L"])) in Ls:
            idx[(round(r["zeta"], 3), round(r["lam"], 4))][int(round(r["L"]))] = r
    out = defaultdict(dict)
    for (z, lam), byL in idx.items():
        a = {}
        for n in (1, 2, 3):
            f = NFIELD[n]
            Lv = [L for L in byL if f in byL[L] and np.isfinite(byL[L][f])]
            a[n] = slope_lnL(Lv, [byL[L][f] for L in Lv]) if len(Lv) >= 3 else float("nan")
        if any(np.isfinite(v) for v in a.values()):
            out[z][lam] = a
    return out


def analyse(slopes, ztarget=None):
    """Per zeta: amplitudes max a_n and a_1/a_2 ratio in the log phase."""
    rows = {}
    for z in sorted(slopes):
        lams = sorted(slopes[z])
        a1 = np.array([slopes[z][l].get(1, np.nan) for l in lams])
        a2 = np.array([slopes[z][l].get(2, np.nan) for l in lams])
        a3 = np.array([slopes[z][l].get(3, np.nan) for l in lams])
        amp1 = np.nanmax(a1) if np.isfinite(a1).any() else np.nan
        amp2 = np.nanmax(a2) if np.isfinite(a2).any() else np.nan
        amp3 = np.nanmax(a3) if np.isfinite(a3).any() else np.nan
        # ratio at the lambda where a2 peaks (deep log phase)
        if np.isfinite(a2).any():
            ip = np.nanargmax(a2)
            ratio = a1[ip] / a2[ip] if np.isfinite(a1[ip]) and a2[ip] else np.nan
        else:
            ratio = np.nan
        rows[z] = dict(lams=lams, a1=a1, a2=a2, a3=a3,
                       amp1=float(amp1), amp2=float(amp2), amp3=float(amp3),
                       ratio_a1_a2=float(ratio))
    return rows


CFT_RATIO = (1 + 1 / 1) / (1 + 1 / 2)  # 1.333: a_1/a_2 for a CFT (no washout)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Renyi-index washout (B4 item 3)")
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--Ls", type=str, default="8,16,24,32,64")
    ap.add_argument("--out", type=str, default="outputs/diagnostics/renyi_washout")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    Ls = [int(x) for x in args.Ls.split(",")]

    recs = load(args.data)
    print(f"records with Renyi: {len(recs)}  (Ls used: {Ls})")
    slopes = build(recs, Ls)
    rows = analyse(slopes)

    print("\n" + "=" * 74)
    print("Log-phase amplitude of a_n = slope of S_n(L/2) vs ln L, per zeta")
    print(f"  CFT (no washout) predicts a_1/a_2 = {CFT_RATIO:.2f} (von Neumann LARGEST).")
    print("  a_1/a_2 well BELOW that near the transition = von Neumann signal washing out.")
    print("=" * 74)
    print(f"{'zeta':>6} {'max a_1':>8} {'max a_2':>8} {'max a_3':>8} {'a1/a2':>7}  flag")
    summary = {}
    for z in sorted(rows):
        r = rows[z]
        flag = ("--" if not np.isfinite(r["ratio_a1_a2"])
                else "WASHED?" if r["ratio_a1_a2"] < 0.9 * CFT_RATIO
                else "stable")
        print(f"{z:>6.3f} {r['amp1']:>8.3f} {r['amp2']:>8.3f} {r['amp3']:>8.3f} "
              f"{r['ratio_a1_a2']:>7.2f}  {flag}")
        summary[f"z{z}"] = {k: r[k] for k in ("amp1", "amp2", "amp3", "ratio_a1_a2")}

    # plot a_n(lambda) per zeta (grid)
    zs = [z for z in sorted(rows) if np.isfinite(rows[z]["a2"]).any()]
    if zs:
        ncol = 3; nrow = int(np.ceil(len(zs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                                 squeeze=False)
        for i, z in enumerate(zs):
            ax = axes[i // ncol][i % ncol]; r = rows[z]
            for a, lab, c in ((r["a1"], "n=1 (vN)", "tab:red"),
                              (r["a2"], "n=2", "tab:blue"),
                              (r["a3"], "n=3", "tab:green")):
                ax.plot(r["lams"], a, "o-", ms=3, color=c, label=lab)
            ax.set_title(f"zeta={z}", fontsize=9); ax.grid(alpha=0.3)
            ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$a_n$ (log-coeff)")
            if i == 0:
                ax.legend(fontsize=7)
        for j in range(len(zs), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out / "renyi_washout.png", dpi=120)
        import json
        (out / "renyi_washout.json").write_text(json.dumps(summary, indent=1, default=float))
        print(f"\nplot -> {out/'renyi_washout.png'}")
        print(f"data -> {out/'renyi_washout.json'}")

    print("\nReading guide: where n=2,3 show a clear a_n rise (log phase) but the")
    print("n=1 curve stays low/smeared, that is the von Neumann transition being")
    print("replaced by a crossover (the Poboiko-Mirlin scenario). Confirm against")
    print("theory; large-L bias and the single L/2 cut are caveats -- decisive run")
    print("is the clean higher-N_c campaign with more cuts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
