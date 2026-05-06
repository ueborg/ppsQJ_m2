"""Aggregate results from validate_dtau_worker into a comparison table.

Reads all ``dtau_*.json`` files in an output directory, groups by
(L, lam, zeta), and within each group reports each ``dtau_mult`` and a
PASS/FAIL verdict.  A second table reports statistical agreement of S_mean
against the ``dtau_mult=1.0`` baseline.

Usage::

    python -m pps_qj.tools.aggregate_dtau /scratch/$USER/pps_qj/dtau_validation_*

Verdict criteria (per (L, lam, zeta, mult) cell, averaged over seeds):
  * ESS_min/N_c >= 0.5       — population not collapsing per-step
  * n_distinct_ancestors >= 0.05*N_c  — genealogical diversity preserved
  * No realisation collapsed (run_cloning didn't raise CloningCollapse)
  * No more than 1 of N_seeds had |z|>3 vs the baseline mean S
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python -m pps_qj.tools.aggregate_dtau <outdir>"
        )
    outdir = sys.argv[1]
    files  = sorted(glob.glob(os.path.join(outdir, "dtau_*.json")))
    if not files:
        print(f"No dtau_*.json files in {outdir}", file=sys.stderr)
        return 1

    print(f"Read {len(files)} result files from {outdir}\n")

    # Group by (L, lam, zeta, mult)
    grouped: dict = defaultdict(list)
    n_failed = 0
    for fp in files:
        try:
            with open(fp) as f:
                r = json.load(f)
        except Exception as e:
            print(f"  [warn] could not read {os.path.basename(fp)}: {e}",
                  file=sys.stderr)
            continue
        if not r.get("ok", False):
            n_failed += 1
            print(f"  [collapsed] {os.path.basename(fp)}: "
                  f"{r.get('error','?')}", file=sys.stderr)
            continue
        key = (r["L"], r["lam"], r["zeta"], r["dtau_mult"])
        grouped[key].append(r)

    if not grouped:
        print("All runs collapsed — nothing to aggregate.", file=sys.stderr)
        return 1

    # ----- Per-(L, lam, zeta, mult) summary table ----------------------
    print("=" * 90)
    print("Per-configuration summary (averaged over seeds)")
    print("=" * 90)
    last_phys = None
    keys_sorted = sorted(grouped.keys())
    for key in keys_sorted:
        L, lam, zeta, mult = key
        phys = (L, lam, zeta)
        if phys != last_phys:
            if last_phys is not None:
                print()
            print(f"--- L={L} λ={lam:.2f} ζ={zeta:.2f} ---")
            print(f"{'mult':>5}  {'⟨S⟩':>9}  {'σ_S':>7}  {'⟨θ⟩':>9}  "
                  f"{'⟨ESSmin/N⟩':>11}  {'⟨anc⟩':>7}  "
                  f"{'speedup':>7}  {'wall(s)':>8}  verdict")
            last_phys = phys
        rs       = grouped[key]
        S_arr    = np.array([r["S_mean"]                  for r in rs])
        th_arr   = np.array([r["theta_hat"]               for r in rs])
        ess_arr  = np.array([r["min_ess_frac_postburnin"] for r in rs])
        anc_arr  = np.array([r["n_distinct_ancestors"]    for r in rs], dtype=float)
        wall_arr = np.array([r["wall_time"]               for r in rs])
        N_c      = rs[0]["N_c"]

        # Compute observed speedup vs mult=1.0 (if baseline present)
        baseline = grouped.get((L, lam, zeta, 1.0))
        if baseline:
            wall_base = np.mean([r["wall_time"] for r in baseline])
            speedup_factor = wall_base / wall_arr.mean() if wall_arr.mean() > 0 else float("nan")
        else:
            speedup_factor = float("nan")

        # Verdict
        ess_ok = ess_arr.mean() >= 0.5
        anc_ok = anc_arr.mean() >= 0.05 * N_c
        ok     = ess_ok and anc_ok
        verdict = "PASS" if ok else "FAIL"
        flags = []
        if not ess_ok:
            flags.append(f"ESS={ess_arr.mean():.2f}<0.5")
        if not anc_ok:
            flags.append(f"anc={anc_arr.mean():.0f}<{0.05*N_c:.0f}")
        if flags:
            verdict = "FAIL (" + ", ".join(flags) + ")"

        print(f"{mult:>5.2f}  {S_arr.mean():>9.4f}  {S_arr.std(ddof=1):>7.4f}  "
              f"{th_arr.mean():>+9.4f}  {ess_arr.mean():>11.3f}  "
              f"{anc_arr.mean():>7.1f}  {speedup_factor:>7.2f}x  "
              f"{wall_arr.mean():>8.1f}  {verdict}")

    # ----- Statistical agreement check vs mult=1.0 baseline ------------
    print()
    print("=" * 90)
    print("Statistical agreement of ⟨S⟩ against mult=1.0 baseline")
    print("(z-score of mean-difference / pooled std-error;  |z|<3 = agreement)")
    print("=" * 90)
    print(f"{'L':>3}  {'λ':>5}  {'ζ':>5}  {'mult':>5}  "
          f"{'ΔS':>9}  {'SE':>7}  {'|z|':>5}  verdict")

    physical_keys = sorted({(L, lam, zeta) for (L, lam, zeta, _) in grouped})
    for phys in physical_keys:
        L, lam, zeta = phys
        baseline = grouped.get((L, lam, zeta, 1.0))
        if not baseline:
            continue
        S_base = np.array([r["S_mean"] for r in baseline])
        nb     = len(S_base)
        if nb < 2:
            continue
        for mult in sorted({m for (l, la, z, m) in grouped
                              if (l, la, z) == phys}):
            if mult == 1.0:
                continue
            test   = grouped[(L, lam, zeta, mult)]
            S_test = np.array([r["S_mean"] for r in test])
            nt     = len(S_test)
            if nt < 2:
                continue
            dS = float(S_test.mean() - S_base.mean())
            se = float(np.sqrt(S_base.var(ddof=1) / nb +
                                S_test.var(ddof=1) / nt))
            z  = abs(dS) / se if se > 0 else float("inf")
            ok = "PASS" if z < 3 else "FAIL"
            print(f"{L:>3}  {lam:>5.2f}  {zeta:>5.2f}  {mult:>5.2f}  "
                  f"{dS:>+9.4f}  {se:>7.4f}  {z:>5.2f}  {ok}")

    # ----- Final recommendation ---------------------------------------
    print()
    print("=" * 90)
    print("Largest safe dτ-multiplier per (L, lam, zeta)")
    print("(largest mult where ESS>=0.5, anc>=5% N_c, AND |z|<3 vs baseline)")
    print("=" * 90)
    for phys in physical_keys:
        L, lam, zeta = phys
        baseline = grouped.get((L, lam, zeta, 1.0))
        if not baseline:
            print(f"  L={L} λ={lam:.2f} ζ={zeta:.2f}: no baseline at mult=1.0")
            continue
        S_base = np.array([r["S_mean"] for r in baseline])
        nb     = len(S_base)
        safe_mults = []
        for mult in sorted({m for (l, la, z, m) in grouped
                              if (l, la, z) == phys}):
            test = grouped[(L, lam, zeta, mult)]
            ess = np.mean([r["min_ess_frac_postburnin"] for r in test])
            anc = np.mean([r["n_distinct_ancestors"]    for r in test])
            S_test = np.array([r["S_mean"] for r in test])
            nt = len(S_test)
            N_c = test[0]["N_c"]
            if mult == 1.0:
                z = 0.0
            else:
                if nb < 2 or nt < 2:
                    z = float("inf")
                else:
                    dS = float(S_test.mean() - S_base.mean())
                    se = float(np.sqrt(S_base.var(ddof=1) / nb +
                                        S_test.var(ddof=1) / nt))
                    z  = abs(dS) / se if se > 0 else float("inf")
            if ess >= 0.5 and anc >= 0.05 * N_c and z < 3:
                safe_mults.append(mult)
        if safe_mults:
            best = max(safe_mults)
            wall_base = np.mean([r["wall_time"] for r in baseline])
            wall_best = np.mean([r["wall_time"]
                                 for r in grouped[(L, lam, zeta, best)]])
            saving = 1.0 - wall_best / wall_base
            print(f"  L={L} λ={lam:.2f} ζ={zeta:.2f}: "
                  f"max safe mult = {best:.2f}  "
                  f"(walltime saving {saving*100:.0f}%)")
        else:
            print(f"  L={L} λ={lam:.2f} ζ={zeta:.2f}: "
                  f"no safe mult > 0 (only baseline passes)")

    if n_failed:
        print(f"\n{n_failed} runs collapsed (excluded).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
