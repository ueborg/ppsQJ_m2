"""N_c scaling diagnostic for cloning variance near criticality.

Usage (on Habrok interactive node, ~30–60 min):
    cd ~/pps_qj && source ~/venvs/pps_qj/bin/activate

    # Quick check at L=64 (cheap):
    python analysis/nc_scaling_diagnostic.py --L 64 --zeta 0.30 --lam 0.31

    # Decision-grade run at L=96 (slower, ~2h):
    python analysis/nc_scaling_diagnostic.py --L 96 --zeta 0.30 --lam 0.26

The key question this resolves
-------------------------------
Is the B_L relative error near criticality proportional to 1/sqrt(N_c)?

  YES  ->  the variance inflation is a finite-N_c sampling problem.
           MPI to N_c ~ 4000–5000 at L=128 would deliver clean crossings.
           Engineering effort on MPI is justified.

  NO   ->  the ESS collapse is genealogical degeneracy that worsens with
           system size regardless of N_c.  MPI won't rescue it.  The
           bottleneck is the resampling design, not the clone count.
           Thesis statement: cloning at L>=128 in the small-zeta window
           is a methodological limit.

Secondary outputs
-----------------
  - CMI rel-err vs N_c (compare against B_L -- CMI should be ~30% tighter)
  - min_ess_frac and n_distinct_ancestors (genealogical health diagnostics)
  - A simple fit of rel-err = A / N_c^p to extract the actual scaling exponent
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def run_point(L: int, lam: float, zeta: float, N_c: int,
              N_real: int, seed_base: int, backend: str) -> dict:
    """Run N_real independent clonings at (L, lam, zeta, N_c)."""
    from pps_qj.cloning import run_cloning
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    from pps_qj.parallel.worker_clone_pps import _batched_compute_B_L

    alpha = lam
    w = 1.0 - lam
    T = 100.0  # use same T as production

    model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)

    BL_vals, CMI_vals = [], []
    ess_mins, anc_counts = [], []

    for r in range(N_real):
        rng = np.random.default_rng(seed_base + r * 999_983)
        t0 = time.perf_counter()
        result = run_cloning(
            model, zeta=zeta, T_total=T, N_c=N_c, rng=rng,
            show_progress=False, backend=backend,
        )
        wall = time.perf_counter() - t0

        comps = _batched_compute_B_L(result.final_covs, L)
        bl = comps["B_L"];  fm = np.isfinite(bl)
        cmi = comps["CMI"]; fc = np.isfinite(cmi)
        if fm.any(): BL_vals.append(float(np.mean(bl[fm])))
        if fc.any(): CMI_vals.append(float(np.mean(cmi[fc])))
        ess_mins.append(float(result.min_ess_frac_postburnin))
        anc_counts.append(int(result.n_distinct_ancestors))
        print(f"    r={r}: BL={BL_vals[-1]:.4f}  CMI={CMI_vals[-1]:.4f}  "
              f"ess_min={ess_mins[-1]:.2f}  anc={anc_counts[-1]}  t={wall:.1f}s",
              flush=True)

    def _stats(vals):
        a = np.array(vals)
        m, s = float(np.mean(a)), float(np.std(a, ddof=1))
        return m, s, s / np.sqrt(len(a))

    bl_m, bl_s, bl_e = _stats(BL_vals) if BL_vals else (float("nan"),) * 3
    cmi_m, cmi_s, cmi_e = _stats(CMI_vals) if CMI_vals else (float("nan"),) * 3

    return dict(
        L=L, lam=lam, zeta=zeta, N_c=N_c, N_real=N_real, backend=backend,
        BL_mean=bl_m, BL_std=bl_s, BL_err=bl_e,
        BL_relErr=abs(bl_e / bl_m) if bl_m else float("nan"),
        CMI_mean=cmi_m, CMI_std=cmi_s, CMI_err=cmi_e,
        CMI_relErr=abs(cmi_e / cmi_m) if cmi_m else float("nan"),
        ess_min_mean=float(np.mean(ess_mins)),
        anc_mean=float(np.mean(anc_counts)),
    )


def main():
    ap = argparse.ArgumentParser(description="N_c scaling diagnostic for cloning variance")
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--zeta", type=float, default=0.30)
    ap.add_argument("--lam", type=float, default=0.31,
                    help="lambda near the known crossing at this zeta")
    ap.add_argument("--Nc_list", type=int, nargs="+",
                    default=[50, 100, 200, 400, 800],
                    help="N_c values to scan (order: cheap to expensive)")
    ap.add_argument("--N_real", type=int, default=5,
                    help="independent realisations per N_c point")
    ap.add_argument("--backend", type=str, default="batched",
                    choices=["scalar", "batched"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="outputs/nc_diagnostic")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"N_c scaling diagnostic: L={args.L}, zeta={args.zeta}, "
          f"lam={args.lam}, N_real={args.N_real}")
    print(f"N_c scan: {args.Nc_list}")
    print("=" * 72)
    print(f"{'N_c':>6} {'BL_mean':>10} {'BL_relErr':>10} {'CMI_relErr':>10} "
          f"{'ess_min':>8} {'anc':>6}")

    rows = []
    for N_c in args.Nc_list:
        print(f"\nRunning N_c={N_c}...")
        row = run_point(args.L, args.lam, args.zeta, N_c, args.N_real,
                        args.seed, args.backend)
        rows.append(row)
        print(f"  -> N_c={N_c:5d}: BL_relErr={row['BL_relErr']:.1%}  "
              f"CMI_relErr={row['CMI_relErr']:.1%}  "
              f"ess={row['ess_min_mean']:.2f}  anc={row['anc_mean']:.0f}")

    # Summary table
    print("\n" + "=" * 72)
    print("Summary: does rel-err scale as 1/sqrt(N_c)?")
    print(f"{'N_c':>6} {'BL_relErr':>10} {'expected_1/sqrtN':>17} {'CMI_relErr':>12}")
    ref = [r for r in rows if not np.isnan(r["BL_relErr"])]
    if ref:
        r0 = ref[0]
        C = r0["BL_relErr"] * np.sqrt(r0["N_c"])
        for row in rows:
            expected = C / np.sqrt(row["N_c"])
            print(f"{row['N_c']:>6} {row['BL_relErr']:>10.1%} {expected:>17.1%} "
                  f"{row['CMI_relErr']:>12.1%}")

    # Fit exponent
    valid = [(r["N_c"], r["BL_relErr"]) for r in rows
             if not np.isnan(r["BL_relErr"]) and r["BL_relErr"] > 0]
    if len(valid) >= 3:
        Ncs, errs = zip(*valid)
        log_N = np.log(Ncs); log_e = np.log(errs)
        p = np.polyfit(log_N, log_e, 1)
        exponent = p[0]
        print(f"\nFitted scaling exponent: rel_err ~ N_c^{exponent:.3f}")
        print(f"  Ideal (pure sampling): -0.500")
        print(f"  Measured: {exponent:.3f}")
        if abs(exponent + 0.5) < 0.15:
            verdict = "CONSISTENT with 1/sqrt(N_c) -> MPI to large N_c is justified"
        elif exponent > -0.3:
            verdict = "FLATTER than 1/sqrt(N_c) -> genealogical degeneracy dominates; MPI alone won't fix it"
        else:
            verdict = f"Intermediate ({exponent:.2f}); borderline -- check ESS and ancestor counts"
        print(f"\nVERDICT: {verdict}")

    # Save results
    out = dict(params=vars(args), rows=rows)
    jpath = outdir / f"nc_diag_L{args.L}_z{args.zeta}_lam{args.lam}.json"
    jpath.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {jpath}")


if __name__ == "__main__":
    main()
