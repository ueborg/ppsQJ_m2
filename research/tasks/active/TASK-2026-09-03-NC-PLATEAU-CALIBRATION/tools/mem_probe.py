#!/usr/bin/env python3
"""Measure the REAL peak RSS of one production population, locally.

Why this exists
---------------
Every predecessor package sized `--mem` from

    peak_MB = 128 + 2 * N_c * ((2L)^2 * 8 + (2L) * L * 16) / 1e6

and TASK-2026-09-01-SMCRUCHE-READY described its output as "the measured
732 MB peak". It is not a measurement: 732 MB is exactly what that formula
returns for L=96, N_c=512. No MaxRSS from any Ruche job appears anywhere in
this repository, so the coefficient 2 has never been checked against a running
process.

This task needs N_c = 4096 and 8192, four and eight times the largest N_c any
package has ever requested memory for, so the coefficient stops being a detail.

Method. Run one real population through the bundled certified sampler and read
`ru_maxrss`. `T` may be shortened: peak RSS is dominated by the per-clone
covariance and orbital stores, which do not depend on the number of windows.
The window-indexed genealogy arrays DO, and are added analytically by
cost_model.mem_mb rather than being extrapolated from a short run.

    tools/mem_probe.py L N_c [T]

Read-only. Runs T0 analysis compute only. Submits nothing.
"""
import os, sys, json, resource

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.abspath(os.path.join(HERE, os.pardir))
REPO = os.path.abspath(os.path.join(TASK, *([os.pardir] * 4)))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(TASK, "support"))
import instrumented as I                                    # noqa: E402


def main():
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    T = float(sys.argv[3]) if len(sys.argv) > 3 else float(L)
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    r = I.run_instrumented(L=L, zeta=0.35, lam=0.3032, N_c=N, T=T, seed=987654321,
                           dtau_mult=6.0, record_anc=True,
                           resample_scheme="systematic")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    per_clone = ((2 * L) ** 2 * 8 + (2 * L) * L * 16) / 1e6          # MB
    geneal = 2.0 * r.n_steps * N * 8 / 1e6                           # MB
    store = N * per_clone
    print(json.dumps(dict(
        L=L, N_c=N, T=T, n_steps=r.n_steps,
        base_mb=round(base, 1), peak_mb=round(peak, 1),
        increment_mb=round(peak - base, 1),
        clone_store_mb=round(store, 1), genealogy_mb=round(geneal, 1),
        K_implied=round((peak - base - geneal) / store, 3),
        old_formula_mb=round(128.0 + 2.0 * store, 1))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
