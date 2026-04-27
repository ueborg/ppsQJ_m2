"""Pre-compute and save the Gaussian backward pass for a cloning grid task.

Usage::

    python -m scripts.habrok.precompute_backward_pass <task_id> <output_dir>

The backward pass is saved as ``<output_dir>/backward_<task_id:05d>.npz``
using the same naming convention expected by ``worker_clone_pps.py``.

When the file exists before a cloning run, the worker automatically loads it
and enables Jack-Sollich feedback control (see ``cloning.run_cloning``).

Typical wall times (single core, measured):
    L=16 : ~2–3 min
    L=32 : ~10–15 min
    L=64 : ~60–90 min  (estimate — not yet benchmarked)

Workflow on Habrok
------------------
Run one backward pass per task as a short SLURM job before the main cloning
scan.  The submit script ``submit_backward_scan.sh`` (not yet written) should
mirror ``submit_clone_pps.sh`` but with a much shorter wall-time limit and
fewer cores (one backward pass per core).

Exit codes: 0 = success, 1 = failure.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 2:
        raise SystemExit(
            "usage: python -m scripts.habrok.precompute_backward_pass <task_id> <output_dir>"
        )
    task_id = int(argv[0])
    output_dir = Path(argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"backward_{task_id:05d}.npz"
    if output_path.exists():
        print(f"backward task {task_id}: already done, skipping", flush=True)
        return 0

    # Import here to keep startup fast for the early-exit case above.
    from pps_qj.backward_pass import run_gaussian_backward_pass
    from pps_qj.backward_pass_io import save_backward_pass
    from pps_qj.gaussian_backend import build_gaussian_chain_model
    from pps_qj.parallel.grid_pps import task_params_clone

    t0 = time.time()
    task = task_params_clone(task_id)
    L    = int(task["L"])
    lam  = float(task["lam"])
    alpha = float(task["alpha"])
    w    = float(task["w"])
    zeta = float(task["zeta"])
    T    = float(task["T"])

    print(
        f"backward task {task_id}: L={L}, λ={lam:.3f} (α={alpha:.3f}, w={w:.3f}), "
        f"ζ={zeta:.2f}, T={T:.0f}",
        flush=True,
    )

    try:
        model = build_gaussian_chain_model(L=L, w=w, alpha=alpha)

        # The backward ODE is stiff near the phase transition and at small ζ.
        # max_step caps the integrator step size; smaller values increase
        # accuracy at the cost of compute time.  The defaults (rtol=1e-5,
        # atol=1e-7) were validated against exact results in the Doob scan.
        bwd = run_gaussian_backward_pass(
            model,
            T=T,
            zeta=zeta,
            rtol=1e-5,
            atol=1e-7,
            sample_points=500,
            show_progress=True,
        )

        metadata = dict(L=L, alpha=alpha, w=w, zeta=zeta, T=T, lam=lam)
        save_backward_pass(bwd, output_path, metadata=metadata)

        elapsed = time.time() - t0
        print(
            f"backward task {task_id}: saved to {output_path.name}  "
            f"(t={elapsed:.1f}s, ODE steps={bwd.solution.t.size})",
            flush=True,
        )
        return 0

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"backward task {task_id}: FAILED ({elapsed:.1f}s) — {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
