"""Cloning worker for the v2 production-scan grid.

Entry point::

    python -m pps_qj.parallel.worker_clone_v2_pps <task_id> <output_dir>

This is a thin shim over ``worker_clone_pps`` that redirects the task-parameter
lookup to the v2 grid (``task_params_clone_v2``) before delegating all
execution to the original worker logic.  No other behaviour changes.

v2 grid at a glance
-------------------
  L:    8, 16, 24, 32, 48, 64, 96, 128  (8 sizes)
  λ:    24 points  (0.02..0.90, dense near 0.30–0.55 and at small λ)
  ζ:    10 points  (0.02..1.00, dense for ζ ≤ 0.20)
  N_c:  2000/1000/800/500/300/200/150/100 per L
  Total: 1920 tasks  (240 per L)

  Node layout:
    tasks   0..719  → submit_clone_v2_small.sh  (L=8,16,24)
    tasks 720..1199 → submit_clone_v2_medium.sh (L=32,48)
    tasks 1200..1919→ submit_clone_v2_large.sh  (L=64,96,128)
"""
from __future__ import annotations

import sys

# Monkey-patch the task-lookup function that worker_clone_pps.main() uses,
# then call the real main.  This avoids duplicating the worker logic.
import pps_qj.parallel.worker_clone_pps as _base
from pps_qj.parallel.grid_pps import task_params_clone_v2

# Replace the lookup.  worker_clone_pps.main() calls task_params_clone(),
# which is imported at module level.  We shadow it in the module's own
# namespace so the running code picks up the v2 version.
_base.task_params_clone = task_params_clone_v2  # type: ignore[attr-defined]

if __name__ == "__main__":
    sys.exit(_base.main())
