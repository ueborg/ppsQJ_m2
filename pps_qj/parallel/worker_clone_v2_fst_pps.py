"""Cloning worker for the finite-size test (FST) grid.

Entry point::

    python -m pps_qj.parallel.worker_clone_v2_fst_pps <task_id> <output_dir>

Thin shim over ``worker_clone_pps`` that redirects the task-parameter lookup
to the FST grid (``task_params_clone_fst``) before delegating all execution
to the original worker logic.  No other behaviour changes.

FST grid at a glance
--------------------
  L:    [192, 256]                                         (2 sizes)
  lam:  [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]         (7 points)
  zeta: [0.05, 0.10, 0.14, 0.18, 0.50, 1.00]               (6 points)
  N_c:  L=192 -> 80,  L=256 -> 40
  T_cap: L=192 -> 80, L=256 -> 50
  Total: 84 tasks  (42 per L)

  L-task ranges:
    L=192: tasks  0..41
    L=256: tasks 42..83

Purpose: test the Scenario A vs Scenario B hypothesis for the empirical
separatrix at zeta ~ 0.143 by extending the cloning algorithm to larger
L.  See the docstring in grid_pps.py for the physical question.
"""

from __future__ import annotations

import sys

# Monkey-patch the task-lookup function that worker_clone_pps.main() uses,
# then call the real main.  Same pattern as worker_clone_v2_pps.
import pps_qj.parallel.worker_clone_pps as _base
from pps_qj.parallel.grid_pps import task_params_clone_fst

_base.task_params_clone = task_params_clone_fst  # type: ignore[attr-defined]


if __name__ == "__main__":
    sys.exit(_base.main())
