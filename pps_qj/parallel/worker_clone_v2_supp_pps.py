"""Cloning worker for the v2 supplement grid (low-λ + large-ζ tasks).

Entry point::

    python -m pps_qj.parallel.worker_clone_v2_supp_pps <task_id> <output_dir>

Thin shim over worker_clone_pps that routes task IDs to the supplement grid
(task_params_clone_v2_supp).  All execution logic is unchanged.

Supplement grid layout (300 tasks)
------------------------------------
  Block A (tasks   0.. 59): low-λ small-ζ
    L=[32,48,64,96,128], λ=[0.005,0.01,0.03,0.075], ζ=[0.02,0.05,0.10]
  Block B (tasks  60..299): large-ζ
    L=[32,48,64,96,128], 24-point λ grid, ζ=[0.90,0.95]
"""
from __future__ import annotations
import sys

import pps_qj.parallel.worker_clone_pps as _base
from pps_qj.parallel.grid_pps import task_params_clone_v2_supp

_base.task_params_clone = task_params_clone_v2_supp  # type: ignore[attr-defined]

if __name__ == "__main__":
    sys.exit(_base.main())
