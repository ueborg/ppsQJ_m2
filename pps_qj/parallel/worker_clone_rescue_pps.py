"""Cloning worker for the lean large-L rescue run.

    python -m pps_qj.parallel.worker_clone_rescue_pps <task_id> <output_dir>

Thin shim over worker_clone_pps; redirects task lookup to
task_params_clone_rescue.  Saves the full field set (B_L + CMI components
+ Renyi) like the dense worker.

Rescue grid: L in {128, 160}, 10 zeta (decisive, all in dense grid),
13 narrow lambda points centered on measured dense crossings, reduced N_c
(250 at L=128, 120 at L=160).  130 tasks per L.
  L=128: tasks   0..129
  L=160: tasks 130..259

Required env at submit: PPS_RECORD_RENYI=1, PPS_FORCE_RERUN=1,
PPS_N_WORKERS=5.
"""
from __future__ import annotations

import sys

import pps_qj.parallel.worker_clone_pps as _base
from pps_qj.parallel.grid_pps import task_params_clone_rescue

_base.task_params_clone = task_params_clone_rescue  # type: ignore[attr-defined]

if __name__ == "__main__":
    sys.exit(_base.main())
