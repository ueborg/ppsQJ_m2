"""Cloning worker for the Phase-2 high-L supplement (195 tasks).

Entry point::

    python -m pps_qj.parallel.worker_clone_phase2_pps <task_id> <output_dir>

Thin shim over ``worker_clone_pps`` that redirects task-parameter lookup
to ``task_params_clone_phase2``.  Same pattern as the dense / v2 shims.

Phase-2 grid at a glance
------------------------
  zeta:  0.10, 0.15, 0.20, 0.25, 0.30        (decisive window)
  L:     160, 192, 256                       (breaks the L<=128 ceiling)
  N_c:   400 (L=160), 300 (L=192), 250 (L=256)
  lambda: 13 points, +/-0.06 around lambda_c^{(128)}(zeta)
  Total: 195 tasks (65 per L)

  Task layout (L outer, zeta middle, lambda inner):
    L=160: tasks   0.. 64
    L=192: tasks  65..129
    L=256: tasks 130..194

  SLURM L-split scripts:
    submit_clone_phase2_L160.sh  (tasks   0.. 64,  N_c=400, ~96h)
    submit_clone_phase2_L192.sh  (tasks  65..129,  N_c=300, ~96h)
    submit_clone_phase2_L256.sh  (tasks 130..194,  N_c=250, ~168h)

CRITICAL PRE-SUBMIT STEP
------------------------
The narrow lambda mesh must be centered on the dense L<=128 crossings.
Before submitting, update ``_PHASE2_LAMBDA_C_128`` in ``grid_pps.py`` with
the measured lambda_c^{(128)}(zeta) values and set
``_PHASE2_CENTERS_VERIFIED = True``.  The grid builder warns while the
centers are placeholders.

Required env vars at submit time (same as dense campaign)
---------------------------------------------------------
  PPS_RECORD_RENYI=1    Enable Renyi-2 / Renyi-3 recording.
  PPS_FORCE_RERUN=1     Recompute even if output .npz exists.
  PPS_N_WORKERS=5       Intra-task realisation parallelism.

Saves the same field set as the dense worker (B_L + full CMI tripartition
components + Renyi-2/3).
"""
from __future__ import annotations

import sys

import pps_qj.parallel.worker_clone_pps as _base
from pps_qj.parallel.grid_pps import task_params_clone_phase2

_base.task_params_clone = task_params_clone_phase2  # type: ignore[attr-defined]

if __name__ == "__main__":
    sys.exit(_base.main())
