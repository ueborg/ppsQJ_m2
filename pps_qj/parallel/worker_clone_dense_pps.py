"""Cloning worker for the dense fine-grid scan (4112 tasks).

Entry point::

    python -m pps_qj.parallel.worker_clone_dense_pps <task_id> <output_dir>

Thin shim over ``worker_clone_pps`` that redirects the task-parameter
lookup to ``task_params_clone_dense`` before delegating all execution
to the original worker logic.  Same pattern as ``worker_clone_v2_pps``.

Dense grid at a glance
----------------------
  L:     8, 16, 24, 32, 48, 64, 96, 128            (8 sizes)
  ζ:     5 small + 8 medium + 8 large = 21 values  (vs. v2's 10)
  λ:     26 (small region) or 24 (medium/large)
  N_c:   asymmetric 2×/2×/3× bump (4000..300 across L)
  Total: 4112 tasks  (514 per L)

  Task layout (L outer, region middle, ζ inner):
    L=8:    tasks    0..  513      L=48: tasks 2056..2569
    L=16:   tasks  514..1027      L=64: tasks 2570..3083
    L=24:   tasks 1028..1541      L=96: tasks 3084..3597
    L=32:   tasks 1542..2055      L=128:tasks 3598..4111

  SLURM L-split scripts:
    submit_clone_dense_small_L.sh  (L=8..32,   tasks    0..2055)
    submit_clone_dense_medium_L.sh (L=48,64,   tasks 2056..3083)
    submit_clone_dense_large_L.sh  (L=96,128,  tasks 3084..4111)

Required env vars at submit time
--------------------------------
  PPS_RECORD_RENYI=1    Enable Renyi-2 / Renyi-3 recording (default off).
                        Without this, S_renyi_* fields are NaN.
  PPS_FORCE_RERUN=1     Recompute tasks even if output .npz already exists
                        in the output directory.  Use with caution — defaults
                        to 0 (skip-if-exists).  Set this on the dense
                        campaign to guarantee fresh data.
  PPS_N_WORKERS=5       Intra-task realisation parallelism (matches N_REAL=5).

New output fields (vs. v2 worker output)
----------------------------------------
The worker now saves the full CMI tripartition components per (L,λ,ζ):
  CMI_mean,  CMI_err      — conditional mutual information S(A:C|B)
  S_AB_mean, S_AB_err     — entropy of A∪B
  S_BC_mean, S_BC_err     — entropy of B∪C
  S_B_mean,  S_B_err      — entropy of B alone
  S_ABC_mean, S_ABC_err   — entropy of A∪B∪C
B_L = CMI * S_AB is still computed (back-compat with v2 analysis).
"""
from __future__ import annotations

import sys

# Monkey-patch the task-lookup function that worker_clone_pps.main() uses,
# then call the real main.  This avoids duplicating the worker logic.
import pps_qj.parallel.worker_clone_pps as _base
from pps_qj.parallel.grid_pps import task_params_clone_dense

# Replace the lookup.  worker_clone_pps.main() calls task_params_clone(),
# which is imported at module level.  We shadow it in the module's own
# namespace so the running code picks up the dense version.
_base.task_params_clone = task_params_clone_dense  # type: ignore[attr-defined]

if __name__ == "__main__":
    sys.exit(_base.main())
