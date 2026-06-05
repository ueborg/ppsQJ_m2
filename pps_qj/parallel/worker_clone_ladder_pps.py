"""Cloning worker for the N_c-ladder campaign (the 2-day decisive run).

    python -m pps_qj.parallel.worker_clone_ladder_pps <task_id> <output_dir>

Thin shim over worker_clone_pps; redirects task lookup to
task_params_clone_ladder.  Saves the full field set (B_L + CMI components
+ Renyi + corr) like the dense/rescue workers.

Ladder grid: L=128, 7 zeta (subset of dense, so (32,64,128) crossings pair
against existing clean L=32,64), 13 narrow lambda points centered on measured
crossings.  N_c rungs {250, 500, 800}; full grid at 500, central-3-lambda
calibration subsets at 250 and 800.  3 seed-blocks per point (15 seeds).

CRITICAL: run each N_c rung into a SEPARATE output dir (the aggregator keys
by (L,lam,zeta) and would merge rungs otherwise).  See submit_clone_ladder.sh.

Required env at submit: PPS_RECORD_RENYI=1, PPS_FORCE_RERUN=1, PPS_N_WORKERS=5,
PPS_BACKEND=scalar.
"""
from __future__ import annotations

import sys

import pps_qj.parallel.worker_clone_pps as _base
from pps_qj.parallel.grid_pps import task_params_clone_ladder

_base.task_params_clone = task_params_clone_ladder  # type: ignore[attr-defined]

if __name__ == "__main__":
    sys.exit(_base.main())
