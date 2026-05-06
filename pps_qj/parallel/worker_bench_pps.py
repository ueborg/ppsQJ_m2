"""Single-task benchmark worker for runtime calibration.

Entry point::

    python -m pps_qj.parallel.worker_bench_pps <task_id> <bench_dir> <prod_dir>

Runs N_REAL=5 realisations for the given v2 task_id, measures wall time, and
compares against the wall_time stored in the production .npz file.  Writes a
JSON result file to bench_dir/result_XXXXX.json.

Called in parallel by submit_benchmark.sh via::

    seq ... | xargs -P 6 -I{} python -m pps_qj.parallel.worker_bench_pps \
        {} <bench_dir> <prod_dir>
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from pps_qj.parallel.grid_pps import task_params_clone_v2
from pps_qj.parallel.worker_clone_pps import _run_one_realisation

N_REAL = 5


def main() -> int:
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: python -m pps_qj.parallel.worker_bench_pps "
            "<task_id> <bench_dir> <prod_dir>"
        )
    tid       = int(sys.argv[1])
    bench_dir = sys.argv[2]
    prod_dir  = sys.argv[3]

    task = task_params_clone_v2(tid)
    L, w, alpha = task["L"], task["w"], task["alpha"]
    zeta, T, N_c, seed = task["zeta"], task["T"], task["N_c"], task["seed"]

    # Load production wall time from existing .npz
    prod_file = os.path.join(prod_dir, f"clone_{tid:05d}.npz")
    if os.path.exists(prod_file):
        t_prod = float(np.load(prod_file)["wall_time"])
    else:
        print(f"task {tid}: WARNING production file not found: {prod_file}",
              flush=True)
        t_prod = float("nan")

    # Run N_REAL realisations — identical to production
    t0 = time.perf_counter()
    for r in range(N_REAL):
        _run_one_realisation(dict(
            L=L, w=w, alpha=alpha, zeta=zeta, T=T, N_c=N_c,
            seed=seed + r * 999_983,
        ))
    t_bench = time.perf_counter() - t0

    ratio = t_bench / t_prod if not np.isnan(t_prod) else float("nan")

    result = dict(
        tid=tid, L=L, lam=task["lam"], zeta=zeta, N_c=N_c, T=int(T),
        t_prod=t_prod, t_bench=t_bench, ratio=ratio,
    )

    out_path = os.path.join(bench_dir, f"result_{tid:05d}.json")
    with open(out_path, "w") as f:
        json.dump(result, f)

    print(
        f"task {tid:5d}: L={L:3d} lam={task['lam']:.3f} zeta={zeta:.2f}  "
        f"t_prod={t_prod:.1f}s  t_bench={t_bench:.1f}s  ratio={ratio:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
