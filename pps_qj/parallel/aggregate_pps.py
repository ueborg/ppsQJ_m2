"""Aggregation helpers for the PPS scan outputs.

Loads all ``doob_*.npz`` / ``clone_*.npz`` files in a directory and returns
nested dicts keyed by ``(L, round(lam, 4), round(zeta, 3))``. Also provides
a completion check and pickle save for downstream local analysis.

CLI::

    python -m pps_qj.parallel.aggregate_pps doob <output_dir>
    python -m pps_qj.parallel.aggregate_pps clone <output_dir>
    python -m pps_qj.parallel.aggregate_pps check_missing doob <output_dir>
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from pps_qj.parallel.grid_pps import (
    make_clone_grid,
    make_doob_grid,
    n_tasks_clone,
    n_tasks_doob,
)


def _npz_to_dict(path: Path) -> dict:
    out: dict = {}
    with np.load(path, allow_pickle=False) as d:
        for k in d.files:
            arr = d[k]
            if arr.ndim == 0:
                # Scalars — unwrap.
                try:
                    out[k] = arr.item()
                except Exception:
                    out[k] = arr
            else:
                out[k] = np.asarray(arr)
    return out


def _key(L, lam, zeta) -> tuple:
    return (int(L), round(float(lam), 4), round(float(zeta), 3))


def aggregate_doob(output_dir) -> dict:
    output_dir = Path(output_dir)
    data: dict = {}
    n_files = 0
    n_traj_total = 0
    for path in sorted(output_dir.glob("doob_*.npz")):
        rec = _npz_to_dict(path)
        key = _key(rec["L"], rec["lam"], rec["zeta"])
        data[key] = rec
        n_files += 1
        n_traj_total += int(rec.get("n_traj", 0))
    expected = n_tasks_doob()
    missing = expected - n_files
    print(
        f"[aggregate_doob] {n_files}/{expected} tasks loaded "
        f"(missing {missing}), total trajectories = {n_traj_total}"
    )
    return data


def aggregate_clone(output_dir) -> dict:
    output_dir = Path(output_dir)
    data: dict = {}
    n_files = 0
    for path in sorted(output_dir.glob("clone_*.npz")):
        rec = _npz_to_dict(path)
        key = _key(rec["L"], rec["lam"], rec["zeta"])
        data[key] = rec
        n_files += 1
    expected = n_tasks_clone()
    missing = expected - n_files
    print(
        f"[aggregate_clone] {n_files}/{expected} tasks loaded (missing {missing})"
    )
    return data


def check_doob_completion(output_dir) -> tuple[list[int], list[int]]:
    output_dir = Path(output_dir)
    all_ids = set(range(n_tasks_doob()))
    done_ids: set[int] = set()
    for path in output_dir.glob("doob_*.npz"):
        try:
            tid = int(path.stem.split("_")[-1])
            done_ids.add(tid)
        except ValueError:
            pass
    missing = sorted(all_ids - done_ids)
    completed = sorted(done_ids)
    return completed, missing


def check_clone_completion(output_dir) -> tuple[list[int], list[int]]:
    output_dir = Path(output_dir)
    all_ids = set(range(n_tasks_clone()))
    done_ids: set[int] = set()
    for path in output_dir.glob("clone_*.npz"):
        try:
            tid = int(path.stem.split("_")[-1])
            done_ids.add(tid)
        except ValueError:
            pass
    missing = sorted(all_ids - done_ids)
    completed = sorted(done_ids)
    return completed, missing


def _compress_ids_to_slurm_ranges(ids: Iterable[int]) -> str:
    ids = sorted(set(int(i) for i in ids))
    if not ids:
        return ""
    ranges: list[str] = []
    start = prev = ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = i
    ranges.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ",".join(ranges)


def save_pkl(data: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: python -m pps_qj.parallel.aggregate_pps "
            "<doob|clone|check_missing> <output_dir> [clone|doob]"
        )
        return 1
    cmd = argv[0]
    if cmd == "doob":
        output_dir = Path(argv[1])
        data = aggregate_doob(output_dir)
        completed, missing = check_doob_completion(output_dir)
        print(f"completed: {len(completed)}, missing: {len(missing)}")
        if not missing:
            save_pkl(data, output_dir / "doob_aggregate.pkl")
            print(f"saved: {output_dir / 'doob_aggregate.pkl'}")
        return 0
    if cmd == "clone":
        output_dir = Path(argv[1])
        data = aggregate_clone(output_dir)
        completed, missing = check_clone_completion(output_dir)
        print(f"completed: {len(completed)}, missing: {len(missing)}")
        if not missing:
            save_pkl(data, output_dir / "clone_aggregate.pkl")
            print(f"saved: {output_dir / 'clone_aggregate.pkl'}")
        return 0
    if cmd == "check_missing":
        which = argv[1] if len(argv) > 1 else "doob"
        output_dir = Path(argv[2]) if len(argv) > 2 else Path(".")
        if which == "doob":
            _, missing = check_doob_completion(output_dir)
        else:
            _, missing = check_clone_completion(output_dir)
        # Emit a SLURM-compatible range spec on stdout.
        print(_compress_ids_to_slurm_ranges(missing))
        return 0
    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
