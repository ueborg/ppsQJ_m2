#!/usr/bin/env python
"""Aggregate per-point ``.npz`` benchmark files into a single summary table.

Scans a results directory (recursively), loads each ``.npz``, and writes a
CSV with one row per ``(L, λ, ζ)`` point containing the mean/SEM of
``S_{L/2}``, mean/variance of the click count, trajectory count, walltime,
and the source file path.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


COLUMNS = [
    "L", "lambda", "zeta", "method", "T",
    "S_mean", "S_sem", "n_clicks_mean", "n_clicks_var",
    "N_traj_completed", "walltime_s", "source",
]


def _iter_npz_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.npz")):
        yield path


def summarize_one(path: Path) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            S = np.asarray(data["entropy"], dtype=np.float64)
            clicks = np.asarray(data["n_clicks"], dtype=np.int64)
            metadata = json.loads(str(data["metadata"].item()))
    except Exception as exc:
        print(f"[warn] failed to load {path}: {exc}", file=sys.stderr)
        return None
    if S.size == 0:
        return None
    sem = float(S.std(ddof=1) / np.sqrt(S.size)) if S.size > 1 else float("nan")
    return {
        "L": int(metadata.get("L", -1)),
        "lambda": float(metadata.get("lambda", float("nan"))),
        "zeta": float(metadata.get("zeta", float("nan"))),
        "method": str(metadata.get("method", "")),
        "T": float(metadata.get("T", float("nan"))),
        "S_mean": float(S.mean()),
        "S_sem": sem,
        "n_clicks_mean": float(clicks.mean()),
        "n_clicks_var": float(clicks.var(ddof=1)) if clicks.size > 1 else float("nan"),
        "N_traj_completed": int(S.size),
        "walltime_s": float(metadata.get("walltime_s", float("nan"))),
        "source": str(path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows = []
    for path in _iter_npz_files(args.results_dir):
        row = summarize_one(path)
        if row is not None:
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["L"], r["lambda"], r["zeta"])):
            writer.writerow(row)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
