#!/usr/bin/env python3
"""Import a manually-collected Ruche snapshot and build ``RUCHE_DATA_INDEX.csv``.

This runs on the LAPTOP, against a bundle the researcher copied back by hand.
It never contacts the cluster: it reads a ``.tar.gz`` or an already-unpacked
snapshot directory and nothing else.

Usage
-----
    .venv/bin/python3 research/tools/import_ruche_snapshot.py \
        research/imports/ruche/ruche_snapshot_2026-08-20.tar.gz

    # or point it at the whole import directory to index every snapshot found
    .venv/bin/python3 research/tools/import_ruche_snapshot.py \
        --all research/imports/ruche/

Output
------
``RUCHE_DATA_INDEX.csv`` next to the snapshot (or at ``--out``), one row per
result file, with the snapshot it came from and a ``provenance_complete``
column that is the whole point of the exercise: it says whether a given file
can be tied back to the code that produced it.

What this does NOT do
---------------------
It does not promote anything into ``research/state/**``. A snapshot is an index
of what exists on a cluster; it is provenance at best and is never support for
a scientific claim. See ``research/imports/ruche/README.md``.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import tarfile
from pathlib import Path
from typing import Iterator, Optional

# Fields that must ALL be present for a run to be reproducible at the run
# level. The historical corpus has none of them.
PROVENANCE_REQUIRED = ("git_commit", "seed", "L", "zeta", "lam", "T", "N_c")

INDEX_COLUMNS = [
    "snapshot",
    "snapshot_date",
    "collected_host",
    "root",
    "relative_path",
    "format",
    "size_bytes",
    "mtime_utc",
    "likely_campaign",
    "status",
    "L", "zeta", "lam", "alpha", "w", "T", "N_c", "seed", "n_real",
    "dtau_mult", "delta_tau", "burn_in", "entropy_stride",
    "algorithm_version", "git_commit", "git_dirty", "hostname",
    "scheduler_job_id", "solver_method", "jump_update_method",
    "task_id", "wall_time", "n_collapses", "CMI_mean", "B_L_mean",
    "provenance_complete",
    "missing_provenance_fields",
    "note",
]

_SNAP_RE = re.compile(r"ruche_snapshot_(\d{4}-\d{2}-\d{2})")


class SnapshotError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Snapshot access — tarball or directory, same interface
# ---------------------------------------------------------------------------

class Snapshot:
    """Read-only accessor for one snapshot, tarred or unpacked."""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self._tar: Optional[tarfile.TarFile] = None
        self._prefix = ""

        if path.is_dir():
            self.kind = "dir"
        elif path.suffixes[-2:] == [".tar", ".gz"] or path.suffix in (".tgz", ".gz"):
            self.kind = "tar"
            self._tar = tarfile.open(path, "r:gz")
            # Snapshots are packed as ruche_snapshot_<date>/... — find the root.
            roots = {
                m.name.split("/")[0] for m in self._tar.getmembers()
                if "/" in m.name
            }
            if len(roots) == 1:
                self._prefix = roots.pop() + "/"
        else:
            raise SnapshotError(
                f"{path} is neither a directory nor a .tar.gz snapshot"
            )

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()

    def __enter__(self) -> "Snapshot":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def read_text(self, member: str) -> Optional[str]:
        try:
            if self.kind == "dir":
                p = self.path / member
                return p.read_text(errors="replace") if p.is_file() else None
            assert self._tar is not None
            fh = self._tar.extractfile(self._prefix + member)
            if fh is None:
                return None
            return fh.read().decode("utf-8", "replace")
        except (KeyError, OSError, tarfile.TarError):
            return None

    @property
    def date(self) -> str:
        m = _SNAP_RE.search(self.name)
        return m.group(1) if m else ""

    def collected_host(self) -> str:
        env = self.read_text("environment.txt") or ""
        for line in env.splitlines():
            if line.startswith("hostname:"):
                return line.split(":", 1)[1].strip()
        readme = self.read_text("README.txt") or ""
        for line in readme.splitlines():
            if line.startswith("hostname"):
                return line.split(":", 1)[1].strip()
        return ""

    def result_rows(self) -> Iterator[dict]:
        text = self.read_text("result_inventory.tsv")
        if text is None:
            raise SnapshotError(
                f"{self.name}: no result_inventory.tsv — not a snapshot, or the "
                f"collector failed. Check its warnings.txt."
            )
        yield from csv.DictReader(io.StringIO(text), delimiter="\t")

    def warnings(self) -> list[str]:
        text = self.read_text("warnings.txt") or ""
        return [ln for ln in text.splitlines() if ln.strip()]

    def code_commit(self) -> str:
        info = self.read_text("git_info.txt") or ""
        lines = info.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("## git rev-parse HEAD") and i + 1 < len(lines):
                cand = lines[i + 1].strip()
                if re.fullmatch(r"[0-9a-f]{40}", cand):
                    return cand
        return ""


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _assess_provenance(row: dict) -> tuple[bool, str]:
    missing = [f for f in PROVENANCE_REQUIRED if not row.get(f, "").strip()]
    return (not missing), ",".join(missing)


def build_index(snapshots: list[Path], out_path: Path) -> dict:
    stats = {
        "snapshots": 0, "rows": 0, "complete": 0, "incomplete": 0,
        "warnings": [], "truncated": False,
    }

    with out_path.open("w", newline="") as out_fh:
        writer = csv.DictWriter(
            out_fh, fieldnames=INDEX_COLUMNS, extrasaction="ignore", restval=""
        )
        writer.writeheader()

        for snap_path in snapshots:
            try:
                with Snapshot(snap_path) as snap:
                    host = snap.collected_host()
                    commit = snap.code_commit()
                    warns = snap.warnings()
                    for w in warns:
                        stats["warnings"].append(f"{snap.name}: {w}")
                        if "TRUNCATED" in w or "truncated" in w:
                            stats["truncated"] = True
                    print(f"  {snap.name}: host={host or '?'} "
                          f"code_commit={commit[:12] or '?'} "
                          f"warnings={len(warns)}")

                    n = 0
                    for row in snap.result_rows():
                        complete, missing = _assess_provenance(row)
                        row.update(
                            snapshot=snap.name,
                            snapshot_date=snap.date,
                            collected_host=host,
                            provenance_complete="yes" if complete else "no",
                            missing_provenance_fields=missing,
                        )
                        writer.writerow(row)
                        n += 1
                        stats["rows"] += 1
                        stats["complete" if complete else "incomplete"] += 1
                    print(f"    {n} result file(s) indexed")
                    stats["snapshots"] += 1
            except SnapshotError as exc:
                print(f"  SKIPPED {snap_path.name}: {exc}", file=sys.stderr)
            except (OSError, tarfile.TarError) as exc:
                print(f"  SKIPPED {snap_path.name}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", nargs="*", type=Path,
                    help=".tar.gz bundle(s) or unpacked snapshot director(ies)")
    ap.add_argument("--all", type=Path, default=None, metavar="DIR",
                    help="index every snapshot found under DIR")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV (default: RUCHE_DATA_INDEX.csv beside the input)")
    args = ap.parse_args(argv)

    snapshots: list[Path] = list(args.snapshot)
    if args.all:
        if not args.all.is_dir():
            print(f"--all: {args.all} is not a directory", file=sys.stderr)
            return 2
        snapshots += sorted(
            p for p in args.all.iterdir()
            if _SNAP_RE.search(p.name)
            and (p.is_dir() or p.name.endswith((".tar.gz", ".tgz")))
        )
    if not snapshots:
        ap.error("no snapshots given; pass a path or use --all DIR")

    base = args.all or snapshots[0].parent
    out_path = args.out or (base / "RUCHE_DATA_INDEX.csv")

    print(f"Indexing {len(snapshots)} snapshot(s) -> {out_path}")
    stats = build_index(snapshots, out_path)

    print()
    print(f"snapshots indexed        : {stats['snapshots']}")
    print(f"result files indexed     : {stats['rows']}")
    print(f"  provenance complete    : {stats['complete']}")
    print(f"  provenance INCOMPLETE  : {stats['incomplete']}")
    if stats["rows"]:
        pct = 100.0 * stats["complete"] / stats["rows"]
        print(f"  reproducible fraction  : {pct:.1f}%")
    if stats["warnings"]:
        print(f"\ncollector warnings ({len(stats['warnings'])}):")
        for w in stats["warnings"][:20]:
            print(f"  {w}")
    if stats["truncated"]:
        print(
            "\nWARNING: at least one snapshot was TRUNCATED by a collector cap.\n"
            "         This index is therefore NOT a complete picture of the\n"
            "         cluster, and must not be read as one."
        )
    print(
        "\nReminder: this index is an inventory of what exists on Ruche.\n"
        "It is provenance, never support. Nothing here may be cited as\n"
        "evidence for a scientific claim without going through a proposal."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
