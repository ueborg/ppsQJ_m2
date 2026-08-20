#!/usr/bin/env python3
"""Extract CHEAP scientific metadata from recognised result files.

Part of the read-only ruche_inventory package.  This script:
  * reads only file headers / small members / small text prefixes
  * NEVER loads a large array into memory
  * NEVER writes outside the output TSV it is given
  * NEVER contacts the network, the scheduler, or git

Supported formats
-----------------
  .npz   zip directory listing; only members whose UNCOMPRESSED size is below
         --max-member-bytes are actually decoded.  This is how we read
         L / zeta / lambda / T / N_c / seed without touching final_covs.
  .json  parsed only if the file is below --max-text-bytes
  .yaml  scanned line-wise for top-level scalar keys (no YAML dependency)
  .csv   header row plus row count only
  .pkl   NOT unpickled — unpickling executes code.  Size/mtime only.
  .npy   dtype/shape from the 128-byte header only.

Written to be stdlib-only so it runs under a bare cluster python3.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import zipfile
from pathlib import Path

# Keys worth extracting.  Anything else is ignored, which keeps the output
# small and avoids accidentally lifting bulk data into the inventory.
WANTED = [
    "L", "zeta", "lam", "lambda", "alpha", "w", "T", "N_c", "Nc", "seed",
    "n_real", "realizations", "dtau_mult", "delta_tau", "burn_in",
    "n_burnin_frac", "entropy_stride", "task_id", "status", "wall_time",
    "algorithm_version", "code_version", "output_schema_version",
    "git_commit", "git_dirty", "hostname", "timestamp_utc",
    "scheduler_job_id", "solver_method", "jump_update_method",
    "proposal_scheme", "n_collapses", "ESS_mean", "CMI_mean", "B_L_mean",
]
# Normalise a few historical spellings onto one column name.
ALIASES = {"lambda": "lam", "Nc": "N_c", "realizations": "n_real"}

OUT_COLUMNS = [
    "root", "relative_path", "format", "size_bytes", "mtime_utc",
    "likely_campaign", "status",
    "L", "zeta", "lam", "alpha", "w", "T", "N_c", "seed", "n_real",
    "dtau_mult", "delta_tau", "burn_in", "entropy_stride",
    "algorithm_version", "git_commit", "git_dirty", "hostname",
    "scheduler_job_id", "solver_method", "jump_update_method",
    "task_id", "wall_time", "n_collapses", "CMI_mean", "B_L_mean",
    "note",
]

RECOGNISED = (".npz", ".json", ".yaml", ".yml", ".csv", ".npy", ".pkl", ".pickle")


def _scalarise(value):
    """Reduce a decoded value to a short scalar string, or None if it is bulk."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            return None
    if isinstance(value, (str, int, float, bool)):
        s = str(value)
        return s if len(s) <= 200 else None
    # numpy scalars / 0-d / size-1 arrays
    try:
        import numpy as np
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.size == 1:
            return str(arr.reshape(-1)[0])
    except Exception:
        pass
    return None


def _add(row, key, value):
    key = ALIASES.get(key, key)
    if key not in OUT_COLUMNS:
        return
    s = _scalarise(value)
    if s is not None and not row.get(key):
        row[key] = s


# ---------------------------------------------------------------------------
# Format readers
# ---------------------------------------------------------------------------

def read_npz(path: Path, row: dict, max_member: int) -> None:
    """Read small members of a .npz without decompressing the big ones."""
    try:
        import numpy as np
    except ImportError:
        row["status"] = "numpy_unavailable"
        return
    try:
        with zipfile.ZipFile(path) as zf:
            infos = {i.filename: i for i in zf.infolist()}
            row["note"] = f"npz members={len(infos)}"

            # Production files embed their whole provenance record as JSON.
            for cand in ("provenance_json.npy", "provenance_json"):
                info = infos.get(cand)
                if info is not None and info.file_size <= 4 * max_member:
                    try:
                        with zf.open(info) as fh:
                            obj = np.load(fh, allow_pickle=False)
                        prov = json.loads(str(obj))
                        _flatten_provenance(prov, row)
                        row["status"] = "ok_provenance"
                        return
                    except Exception:
                        pass  # fall through to member-wise reading

            for name, info in infos.items():
                key = name[:-4] if name.endswith(".npy") else name
                base = ALIASES.get(key, key)
                if base not in OUT_COLUMNS and key not in WANTED:
                    continue
                if info.file_size > max_member:
                    continue  # bulk array — never decoded
                try:
                    with zf.open(info) as fh:
                        val = np.load(fh, allow_pickle=False)
                except Exception:
                    continue
                _add(row, key, val)
            row["status"] = row.get("status") or "ok"
    except zipfile.BadZipFile:
        row["status"] = "corrupt_or_truncated"
    except Exception as exc:
        row["status"] = f"error:{type(exc).__name__}"


def _flatten_provenance(prov: dict, row: dict) -> None:
    """Pull the inventory columns out of a production provenance record."""
    cfg = prov.get("config", {}) or {}
    for k in ("L", "zeta", "lam", "alpha", "w", "T", "N_c", "seed",
              "realizations", "dtau_mult", "delta_tau", "entropy_stride",
              "n_burnin_frac"):
        _add(row, k, cfg.get(k))
    _add(row, "burn_in", cfg.get("n_burnin_steps"))
    _add(row, "algorithm_version", prov.get("algorithm_version"))
    git = prov.get("git", {}) or {}
    _add(row, "git_commit", git.get("git_commit"))
    _add(row, "git_dirty", git.get("git_dirty"))
    env = prov.get("environment", {}) or {}
    _add(row, "hostname", env.get("hostname"))
    _add(row, "scheduler_job_id", env.get("scheduler_job_id"))
    algo = prov.get("algorithm", {}) or {}
    _add(row, "solver_method", algo.get("waiting_time_solver"))
    _add(row, "jump_update_method", algo.get("jump_update"))
    _add(row, "wall_time", prov.get("runtime_seconds"))
    summary = prov.get("summary", {}) or {}
    _add(row, "CMI_mean", summary.get("CMI_mean"))
    _add(row, "B_L_mean", summary.get("B_L_mean"))


def read_json(path: Path, row: dict, max_text: int) -> None:
    if path.stat().st_size > max_text:
        row["status"] = "skipped_too_large"
        return
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        row["status"] = f"unparsable:{type(exc).__name__}"
        return
    if not isinstance(data, dict):
        row["status"] = "ok_non_mapping"
        return
    if "provenance_schema_version" in data or "algorithm_version" in data:
        _flatten_provenance(data, row)
        row["status"] = "ok_provenance"
        return
    for k in WANTED:
        if k in data:
            _add(row, k, data[k])
    row["status"] = "ok"


def read_yaml(path: Path, row: dict, max_text: int) -> None:
    """Top-level `key: scalar` scan. No YAML library required."""
    if path.stat().st_size > max_text:
        row["status"] = "skipped_too_large"
        return
    pat = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$")
    try:
        for line in path.read_text(errors="replace").splitlines():
            m = pat.match(line)
            if not m:
                continue
            key, raw = m.group(1), m.group(2).strip().strip("'\"")
            if ALIASES.get(key, key) in OUT_COLUMNS:
                _add(row, key, raw)
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = f"error:{type(exc).__name__}"


def read_csv_head(path: Path, row: dict) -> None:
    try:
        with path.open(newline="", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            n = sum(1 for _ in reader)
        row["note"] = f"csv cols={len(header)} rows={n}"
        row["status"] = "ok_header_only"
    except Exception as exc:
        row["status"] = f"error:{type(exc).__name__}"


def read_npy_header(path: Path, row: dict) -> None:
    """dtype/shape from the .npy header only — the array is never read."""
    try:
        with path.open("rb") as fh:
            magic = fh.read(6)
            if magic != b"\x93NUMPY":
                row["status"] = "not_npy"
                return
            major = fh.read(1)[0]
            fh.read(1)
            hlen = int.from_bytes(fh.read(2 if major == 1 else 4), "little")
            header = fh.read(hlen).decode("latin1")
        meta = ast.literal_eval(header.strip())
        row["note"] = f"npy dtype={meta.get('descr')} shape={meta.get('shape')}"
        row["status"] = "ok_header_only"
    except Exception as exc:
        row["status"] = f"error:{type(exc).__name__}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file-inventory", required=True,
                    help="file_inventory.tsv produced by collect_inventory.sh")
    ap.add_argument("--out", required=True, help="result_inventory.tsv to write")
    ap.add_argument("--warnings", default=None)
    ap.add_argument("--max-member-bytes", type=int, default=64 * 1024,
                    help="largest .npz member that will be decoded")
    ap.add_argument("--max-text-bytes", type=int, default=2 * 1024 * 1024,
                    help="largest json/yaml file that will be parsed")
    ap.add_argument("--max-files", type=int, default=100000)
    args = ap.parse_args(argv)

    warn_fh = open(args.warnings, "a") if args.warnings else None

    def warn(msg: str) -> None:
        print(f"WARNING: {msg}", file=sys.stderr)
        if warn_fh:
            warn_fh.write(f"WARNING: {msg}\n")

    inv = Path(args.file_inventory)
    if not inv.exists():
        warn(f"{inv} not found")
        return 1

    n_seen = n_written = 0
    with inv.open(newline="") as fh, open(args.out, "w", newline="") as out_fh:
        reader = csv.DictReader(fh, delimiter="\t")
        writer = csv.DictWriter(
            out_fh, fieldnames=OUT_COLUMNS, delimiter="\t",
            extrasaction="ignore", restval="",
        )
        writer.writeheader()

        for rec in reader:
            n_seen += 1
            if n_seen > args.max_files:
                warn(f"max-files {args.max_files} reached — metadata TRUNCATED")
                break
            root = rec.get("root", "")
            rel = rec.get("relative_path", "")
            ext = ("." + rec.get("extension", "")).lower()
            if ext not in RECOGNISED:
                continue

            path = Path(root) / rel
            row = {c: "" for c in OUT_COLUMNS}
            row.update(
                root=root, relative_path=rel, format=ext.lstrip("."),
                size_bytes=rec.get("size_bytes", ""),
                mtime_utc=rec.get("mtime_utc", ""),
                likely_campaign=rec.get("likely_campaign", ""),
            )
            try:
                if not path.is_file():
                    row["status"] = "vanished"
                elif ext == ".npz":
                    read_npz(path, row, args.max_member_bytes)
                elif ext == ".json":
                    read_json(path, row, args.max_text_bytes)
                elif ext in (".yaml", ".yml"):
                    read_yaml(path, row, args.max_text_bytes)
                elif ext == ".csv":
                    read_csv_head(path, row)
                elif ext == ".npy":
                    read_npy_header(path, row)
                elif ext in (".pkl", ".pickle"):
                    # Unpickling executes arbitrary code. Never do it here.
                    row["status"] = "skipped_pickle_not_unpickled"
                    row["note"] = "unpickling executes code; size/mtime only"
            except OSError as exc:
                row["status"] = f"unreadable:{exc.__class__.__name__}"

            writer.writerow(row)
            n_written += 1

    print(f"  result metadata: {n_written} file(s) described "
          f"from {n_seen} indexed")
    if warn_fh:
        warn_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
