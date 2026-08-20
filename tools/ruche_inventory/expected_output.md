# Expected output of `collect_inventory.sh`

Use this to decide whether a run on Ruche worked, before you spend time moving
the bundle home.

---

## Directory layout

```
ruche_snapshot_YYYY-MM-DD/
    README.txt              collection parameters and a manifest
    git_info.txt            code checkout state at --code-root
    environment.txt         host, CPU, python, packages, modules, scheduler probe
    file_inventory.tsv      one row per indexed file
    result_inventory.tsv    cheap parameter metadata from recognised outputs
    scripts_inventory.tsv   sha256 of scripts/configs under --code-root
    slurm_history.tsv       only if you passed --slurm-history
    warnings.txt            anything that could not be collected
    checksums.txt           sha256 of every file in the snapshot
ruche_snapshot_YYYY-MM-DD.tar.gz   only if you passed --tar
```

---

## Column schemas

**`file_inventory.tsv`**

```
root  relative_path  size_bytes  mtime_utc  extension  directory  likely_campaign
```

`likely_campaign` is a **path heuristic only** — it is derived from directory
and file names, never from file contents, and `unknown` is a normal value. It
is an orientation aid, not a classification you may cite.

**`result_inventory.tsv`**

```
root  relative_path  format  size_bytes  mtime_utc  likely_campaign  status
L  zeta  lam  alpha  w  T  N_c  seed  n_real  dtau_mult  delta_tau
burn_in  entropy_stride  algorithm_version  git_commit  git_dirty  hostname
scheduler_job_id  solver_method  jump_update_method  task_id  wall_time
n_collapses  CMI_mean  B_L_mean  note
```

`status` values and what they mean:

| status | meaning |
|---|---|
| `ok_provenance` | a full production provenance record was found and unpacked |
| `ok` | individual parameter fields were read |
| `ok_header_only` | only shape/dtype or a CSV header was read |
| `skipped_pickle_not_unpickled` | `.pkl` seen; **never** unpickled, since unpickling executes code |
| `skipped_too_large` | text file above the parse cap |
| `corrupt_or_truncated` | `.npz` failed to open as a zip — worth investigating |
| `unparsable:*`, `unreadable:*`, `error:*` | see the suffix |
| `vanished` | file disappeared between indexing and reading |

**`scripts_inventory.tsv`**

```
relative_path  size_bytes  mtime_utc  sha256
```

**`slurm_history.tsv`** (only with `--slurm-history`) — tab-separated
accounting export with job id, name, partition, state, exit code, submit/start/
end, elapsed, CPUs, nodes, requested memory, max RSS, CPU time, and work dir.

---

## What a healthy run looks like

```
ruche_inventory v1.0 — read-only snapshot into ruche_snapshot_2026-08-20
  indexing /gpfs/workdir/<user>/pps_qj ...
  indexed 48213 files
  checksummed 612 script/config file(s)
  extracting result metadata ...
  result metadata: 4192 file(s) described from 48213 indexed
  (skipping slurm history; pass --slurm-history to include it)

Done. Snapshot: ruche_snapshot_2026-08-20
  size: 8.4M
  warnings: 0 line(s) — see ruche_snapshot_2026-08-20/warnings.txt

Nothing was submitted, cancelled, modified or transferred off this cluster.
```

Sanity checks before you move it:

1. `git_info.txt` shows a **40-character commit hash**, not `NOT A GIT
   CHECKOUT`. If it does not, `--code-root` is wrong.
2. `file_inventory.tsv` has more than one line. If it has only the header,
   every `--results-root` was wrong or empty.
3. `wc -l result_inventory.tsv` is non-trivial, and at least some rows have a
   populated `L` / `zeta` / `lam`.
4. `warnings.txt` is empty or contains only things you expect.
5. The bundle is **megabytes, not gigabytes**. If it is huge, `--max-depth` is
   too deep.

---

## Expected warnings, and what they mean

| warning | meaning | action |
|---|---|---|
| `results root … does not exist — skipped` | a `--results-root` path is wrong | fix the path and re-run |
| `find(1) lacks -printf` | BSD `find` (a Mac, not Ruche) | fine, just slower |
| `numpy: NOT AVAILABLE` in `environment.txt` | cluster python has no numpy | `.npz` metadata will be thin; load the project venv first |
| `file cap … reached — inventory TRUNCATED` | more files than `--max-files` | raise the cap or narrow the roots — **do not** silently accept it |
| `scripts_inventory truncated at N entries` | more scripts than `--max-scripts` | as above |
| `no sha256 tool` | neither `sha256sum` nor `shasum` present | checksums will be blank |

Truncation warnings matter: a truncated inventory looks exactly like a complete
one downstream. If you see one, record it, because the local importer will
otherwise treat the snapshot as a full picture of the cluster.

---

## Rough size guide

| indexed files | `file_inventory.tsv` | bundle `.tar.gz` |
|---|---|---|
| 1,000 | ~120 KB | ~20 KB |
| 10,000 | ~1.2 MB | ~200 KB |
| 100,000 | ~15 MB | ~2 MB |

Bulk simulation arrays are never included at any size.
