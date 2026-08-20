# `ruche_inventory` — read-only HPC inventory collector

A small, self-contained package that the **researcher copies to Ruche and runs
manually**. It answers "what actually exists on the cluster?" without any agent
ever touching the cluster.

No component of this repository's automation runs these scripts. They are
operated by a human, on Ruche, by hand.

---

## Safety contract

The collector is read-only with respect to everything except its own output
directory. It **never**:

| never | why it matters |
|---|---|
| submits, cancels or modifies a job | agents never submit HPC jobs, at any gate (`research/RESOURCE_POLICY.md` §4) |
| writes outside `ruche_snapshot_<date>/` | it cannot damage simulation data |
| deletes or moves anything | ditto |
| modifies git state | only read subcommands (`rev-parse`, `status`, `log`, `describe`, `remote -v`) |
| installs packages | the cluster environment is left exactly as found |
| copies files off Ruche | no network egress at all; **you** move the bundle |
| reads credentials | environment capture is a strict allow-list, never a dump |

The only scheduler contact is:

- availability probes (`--version` queries) recorded in `environment.txt`;
- one accounting **query** over your own completed jobs, and only when you
  explicitly pass `--slurm-history`.

Both are read-only. Neither changes queue state. Nothing is ever enqueued,
held, released or cancelled.

You can verify the contract yourself before trusting it:

```bash
sed -n '/SAFETY CONTRACT/,/^# ====/p' collect_inventory.sh
```

---

## Files

| file | role |
|---|---|
| `collect_inventory.sh` | the collector; run this |
| `collect_results_metadata.py` | cheap metadata extraction from result files (called by the above) |
| `expected_output.md` | what the bundle should look like, so you can tell a good run from a broken one |
| `README.md` | this file |

`collect_results_metadata.py` is stdlib-only, so it runs under a bare cluster
`python3`. `numpy` is used if present (it improves `.npz` reading) and its
absence is recorded as a warning rather than a failure.

---

## Usage

```bash
cd ~                     # the snapshot is written into the CWD
./collect_inventory.sh \
    --code-root    "$HOME/ppsQJ_m2" \
    --results-root /gpfs/workdir/$USER/pps_qj \
    --results-root /gpfs/scratch/$USER/pps_qj \
    --tar
```

Options:

| flag | default | meaning |
|---|---|---|
| `--code-root DIR` | `$HOME/ppsQJ_m2` | git checkout to report on |
| `--results-root DIR` | placeholders — **set these** | result tree to index; repeatable |
| `--out DIR` | `ruche_snapshot_<date>` | output directory |
| `--max-depth N` | 8 | recursion depth under each results root |
| `--max-files N` | 200000 | hard cap on indexed files |
| `--max-scripts N` | 5000 | hard cap on checksummed scripts/configs |
| `--tar` | off | also write `<out>.tar.gz` |
| `--slurm-history [DAYS]` | off, 180 | include the scheduler accounting export |

Environment equivalents: `RUCHE_CODE_ROOT`, `RUCHE_RESULTS_ROOT`,
`RUCHE_MAX_FILES`, `RUCHE_MAX_SCRIPTS`.

---

## Cost and footprint

The collector `stat`s files and reads only small headers; it does not read bulk
arrays. On a Linux cluster it takes the fast `find -printf` path, so indexing
10^5 files is seconds, not minutes.

Bundle size is dominated by `file_inventory.tsv` (one line per file). A
100,000-file tree gives roughly a 15 MB TSV, ~2 MB compressed. If that is too
large, lower `--max-depth` or point `--results-root` at a subtree.

**No simulation arrays are ever copied into the bundle.**

---

## What it does *not* establish

The snapshot is an index of what exists. It is **not** scientific provenance.

The historical corpus has no `git_commit`, no `seed` and no `burn_in` in any
file (`TASK-2026-08-14-C2CONV`, `NEXT_NUMERICS_QUESTION.md` §5), so for old
runs the collector will legitimately report those columns empty. That is a true
record of an unreproducible corpus, not a collector bug.

Files written by the current production entry point
(`pps_qj.production.run`) carry a full embedded provenance record, and the
collector extracts it. The contrast between the two is the point.
