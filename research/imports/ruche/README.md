# `research/imports/ruche/` — manually supplied HPC snapshots

Drop zone for inventory bundles the **researcher collected by hand on Ruche**
and copied back to this laptop.

Nothing in this repository's automation produced these files, and nothing in it
can refresh them. No agent has, or will have, access to Ruche
(`research/RESOURCE_POLICY.md` §4; `CLAUDE.md` "Working rules"). A snapshot here
is as current as the day the human collected it, and no more.

---

## What goes here

```
research/imports/ruche/
    README.md                              this file
    ruche_snapshot_YYYY-MM-DD.tar.gz       bundle you copied back  (gitignored)
    ruche_snapshot_YYYY-MM-DD/             unpacked bundle         (gitignored)
    RUCHE_DATA_INDEX.csv                   built by the importer   (gitignored)
```

Bundles and the derived index are **gitignored**. They are machine-local data
with no provenance record in `research/state/`, and committing them would be
exactly the mistake the project's data-hygiene rules exist to prevent.

## Building the index

```bash
.venv/bin/python3 research/tools/import_ruche_snapshot.py \
    --all research/imports/ruche/
```

This reads every `ruche_snapshot_*` bundle or directory it finds and writes
`RUCHE_DATA_INDEX.csv` — one row per result file discovered on the cluster,
carrying whatever parameter metadata the collector could cheaply extract.

The importer never contacts Ruche. It reads the bundle and nothing else.

## The column that matters

`provenance_complete` is `yes` only when **all** of `git_commit`, `seed`, `L`,
`zeta`, `lam`, `T`, `N_c` are present on that file.

Expect most of the historical corpus to come back `no`. That is a true finding,
not an import failure: across the 15,139-run historical corpus, `git_commit`,
`seed`, `burn_in` and `job_id` are absent from **every** file
(`TASK-2026-08-14-C2CONV`, `NEXT_NUMERICS_QUESTION.md` §5). Runs written by the
current production entry point carry a full embedded provenance record and come
back `yes`.

The reproducible-fraction number the importer prints is a useful measure of how
much of the cluster's data can be tied to the code that made it.

---

## Epistemic status — read before citing anything from here

A snapshot is an **inventory of what exists**, not evidence about physics.

- It is **provenance**, in the charter's four-tier sense. It is never
  **support** for a claim.
- `likely_campaign` is a filename/path heuristic. It is an orientation aid and
  must never be treated as a classification of what a run actually computed.
- A snapshot is **not** complete scientific provenance even when
  `provenance_complete` is `yes`: it records a file's stated parameters, not
  whether the run was correct, converged, or computed the observable convention
  you think it did.
- **Truncation is silent downstream.** If the collector hit a `--max-files` or
  `--max-scripts` cap, the importer prints a warning and the index is a partial
  picture of the cluster. Read the importer's output, not just the CSV.
- Nothing here may be promoted into `research/state/**` except through a
  proposal that passes red-team review and the human gate.

## Related

| path | role |
|---|---|
| `tools/ruche_inventory/` | the collector you copy TO Ruche |
| `RUCHE_MANUAL_INSTRUCTIONS.md` | step-by-step manual procedure, at the repo root |
| `research/tools/import_ruche_snapshot.py` | the importer that builds the index |
| `research/state/DATA_ROOTS.yaml` | canonical logical data roots — has **no** Ruche root yet; adding one needs a proposal |
