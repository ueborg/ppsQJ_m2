# NOVELTY_GATE — <TASK-ID>            (Charter §4.2, §6 warning 12)

**Mandatory before the lead may call any candidate new, novel, a finding, or a
contribution.** Freezes at `stage_3_candidates`.

Run, for every candidate:

```bash
.venv/bin/python3 research/tools/find_predecessors.py "<candidate statement>"
```

It searches claims, disputes, decisions and observables **including withdrawn,
contradicted, superseded and negative-result entries**, which are boosted rather
than filtered — a dead record is the thing most likely to be rediscovered,
precisely because nobody has it in their working set.

## Closest predecessor, per candidate

| candidate | closest predecessor | its status | classification | why |
|---|---|---|---|---|
| C1 | | | | |

`classification` is one of:

- **replication** — same result, independently obtained
- **corroboration** — same conclusion, different evidence or method
- **rediscovery** — we already knew this and did not notice
- **provenance repair** — the record was wrong about its own history
- **no predecessor found** — and note that this is a statement about the
  search, not about the field (charter §3: an agent may not certify novelty)

## The regression case this gate exists for

In `TASK-2026-08-10-AMP096` the lead called candidate C2 "the finding of the
task". `METH-EXTRAP-001` — status `withdrawn`, and so invisible to anyone
reading only live claims — already recorded the same content, with the same
four-form chi²/dof table. Classification should have been **rediscovery**.

## External prior-art search

The canonical predecessor search covers **our own records only**.
`research/state/sources/**` is not an exhaustive corpus, so a canonical miss
says nothing about the literature. Where novelty is scientifically relevant,
`literature` performs an external prior-art search and records it here.

| candidate | external queries run | sources inspected (EXT-*) | prior art found? |
|---|---|---|---|

Task-verified external prior art (`EXT-*` in `TASK_EVIDENCE.yaml`) counts as
evidence here **only at inspection levels above snippet/abstract**.

**`no predecessor found` means exactly: none found under the searches actually
performed.** It does not mean "novel in the literature", and this file must say
so. If the task's scientific value depends materially on novelty and the
external coverage is inadequate, classify novelty as **unresolved** and return
`Infrastructure first` rather than implying a discovery.

## Statements of novelty

None may appear anywhere in this task's output unless the table above classifies
the candidate as `no predecessor found`, **and even then novelty is the
researcher's call, not this task's.**
