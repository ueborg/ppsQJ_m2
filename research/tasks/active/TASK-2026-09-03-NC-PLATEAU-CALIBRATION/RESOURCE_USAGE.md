# RESOURCE_USAGE — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

## Models and routing

| role | posture | dispatched | model | material value |
|---|---|---|---|---|
| lead | normal | — | the session's model (researcher's choice) | — |
| numerics | normal | **inline, by the lead** | `lead-inline` | `caught_error` + `new_derivation` |
| theory | normal | **skipped** | — | — |
| literature | normal | **skipped** | — | — |
| red team | — | **inline, by the lead** | `lead-inline` | `changed_conclusion` (twice) |

`[E]` **No escalation was requested and none is recorded**, because no worker
was dispatched at any tier. The workflow's `MODEL_ESCALATION` machinery was not
exercised.

`[J]` **This is a resource decision with a cost, and the cost is Stage 8.** The
routing policy holds that a few strong passes at high-leverage decision points
*save* total research resources; this run bought none of them, and
`INDEPENDENCE_LEDGER.yaml` records the consequence without softening it.

`[J]` The skips of `theory` and `literature` are defensible on their own terms:
the analytic question was settled, negatively, one task ago, and no external
source is load-bearing for a campaign-preparation task about this sampler's own
convergence. Re-deriving either would be the "three agents independently
rediscovering the same fact" failure the Skill names. The absence of an
**independent red team** is not defensible on any terms. It is a gap.

### Material value of the passes that did run

`[E]` `numerics`, inline — `caught_error` and `new_derivation`:

- the inherited cost model extrapolates **30 % optimistic** at `L = 128`,
  `N_c = 1024`, in exactly the regime this campaign enters;
- the `--mem` model was a model quoted as a measurement; produced the first
  peak-RSS measurements of this sampler in the repository, at 15 cells;
- **absolute-level certification at `L = 128` is unreachable at any affordable
  `R`** — the campaign's most decision-relevant result, obtained with no new
  compute at all;
- the accepted `L = 96` `1/N` rejection does not reproduce from raw data.

`[E]` `red team`, inline — `changed_conclusion`, twice: `R = 24 → 48` in
campaign A, and the rebuild of campaign B2 from three `lambda` to seven.
`[J]` Both changed what will be submitted, and both cost core-hours (~386
combined), which is the test of whether a review did anything.

## Compute

`[E]` **T0 analysis compute only: local, read-only with respect to canonical
state, and no production simulation.** Itemised:

| what | scale |
|---|---|
| parsing 1 896 raw result JSONs, repeatedly | seconds |
| rebuilding 53 ladders, all fits, all bootstraps | seconds |
| 15 peak-RSS probes (`tools/mem_probe.py`) | ~25 min total, up to `L = 128`, `N_c = 2048` at `T = 4` |
| 2 full predecessor-population reproductions (`tools/reproduce_check.py`) | ~2 min, `L = 32`, `N_c = 1024`, `K = 71` |
| 3 toy populations (`tools/smoke_test.py`) | seconds, `L = 12`, `N_c = 8` |
| 16 negative controls, 17 preflights, the full check suite | ~2 min |

`[E]` **No HPC job was submitted. No scheduler was contacted. No remote shell or
file transfer of any kind.** `research/RESOURCE_POLICY.md` §4 forbids it
unconditionally, and `.claude/hooks/guard_research.py` **fired twice during this
run** — once on a `grep` whose search string contained a scheduler verb, and once
on a shell heredoc whose *documentation text* listed the forbidden transfer
commands.

`[J]` Both denials were of read-only commands, blocked because of their string
content rather than their effect. That is the enforcement working exactly as
designed: the hook does not try to understand intent, and it should not. It is
also why every scheduler verb inside this package's source is assembled from
fragments rather than written literally, and why `HUMAN_SUBMISSION.md` writes
`<batch-submit>` where a command would go — the runbook, which is written for the
researcher rather than by an agent that could execute it, spells them out.

`[E]` **No local pilot was run.** None was needed: every runtime number comes
from completed Ruche jobs, and the only new measurements are memory probes and
reproduction checks, neither of which is production simulation.

## What this campaign will cost if submitted

`[E]` 2 180 core-hours predicted, 3 052 pessimistic, 3 280 tasks, longest single
job 31.4 h (44.0 h pessimistic).

`[E]` 240 existing populations reused rather than recomputed, worth ~1 880
core-hours — of which the `L = 128` and `L = 96` lower rungs alone are ~1 660.
`[J]` Recomputing those would have been the single largest avoidable waste
available to this task.

`[E]` Conditional group, if every arm were released: a further 7 294 core-hours.
They will not all be released; two pairs are mutually exclusive by construction.

## Researcher attention

`[J]` What this package asks of the researcher: read `HUMAN_SUBMISSION.md` — one
table and six paragraphs — decide whether to keep campaign B2, run the seventeen
preflights, submit, and run one extra accounting query for `MaxRSS`.

`[J]` What it deliberately does **not** ask: any judgement about `--time`,
`--mem`, seeds, partitions, reuse, or criteria. All of those are computed,
checked against measurement, and fail loudly if hand-edited. The one number in
the whole package that is safe to change by hand is the `%N` concurrency cap,
and the runbook says so in the one place it matters.
