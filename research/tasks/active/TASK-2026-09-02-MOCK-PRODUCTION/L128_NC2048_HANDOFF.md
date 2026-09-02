# The L = 128, N_c = 2048 rung — handed over, not duplicated

TASK-2026-09-02-MOCK-PRODUCTION, brief §15.

## Status: already prepared. Not touched by this task.

**[E]** The cell

```
L = 128, T = 128, zeta = 0.35, lambda = 0.3032, N_c = 2048,
dtau_mult = 6.0, systematic, R = 16, seeds 30500000-30500015
```

is packaged at

```
research/tasks/active/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/armA2048_optional/
```

with its own `manifest.csv` (16 rows), `submit.slurm` (`--array=0-15%64`,
`cpu_long`, `--time=48:00:00`, `--mem=9G`), `preflight.py` and `run_cell.py`.
That preflight passed and its ten injected-fault negative controls were recorded
in that task's `VALIDATION.md`. Its results directory is empty: **it was never
submitted.**

**This task does not duplicate it, regenerate it, modify it or re-cost it into
its own budget.** It appears here as a separate recommended human submission,
because it answers the remaining central `N_c`-convergence question and has a
runtime far outside this campaign's ≈3 h envelope.

## Why it still matters

**[E]** The completed ARM A rungs at `L = 128, lambda = 0.3032` give:

| N_c | R | mean CMI | SEM |
|---:|---:|---:|---:|
| 64 | 64 | 0.51957 | 0.02494 |
| 128 | 64 | 0.42059 | 0.02354 |
| 256 | 64 | 0.29932 | 0.01679 |
| 512 | 48 | 0.25109 | 0.02164 |
| 1024 | 32 | 0.19088 | 0.00898 |

with the direct rung-to-rung difference `Delta_512->1024 = −0.06021`,
95 % CI `[−0.1104, −0.0203]` — **excluding zero**. **[I]** At `L = 128` the mean
is therefore *still moving* at `N_c = 1024`, and the programme does not yet know
where it stops. `N_c = 2048` is the next direct rung-to-rung test, and no
`1/N_c` fit substitutes for it (`TASK-2026-08-31-SMCCERT` killed that claim).

That question is orthogonal to this campaign, which asks whether a *whole scan*
at a *cheap* `L` behaves like production. Both are worth answering; only one of
them fits in three hours.

## The cost has changed, and the human should know before submitting

**[E]** The predecessor costed this arm at **20.12 h predicted / 28.16 h
pessimistic** per task, using a rate model that extrapolated the `L = 128,
N_c = 256` rate (21.522 ms) flat to higher `N_c`. Its own returned results
falsify that assumption: the rate **rises** with `N_c` above 256 —
21.522 → 23.416 → 27.898 ms at `N_c` = 256 → 512 → 1024 (`COST_MODEL.md` §1b).

**[I]** Continuing that trend for one further doubling, at the `+19.1 %` rate
observed for the 512 → 1024 step:

```
rate(L=128, N_c=2048) ~ 27.898 * 1.20 = 33.478 ms
wall  = 33.478e-3 * 2048 * 1643 = 112 620 s = 31.3 h per task
```

| | predecessor's figure | re-measured here |
|---|---:|---:|
| slowest task | 20.12 h | **31.29 h** |
| pessimistic (×1.40) | 28.16 h | **43.81 h** |
| core-hours (R = 16) | 321.9 | **500.7** |
| `--time` in the packaged script | 48:00:00 | unchanged |

**The packaged `--time=48:00:00` still covers it**, with 43.81 h pessimistic
against a 48 h limit — a **9.5 % margin**, where the predecessor believed it had
70 %. **[J]** That is thinner than it looks on paper and the human should decide
deliberately rather than inherit it. Two ways to widen it:

- the arm is 16 independent tasks in one array, so a task that hits the wall can
  simply be resubmitted (`run_cell.py` is idempotent and skips completed rows),
  losing that task's work but not the arm's;
- or raise `--time` in that package before submitting. `cpu_long`'s MaxTime is
  7 days, so `--time=72:00:00` is valid and would restore a comfortable margin.
  **Editing the predecessor's `submit.slurm` is a change to a sibling task's
  archive and this task has deliberately not made it.** If the human wants it,
  it is a one-line edit followed by re-running that arm's `run_preflight.sh`,
  which will re-derive the partition and re-check the array range.

Note the preflight in that package carries the **old** rate table, so it will
report the old 20.12 h figure and pass. That is not a defect in it — it was
correct against the evidence available when it was written — but it means the
preflight will not warn about the margin. This file is the warning.

## Recommendation

**Submit it separately from, and after, the mock-production campaign.**
It is 500.7 core-hours in 16 tasks, elapsed ≈ 31 h (44 h pessimistic), and it is
in no way a dependency of anything in this task. Queueing it alongside the
mock-production arrays is harmless if the allocation grants per-array slots, but
it will still be running long after the 3-hour campaign has finished and been
analysed.

The predecessor's own decision rule was "queue it if and only if F2 comes back
SUPPORTED". **[E]** F2 came back **INCONCLUSIVE** — `Delta_512->1024` excludes
zero but `|Delta| = 0.060 < tau_step = 0.0732`, so the mean is demonstrably
still moving yet by less than one lambda-grid step. **[J]** That is precisely
the situation in which the next rung is informative either way: if
`Delta_1024->2048` is small, the ladder has converged at a resolution that
matters for the production plots; if it is not, `N_c = 1024` at `L = 128` is not
enough and the production budget has to change. Recommending it is a judgement,
and the human's call.

The exact commands are in that task's own `RUCHE_RUNBOOK.md` §5. Nothing in this
task submits it, and nothing in this task may.
