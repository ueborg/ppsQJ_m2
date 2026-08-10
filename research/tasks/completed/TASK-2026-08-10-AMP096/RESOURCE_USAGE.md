# RESOURCE_USAGE — TASK-2026-08-10-AMP096

**Non-authoritative.** Never scientific evidence. **Resource diagnosis only —
this file makes no assessment of the run's scientific conclusions**, which are
reviewed separately.

Measured 2026-08-10 from Claude Code workflow transcripts under
`~/.claude/projects/.../subagents/workflows/wf_*` (per-agent `.meta.json` for
the model and `.jsonl` `usage` blocks for tokens). Written after
`research/RESOURCE_POLICY.md` was adopted, so it is also the baseline the policy
is meant to improve on.

## Headline

**Every worker ran on `claude-opus-5`, including all three investigators.** The
agent definitions said `model: inherit`, so they silently took the lead's model.
Nothing in the run required Opus-level reasoning from `literature` or
`numerics`. This is the single largest correctable inefficiency and is now fixed
by the explicit routing table in `RESOURCE_POLICY.md` §5.4.

## Workers actually invoked

| run | role | model | output tok | cache create | cache read | API msgs | tool uses | kept? |
|---|---|---|---|---|---|---|---|---|
| `wf_5e07aa0c` | theory | `claude-opus-5` | 34,104 | 149,074 | 1,428,095 | 32 | 18 | **yes** |
| `wf_5e07aa0c` | numerics | `claude-opus-5` | 53,670 | 306,399 | 5,271,599 | 68 | 43 | **yes** |
| `wf_5e07aa0c` | literature | `claude-opus-5` | 26,588 | 284,450 | 7,003,315 | 89 | 51 | **yes** |
| `wf_d44983e8` | red-team | `claude-opus-5` | 21,767 | 196,948 | 2,636,617 | 48 | 27 | **yes** |
| `wf_c2930383` | general-purpose ×3 | `claude-opus-5` | 34,957 | 691,333 | 7,505,899 | 127 | 79 | **NO — discarded** |

Kept work: **136,129 output tokens**, ~16.3 M cache reads.
Wasted work: **34,957 output tokens**, ~7.5 M cache reads — **26% of output and
31% of all cache reads produced nothing.**

Reported workflow totals for the kept runs were 350,513 and 101,497 subagent
tokens (452,010 combined).

## Retries, restarts and fallbacks — four wasted launches

| run | outcome | cost | cause |
|---|---|---|---|
| `wf_36fdf6f1` | failed instantly, 0 agents | ~0 | `args` passed as a JSON string; script read `args.question` as undefined |
| `wf_e223b502` | 3 agents, all errored, 0 tokens | ~0 | project agent types not registered in that session |
| `wf_c2930383` | 3 agents ran to near-completion, **all discarded** | 35k output, 7.5M cache | **generic `general-purpose` fallback**, then killed by a session boundary before returning |
| `wf_5e07aa0c` | 3 agents, succeeded | 114k output | relaunched after agent types registered |

Two policy violations by the standards now written down: a **generic fallback**
was substituted for missing project agents (`RESOURCE_POLICY.md` §5.8 now
forbids this and requires `Infrastructure first`), and an **entire phase was
restarted** rather than one retry (§5.8 permits at most one).

The fallback has been removed from `.claude/workflows/research.js`.

## Duplicated context

Measured from tool-call inputs in each transcript:

| role | loaded full `SKILL.md` | loaded `RESEARCH_CHARTER.md` |
|---|---|---|
| theory | yes | yes |
| numerics | yes | yes |
| literature | yes | yes |
| general-purpose ×3 (discarded) | yes | yes |
| red-team | no | yes |

**Six of seven agents ingested the full ~30 KB charter, and six of seven also
ingested the full Skill** — a document written for the lead. That is the second
large avoidable cost, and it is why
`.claude/skills/research/WORKER_CONTRACT.md` now exists.

`numerics` and `literature` also performed their own repository reconstruction
(43 and 51 tool calls; 9 and 1 distinct files read plus extensive searching),
partly re-deriving scope the lead had already resolved.

Cache-read growth is the visible symptom: `literature` accumulated 7.0 M cache
reads across 89 API messages, i.e. a context that kept growing rather than
staying scoped.

## Report verbosity

| report | approx words | target (regression mode) |
|---|---|---|
| `theory.json` | ~3,570 | ≤ 1,000 |
| `literature.json` | ~3,220 | ≤ 1,000 |
| `numerics.json` | ~3,090 | ≤ 1,000 |

**Roughly 3× over** what the policy now sets for a regression test. The content
was largely decision-relevant, so this is a formatting and scoping failure
rather than padding — but three 3,000-word JSON blobs also became the red team's
input context, multiplying the cost downstream.

## Local compute executed

| what | detail |
|---|---|
| numerics scratch scripts | 8 short Python analyses in `scratch/`, all T0 read-only re-analysis of existing aggregates |
| wall time | seconds each; nothing approached the 10-minute pilot ceiling |
| simulation | **none** |
| HPC | **none submitted; none attempted** |

Compute behaviour was correct. The cost was model usage, not CPU.

## Was any worker unnecessary?

**No — all three investigators produced decision-relevant output**, and
independently converged on the finding that the recorded withdrawal reason was
contradicted. `red-team` was decisive: it killed four of five candidates,
including the one the lead had promoted.

But **all four could have run on Sonnet.** The work was source inspection,
arithmetic reproduction and record-checking. Under the new policy this task —
a historical regression test — routes every worker to Sonnet.

## Efficiency verdict

**Unacceptable for a narrow regression test, and the causes are all structural
rather than inherent to the question.** In order of cost:

1. **Model routing** — four Opus workers where Sonnet was sufficient.
2. **A discarded full phase** — 26% of output tokens, caused by a generic
   fallback plus a phase restart.
3. **Duplicated context** — six of seven agents loading the charter and the
   lead's Skill.
4. **Oversized reports** — 3× the regression-mode budget, compounding into the
   red team's input.
5. **Two launch failures** — cheap in tokens, but they cost a session boundary
   and forced the restart in (2).

All five are addressed by `research/RESOURCE_POLICY.md` and the changes made
alongside it. **None of this bears on whether the run's conclusions were
right**, which is a separate review.
