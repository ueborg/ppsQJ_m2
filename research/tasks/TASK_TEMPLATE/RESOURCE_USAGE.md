# RESOURCE_USAGE — <TASK-ID>

**Non-authoritative.** Never scientific evidence, never cited as support.
Written by the lead after the run, per `research/RESOURCE_POLICY.md` §5.10.

## Workers

| role | invoked? | model | approx tokens | retries | decision-relevant output? |
|---|---|---|---|---|---|
| literature | | | | | |
| theory | | | | | |
| numerics | | | | | |
| red-team | | | | | |

Roles **not** spawned, and why:

## Retries and failures

| role | failure | transient? | retried? | reused completed work? |
|---|---|---|---|---|

No generic fallback is permitted. A missing project agent means
`Infrastructure first`.

## Local compute executed

| what | wall time | peak memory | threads | output size | decision it changed |
|---|---|---|---|---|---|

`none` is the expected entry for a `/research` run.

## HPC

Submitted by an agent: **never**. If a package was prepared, its terminal state
and where it is:

## Efficiency verdict

- Was any worker unnecessary in hindsight?
- Did any worker duplicate the lead's orientation or another worker's reading?
- Was context wasted (full Skill loaded, repository reconstructed, oversized
  reports)?
- Would a cheaper model have produced the same decision?
- Overall: was the information gained worth the model usage spent?
