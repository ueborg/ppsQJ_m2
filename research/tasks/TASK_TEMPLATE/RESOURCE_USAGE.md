# RESOURCE_USAGE — <TASK-ID>

**Non-authoritative.** Never scientific evidence, never cited as support.
Written by the lead after the run, per `research/RESOURCE_POLICY.md` §5.10.

## Model routing

Posture (`economical` | `normal` | `deep`), and why it was chosen:
`/research` defaults to `normal`; regression/historical validation is
`economical`. Policy: `research/RESOURCE_POLICY.md` §5.4, table:
`research/model_routing.yaml`.

| tier | alias | invocations this run |
|---|---|---|
| tier 1 | `sonnet` | |
| tier 2 | `opus` | |
| tier 3 | `best` | |

Any Tier-3 request the runtime degraded, and to what (`best` → `fable` → `opus`
is the chain; a degradation is not a failure, but the tier recorded must be the
tier actually used):

### Escalations

One five-field record per escalation above the role's posture default
(§5.4e), plus the outcome. **No failed cheaper pass is required first**
(§5.4d) — record the expected gain, not a justification essay.

```
MODEL_ESCALATION:
  from:
  to:
  role:
  question:
  decision_at_stake:
  material_value:   # changed_conclusion | new_derivation | caught_error |
                    # confirmed_existing | no_material_gain
```

Escalations **refused** by the router (missing record, regression mode), and
whether the refusal cost anything:

### Was the stronger model worth it? (§5.11)

- Which escalations returned `changed_conclusion`, `new_derivation` or
  `caught_error` — and what would the Tier-1/2 path have cost had they not?
- Which returned `confirmed_existing` or `no_material_gain`? Several of these in
  a row means the escalation triggers are too loose.
- Was any *conceptual* bottleneck handled by a high-effort cheap search instead
  of a stronger model (§5.4f)? That is the failure this pass exists to prevent.
- Was any *mechanical* work sent to Tier 2/3? That is the opposite failure, and
  the 2026-08-10 all-strongest-model run is the standing example.

## Workers

| role | invoked? | tier | model | effort | approx tokens | retries | decision-relevant output? |
|---|---|---|---|---|---|---|---|
| literature | | | | | | | |
| theory | | | | | | | |
| numerics | | | | | | | |
| red-team | | | | | | | |

Exact token accounting is **not required** — Claude Code does not expose it
reliably per subagent. Record what is exposed and leave the rest blank.

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
- Would a cheaper model have produced the same decision — **and** would a
  stronger one have produced a better one? Both questions are live now; only
  the first was before.
- Overall: was the information gained worth the model usage spent?
