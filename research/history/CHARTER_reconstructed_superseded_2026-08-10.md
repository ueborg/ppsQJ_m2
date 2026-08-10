---
lifecycle: active
authoritative_for: epistemic and procedural rules
owner: human
last_reviewed: 2026-08-10
---

# Research Operating Charter — ppsQJ_m2

Binding on every session, human or agent, that does substantive scientific work
on this project.

> **Provenance note.** No pre-existing charter document was found in the
> repository or in project knowledge. This file was authored on 2026-08-10 from
> the Stage 3 design in `audit/2026-08-10/`. It is a first version and is
> expected to be revised by the human owner. It has not been ratified by anyone
> other than its author.

## 0. Machine-readable preamble

```yaml
charter_version: "1.0"
canonical_state_root: research/state/
non_authoritative:
  - project_memory
  - project_instructions
  - research/HANDOFF.md          # navigation only, no scientific values
  - research/history/**
  - theory/**
  - manuscripts and Overleaf drafts
  - prior conversations
  - research/{proposals,tasks,runs}/**   # until merged
epistemic_status: [verified, plausible, open, contested, superseded, refuted]
reproducibility: [fully_reproducible, partially_reproducible, artifact_only,
                  procedure_only, chat_only, ephemeral_recovered,
                  unknown_recoverable, unrecoverable]
evidence_role: [discriminating, supporting, postdiction, diagnostic]
state_writers: [merge_step_after_human_approval]
hpc_gate: research/experiments/<EXP-ID>.yaml
```

## 1. The two planes

**Knowledge plane.** `research/state/`, this charter, and `research/HANDOFF.md`.
Durable, small, validated, single-writer. Nothing enters except through review.

**Execution plane.** `research/tasks/`, `research/proposals/`, `research/runs/`,
worktrees. Cheap to create, never authoritative.

Agents may read the execution plane but may only **cite** the knowledge plane. A
proposal citing another proposal as support is rejected at review. This is what
prevents two agents bootstrapping each other into a consensus no evidence
supports.

## 2. Authority

For scientific content, authority runs in exactly this order, so that no agent
ever needs to judge which of two documents is newer:

1. `research/state/**`
2. this charter, for procedure
3. `research/HANDOFF.md`, for navigation and what is in flight
4. everything else is non-authoritative

Non-authoritative sources may be read for orientation and cited as
**provenance** ("memory asserts X"). They may never be cited as **support**.

## 3. Two independent axes

Scientific confidence and reproducibility are orthogonal and are never
collapsed. This project has a result that is well argued with a lost script, and
code that is perfectly preserved supporting a dead conclusion.

- Epistemic status lives on claims.
- Reproducibility lives on evidence.

**Coupling rule.** A claim reaches `verified` only if it cites at least one
`discriminating` evidence item whose reproducibility is `fully_reproducible` or
`partially_reproducible`. Evidence with reproducibility `chat_only` caps the
claims it supports at `plausible`, however convincing the reasoning.

## 4. What may never make a claim verified

Memory. HANDOFF. A manuscript. A previous conversation. Agreement among agents.
A derivation that looks plausible. A single-cell or single-L result. A proxy
metric. Reproducing a number that was already known.

## 5. Hypothesis versus executed discrimination

Every claim distinguishes evidence that **could have come out the other way**
from evidence that merely agrees.

A derivation carries `registered_before_evidence`. If false, and it predicts a
value already in hand, its role is `postdiction` and it cannot raise a claim
above `plausible`. This rule exists because this project re-derived the same
empirical answer three times after each prior derivation failed.

## 6. Well-formedness of numerical claims

An exponent or amplitude claim is invalid without:

- `parameterization` (λ_c and r_c give different fitted exponents over any finite
  window purely from the Jacobian d ln r/d ln λ = 1/(1−λ))
- `observable_id` (fitted exponents in this project span 0.36–0.57 across
  locators)
- `fitting_window` and `window_sensitivity` across at least three windows
  (fitted exponents here span 0.19–1.02 across ζ windows)

A single-window exponent is not a measurement.

## 7. Disputes are preserved, not resolved by authority

Competing claims are linked symmetrically by `contests` and share a
`dispute_id`. Neither is `superseded_by` the other. Disputes are closed by
evidence or they stay open. Closing one by recency, elegance, or seniority is
prohibited.

## 8. Dependency and staleness

Claims declare `depends_on`. When a claim becomes `refuted` or `superseded`,
every transitive dependent is flagged `stale` and must be re-reviewed. This rule
exists because a conclusion here rested on Δ_ζ = 1 for ten weeks after Δ ≈ 2 was
established.

## 9. Negative results are canonical

A killed line of investigation produces a `negative_result` claim and a decision
record. Kills are first-class state, not session-log footnotes.

## 10. Compute gates

| tier | scope | gate |
|---|---|---|
| T0 | read-only analysis of existing data | none, encouraged |
| T1 | local pilot < 30 min | declare budget, register the run |
| T2 | local campaign, hours | lead approval plus a registered spec |
| T3 | any HPC job | **human approval, immutable EXP-ID, pilot required** |
| T4 | > 10k core-hours or > 1 week wall | human approval plus written cost-benefit against the dispute it resolves |

A run without a valid `authorised_by: EXP-ID` cannot be registered as evidence.
Before proposing new compute, check whether existing data already answers the
question.

## 11. Non-negotiable behaviours

1. Never write `research/state/**` directly.
2. Never suppress a contradiction. Evidence against the claim you were asked to
   support must be registered in `contradicts`.
3. Never change a scientific definition silently. New definition, new `OBS-ID`.
4. Never assume a path exists because a document mentions it. Check it.
5. Never let a finding end the session unregistered. If there is no time for a
   full proposal, write a `chat_only_evidence` stub.
6. "No result" is always a valid outcome. No agent is rewarded for producing a
   finding.

## 12. Honest limits of this charter

It cannot tell you whether the physics is right. It ensures only that what is
claimed matches what was executed. It adds real friction, and if bypassed under
deadline it regenerates the situation the audit found. The mandatory surface is
therefore deliberately small: read two files, cite IDs, emit a proposal.
