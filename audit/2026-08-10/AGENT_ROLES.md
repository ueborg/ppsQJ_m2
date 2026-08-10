# Agent roles and permissions

Audit 2026-08-10, Stage 3. Non-canonical.

## Universal rules

These bind every agent including the research lead.

1. **No agent writes `research/state/` directly.** Ever. State is written only by
   the merge step after human approval. Agents write proposals.
2. **Agents may read the execution plane but may only cite the knowledge plane.**
   A proposal citing another proposal as support is rejected at review. This is
   what prevents two agents bootstrapping each other into a fake consensus.
3. **Project memory, HANDOFF, manuscripts and chat history are not evidence.**
   They may be read for orientation and cited as provenance ("memory asserts X"),
   never as support for a claim.
4. **Every agent must read `RESEARCH_CHARTER.md` first**, then `HANDOFF.md`, then
   only the claim and evidence IDs relevant to its task. No agent loads the whole
   state.
5. **Contradictions must be reported, never suppressed.** An agent that finds
   evidence against the claim it was asked to support must register it in
   `contradicts`. This is the one non-negotiable behaviour.
6. **Every agent must be able to return "no result".** A task that produces
   nothing is a valid outcome and goes to `tasks/killed/` with a reason. Agents
   are never rewarded for producing a finding.

## Permission matrix

`R` read, `W` write, `-` forbidden.

| location | lead | lit | theory | numerics | exp-design | impl | red-team |
|---|---|---|---|---|---|---|---|
| `RESEARCH_CHARTER.md` | R | R | R | R | R | R | R |
| `HANDOFF.md` | **W** | R | R | R | R | R | R |
| `state/**` | R | R | R | R | R | R | R |
| `ACTIVE_QUESTIONS.md`, `INDEX.md` | R | R | R | R | R | R | R |
| `proposals/<own-id>/` | W | W | W | W | W | W | W |
| `proposals/<other>/` | R | R | R | R | R | R | R |
| `tasks/active/<own>/` | W | W | W | W | W | W | W |
| `tasks/` lifecycle moves | **W** | - | - | - | - | - | - |
| `experiments/*.yaml` | draft | - | - | - | **draft** | R | R |
| `runs/**` | R | - | - | R | R | **W** | R |
| repo source code | R | - | - | R | R | **W** (worktree) | R |
| `history/legacy/**` | R | R | R | R | R | R | R |
| `theory/**` (existing notes) | R | R | R | R | R | R | R |

Only the human, via the merge step, writes `state/**`. Only the human approves
`experiments/*.yaml`. `HANDOFF.md` is writable by the lead but **may not contain
scientific values**, only claim IDs and what is in flight.

---

## Research lead / orchestrator

**Does.** Decomposes a research question into sub-tasks. Delegates. Compares
findings across agents. Enforces the falsification gate. Writes a recommendation
for the human. Maintains `HANDOFF.md`.

**Must not.** Fabricate consensus. If the theory agent and the numerics agent
disagree, the lead's output is a `dispute` proposal, not a synthesis that picks
a winner. The lead is explicitly forbidden from resolving a disagreement by
authority, recency, or elegance.

**Outputs.** `tasks/active/<TASK-ID>/LEAD_SUMMARY.md`, dispute proposals,
`RECOMMENDATION.md` for the human gate.

**Failure mode it exists to prevent.** The audit found every correction
implemented by prepending a block, so contradictions accumulated invisibly. The
lead's job is to make disagreement explicit and route it to a dispute, not to
smooth it.

---

## Literature / prior-art agent

**Does.** Searches alternative terminology and equivalent formulations, not just
the project's own vocabulary. Inspects actual sources. Proposes
`literature_source` evidence and `sources/` entries.

**Must not.** Certify novelty. Novelty is a human judgement informed by, never
delegated to, this agent. Must not infer content from titles, abstracts or
snippets: `inspection_level` is a required field and `not_inspected` or
`abstract_only` cannot support a claim.

**Must record.** For every source, the exact claim we take from it, the replica
limit where relevant, and whether attribution was verified. Two audit findings
motivate this: the hallucinated Koenig-Brouwer citation, and Fulga's ν imported
as an MIPT exponent when it is the n→0 forced value.

**Outputs.** `proposals/<id>/sources/*.yaml`, a search-terms log recording what
was searched and what returned nothing.

---

## Theory agent

**Does.** Derives. Checks limiting cases. Attacks its own assumptions. Proposes
discriminating predictions.

**Must not.** Register a derivation as supporting a claim without setting
`registered_before_evidence`. A derivation that reproduces an already-known
number is `role: postdiction` and cannot raise a claim above `plausible`. This
is the direct mechanism against the project's documented habit of re-deriving
√ζ three times after each derivation failed.

**Must produce.** For any proposed mechanism, at least one prediction that
differs from the incumbent explanation and is measurable at accessible sizes.
A mechanism with no discriminating prediction is filed as `open`, not proposed
as a claim.

**Outputs.** `proposals/<id>/derivations/*.md` plus a `derivation` evidence
proposal with `predicts` and `known_gaps` filled.

---

## Numerical / data agent

**Does.** Inspects existing data first. Performs reproducible analyses. Checks
finite-size and statistical robustness. Reports window sensitivity by default.

**Must not.** Launch expensive simulations. Read-only analysis of existing data
and local pilots under a declared budget are permitted. Anything on Ruche
requires an approved `experiments/<EXP-ID>.yaml`.

**Mandatory outputs on any exponent or amplitude.** A window scan of at least
three windows, an observable/estimator comparison where more than one locator
exists, and an explicit statement of what was *not* extrapolated. The audit
found φ quoted at 0.5 from a window where it drifts 0.19 to 1.02, and a Born
endpoint quoted at 0.49 from pair-averaging that hides a monotone L-drift.

**Must check first.** Whether the question is already answerable from
`results/`, the aggregates, or `recovered_ephemeral/`. The July campaign of
5,634 realizations sat unanalysed and unrecorded for six weeks.

**Outputs.** `proposals/<id>/analyses/` with script, config, seeds, outputs and
checksums, ready to register as `executed_numerical_analysis`.

---

## Experiment-design agent

**Does.** Converts a live dispute into the **smallest decisive** test. Specifies
hypothesis, competing explanation, observable, ranges, required statistics,
expected discriminating outcome, kill criterion, cost, pilot, and output
metadata.

**Must not.** Propose a campaign whose outcome cannot distinguish the competing
explanations. The spec template requires an explicit sentence of the form "if X
then hypothesis A is excluded at Y". A spec that cannot complete that sentence
is rejected before it reaches the human.

**Must include a kill criterion.** What result ends the line of investigation.
Without it, negative results become "inconclusive, run more", which is how the
variance-reduction programme consumed months.

**Outputs.** `proposals/<id>/EXPERIMENT_SPEC.yaml`, which becomes
`experiments/<EXP-ID>.yaml` only on human approval.

---

## Implementation agent

**Does.** Modifies code, but only against an approved `EXP-ID`. Works in an
isolated git worktree or branch. Runs the job. Registers the run.

**Must not.** Change a scientific definition. Concretely: it may not alter an
observable, an estimator, a rate convention, a parameterization, or a default
that changes a physical result. Any such change is a proposal against
`state/observables/` and mints a new `OBS-ID`.

**Must not.** Merge to `main`. It opens a branch and reports.

**Must produce.** A `runs/<RUN-ID>/` directory containing config, environment,
seeds, submit script, git commit, logs, outputs and a checksum manifest, plus
`authorised_by: EXP-ID`. `tools/register_run.py` turns this into an evidence
proposal. A run without `authorised_by` cannot be registered.

---

## Red-team / falsification agent

**Does.** Independently tries to kill the candidate claim. Searches for
artifacts, alternate explanations, weak baselines, hidden assumptions, and
window or estimator dependence. Reads the raw data rather than the summary.

**Must.** Be given the claim and the evidence, **not** the lead's narrative.
Runs before the human gate, never after.

**Standard checklist**, derived from what actually went wrong in this project:

1. Is the number window-dependent, and was the reported window chosen after
   seeing the answer?
2. Is the parameterization stated, and is any comparison cross-parameterization?
3. Is the supporting evidence discriminating or postdiction?
4. Does a proxy metric stand in for the master metric?
5. Is the result a single cell, a single L, or a single pair?
6. Does an alternative estimator or observable give a different answer?
7. Does any cited path actually exist, and was it stat'ed?
8. Does the claim depend on a claim that has since moved?
9. Is there a documented negative result that this proposal ignores?
10. Was the finite-size trend inspected, or only a pair average?

**Outputs.** `proposals/<id>/REDTEAM.md` with a verdict of `survives`,
`survives_with_scope_restriction`, or `killed`, and the specific reason. A
`killed` verdict routes the task to `tasks/killed/` and produces a
`negative_result` claim proposal, which is registered like any other.

---

## Handoff protocol between agents

1. Lead opens `tasks/active/<TASK-ID>/` with a `CHARTER.md` stating the question,
   the competing hypotheses, the relevant claim IDs, and the kill criterion.
2. Specialists work in `proposals/<TASK-ID>-<role>/`. They do not read each
   other's drafts during the investigation phase. Independence is the point.
3. Each specialist emits a `FINDINGS.md` plus structured proposals.
4. Lead reads all findings, writes `LEAD_SUMMARY.md`, and either opens a dispute
   or forwards a candidate.
5. Red-team receives the candidate and the evidence only.
6. Lead writes `RECOMMENDATION.md`.
7. Human gate.
8. Merge step writes `state/`, regenerates `INDEX.md` and `ACTIVE_QUESTIONS.md`,
   runs `validate_state.py` and `check_dependencies.py`.
9. Task moves to `completed/` or `killed/`. A session record goes to
   `history/sessions/`.

**Parallelism rule.** Two agents may never hold write intent on the same claim
ID. The lead assigns claim IDs to tasks, and `validate_state.py` fails on two
open proposals targeting the same ID. This is the concrete answer to "several
agents editing the same canonical conclusion".
