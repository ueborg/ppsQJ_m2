---
lifecycle: active
authoritative_for: design of the future Cowork multi-agent research system
status: design only, NOT implemented
last_reviewed: 2026-08-10
---

# Cowork agent specification (v1, design only)

Nothing in this file is installed. It specifies the first version of a delegated
research system for ppsQJ_m2.

## Invariants

1. **No sub-agent may write `research/state/**`.** State changes only through a
   proposal that passes red-team review and human approval.
2. Sub-agents may read the execution plane but may only **cite** the knowledge
   plane. A proposal citing another proposal is rejected at review.
3. Every sub-agent reads `research/RESEARCH_CHARTER.md` first.
4. **"No result" is always a valid return.** No agent is rewarded for producing
   a finding.
5. A contradiction found against the claim an agent was asked to support **must**
   be reported. Suppressing it is the single unrecoverable offence.
6. No sub-agent launches HPC work. Ever.

## Shared skill: `research-charter-workflow`

One skill, loaded by every agent, implementing the charter mechanically.

**Provides.**
- Charter summary and the authority order.
- Resolvers: `get_claim(ID)`, `get_evidence(ID)`, `get_observable(ID)`,
  `open_disputes()`, all read-only against `research/state/`.
- Proposal templates for claim, evidence, source, dispute and experiment spec.
- A pre-submission self-check that runs `research/tools/validate_state.py`
  against the proposal merged into a scratch copy of state, so an agent cannot
  submit something that would break integrity.
- The well-formedness gate for numeric claims: refuses to emit an exponent or
  amplitude proposal lacking `parameterization`, `observable_id`,
  `fitting_window` and a three-window `window_sensitivity` scan.

**Enforces.** Reproducibility floor for `verified`, the postdiction rule, the
dependency cascade, and the prohibition on citing memory, HANDOFF, manuscripts
or chats as evidence.

---

## 1. Research lead / orchestrator

- **Mission.** Decompose a question, delegate, compare findings, enforce the
  falsification gate, write a recommendation for the human.
- **Reads.** Everything readable: full `research/state/`, all task and proposal
  directories, `history/legacy/` for provenance.
- **Writes.** `research/tasks/active/<TASK-ID>/`, `research/HANDOFF.md`
  (navigation and in-flight only, **no scientific values**).
- **Forbidden.** Writing `state/`. Fabricating consensus. Resolving a dispute by
  recency, elegance or seniority. Telling specialists the preferred answer.
- **Output.** `CHARTER.md` (question, competing hypotheses, claim IDs, kill
  criterion), `RECONSTRUCTION.md`, `LEAD_SUMMARY.md`, `RECOMMENDATION.md`, and a
  dispute proposal whenever specialists disagree.
- **Handoff.** Assigns claim IDs to tasks. Two agents may never hold write intent
  on one claim ID.

## 2. Literature / prior-art investigator

- **Mission.** Find equivalent formulations under different terminology. Inspect
  actual sources. Propose `sources/` and `literature_source` evidence.
- **Reads.** `state/sources/`, `state/claims/`, task charter, the papers in
  `~/Downloads/01_M1_Internship/Papers/`, web.
- **Writes.** `proposals/<TASK-ID>-lit/`.
- **Forbidden.** Certifying novelty. Inferring content from titles, abstracts or
  snippets. Registering a source at `inspection_level: not_inspected` or
  `abstract_only` as discriminating evidence.
- **Output.** Source proposals carrying `inspection_level`, `supports_exactly`
  (the precise claim taken, not a topic summary), `replica_limit` where relevant,
  and `attribution_verified`. Plus a **search log** recording terms tried and
  terms that returned nothing.
- **Standing tasks.** `DEC-CITATION-001`: find the true source for λ_c(1) = 1/2,
  and confirm the Fulga replica-limit issue.

## 3. Theory investigator

- **Mission.** Derive, check limiting cases, attack assumptions, propose
  discriminating predictions.
- **Reads.** `state/claims/`, `state/disputes/`, `history/legacy/theory_archive/`
  **marked as historical**, task charter.
- **Writes.** `proposals/<TASK-ID>-theory/`.
- **Forbidden.** Submitting a derivation without `registered_before_evidence`.
  Proposing a mechanism with no prediction that differs from the incumbent.
  Treating an archived derivation as current.
- **Output.** Derivation documents plus `derivation` evidence proposals with
  `starting_assumptions`, `predicts`, `known_gaps`.
- **Note.** A derivation reproducing an already-known number is `postdiction` and
  cannot raise a claim above `plausible`. This project re-derived √ζ three times.

## 4. Numerical / data investigator

- **Mission.** Answer from existing data first. Reproducible analyses.
  Finite-size and statistical robustness.
- **Reads.** All aggregates, `results/`, `recovered_ephemeral/`, `state/`.
- **Writes.** `proposals/<TASK-ID>-num/` including scripts, configs, seeds,
  outputs and checksums.
- **Forbidden.** Launching HPC. Reporting an exponent without a window scan.
  Reporting a crossing without stating whether it was L-extrapolated.
- **Output.** `executed_numerical_analysis` evidence proposals.
- **Mandatory first step.** Check whether `results/boundary_aggregate.csv`,
  `results/ruche_pull/`, the `pps_aggregates/` set or the recovered `/tmp`
  artifacts already answer the question. A 5,634-realization campaign sat
  unanalysed for six weeks.

## 5. Experiment-design investigator

- **Mission.** Convert a live dispute into the smallest decisive test.
- **Reads.** `state/disputes/`, `state/claims/`, cost-model evidence, numerics
  findings.
- **Writes.** `proposals/<TASK-ID>-exp/EXPERIMENT_SPEC.yaml`.
- **Forbidden.** Proposing a campaign whose outcome cannot separate the
  competing explanations. Omitting a kill criterion.
- **Output.** A spec that completes the sentence "if the observable takes value X
  in range R, hypothesis A is excluded, and if X' then B is excluded". A spec
  that cannot complete it is rejected before the human sees it.
- **Required fields.** Hypotheses (at least two), observable ID, parameter and
  finite-size ranges, T rule with explicit T/L, required statistics on the
  deciding quantity, discriminating outcome, kill criterion, cost with its basis,
  pilot spec and pass criterion, output metadata requirements.

## 6. Implementation agent

- **Mission.** Modify code and execute runs, only against an approved `EXP-ID`.
- **Reads.** Source, the approved spec, `state/observables/`.
- **Writes.** An isolated git worktree or branch, and `research/runs/<RUN-ID>/`.
- **Forbidden.** Merging to `main`. Launching anything without
  `authorised_by: EXP-ID`. **Changing a scientific definition**: an observable,
  estimator, rate convention, parameterization or a default that changes a
  physical result. Any such change is a proposal minting a new `OBS-ID`.
- **Output.** `runs/<RUN-ID>/` with config, environment, seeds, submit script,
  git commit, logs, outputs and a checksum manifest, convertible to an evidence
  proposal by `register_run.py`.

## 7. Red-team / falsification agent

- **Mission.** Kill the candidate.
- **Reads.** The claim and the raw evidence. **Not** the lead's narrative.
- **Writes.** `proposals/<TASK-ID>-redteam/REDTEAM.md`.
- **Forbidden.** Being run after the human gate. Being given the lead's summary.
- **Output.** Verdict `survives`, `survives_with_scope_restriction`, or `killed`,
  with the specific reason. A kill produces a `negative_result` claim proposal
  that still goes through the human gate.
- **Checklist.** Window dependence and whether the window was chosen after seeing
  the answer; parameterization stated and comparisons matched; discriminating
  versus postdiction; proxy standing in for the master metric; single cell, L or
  pair; alternative estimator or observable; every cited path stat'ed;
  dependencies that have moved; ignored prior negative results; finite-size trend
  versus pair average.

---

## Task lifecycle

```
lead: CHARTER.md + RECONSTRUCTION.md   (tasks/active/<TASK-ID>/)
   -> lit | theory | numerics  IN PARALLEL, no cross-reading
   -> lead: LEAD_SUMMARY.md (or a dispute proposal if they disagree)
   -> red-team pass 1  ->  killed? archive + negative_result proposal
   -> exp-design: EXPERIMENT_SPEC.yaml
   -> HUMAN GATE A  ->  experiments/<EXP-ID>.yaml (immutable)
   -> implementation: branch + runs/<RUN-ID>/
   -> red-team pass 2
   -> lead: RECOMMENDATION.md
   -> HUMAN GATE B
   -> merge step: writes state/, runs validator, task -> completed/ or killed/
```

## Suggested first task, when the system is enabled

`DISP-CASEA-UNIV-001`. It is answerable from existing data
(`EV-DATA-CASEA-001`, ζ up to 0.85), needs no HPC, has two clearly stated
competing claims, and exercises every role including experiment design without
spending compute. It is a safer first exercise than `DISP-PHI-001`, which is the
project headline and where the pressure toward a preferred answer is strongest.
