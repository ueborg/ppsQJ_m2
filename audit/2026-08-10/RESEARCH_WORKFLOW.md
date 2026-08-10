# Research workflow

Audit 2026-08-10, Stage 3. Non-canonical. The executable form of the charter.

## 0. Diagram with artifacts

```text
RESEARCH QUESTION                      human, or ACTIVE_QUESTIONS.md
        |                              -> tasks/active/<TASK-ID>/CHARTER.md
        v
PROBLEM RECONSTRUCTION                 lead
   what is already claimed?            -> RECONSTRUCTION.md
   what evidence exists?               (claim IDs, evidence IDs, known negatives)
   what was already killed?            MUST cite state/decisions + tasks/killed
        |
        v
PARALLEL INVESTIGATION                 lit / theory / numerics / red-team
   independent, no cross-reading       -> proposals/<TASK-ID>-<role>/FINDINGS.md
        |                                 + structured proposals
        v
CANDIDATE COMPARISON                   lead
        |                              -> LEAD_SUMMARY.md
        v                                 (dispute proposal if unresolved)
+----- FALSIFICATION GATE -----+        red-team
|                              |        -> REDTEAM.md {survives|scoped|killed}
killed                    survives
  |                            |
  v                            v
tasks/killed/<ID>/       MINIMAL EXPERIMENT SPEC     exp-design
+ negative_result              |                     -> EXPERIMENT_SPEC.yaml
  claim proposal               v
  -> still goes to       HUMAN EXPERIMENT GATE       human only
     the human gate            |                     -> experiments/<EXP-ID>.yaml
                               v                        (immutable once approved)
                         IMPLEMENTATION               impl agent, worktree
                               |                     -> branch + runs/<RUN-ID>/
                               v
                         EXECUTED EVIDENCE           register_run.py
                               |                     -> evidence proposal
                               v                        + checksums
                         RED-TEAM REVIEW #2          red-team
                               |                     -> REDTEAM_POST.md
                               v
                         HUMAN VERIFICATION GATE     human only
                               |                     -> APPROVAL record
                               v
                         CANONICAL STATE UPDATE      merge step
                                                     -> state/**, INDEX.md,
                                                        ACTIVE_QUESTIONS.md,
                                                        history/sessions/
```

Note the two red-team passes and the two human gates. The first pair guards
**whether to spend compute**. The second pair guards **whether a result becomes
truth**. Collapsing them is how expensive campaigns get launched on hypotheses
that a cheap check would have killed.

## 1. Problem reconstruction (mandatory, cheap, skipped at your peril)

Before any new work, the lead produces `RECONSTRUCTION.md` answering:

- Which claim IDs does this question touch, and what is their status?
- What evidence already exists, and is any of it `chat_only` or unanalysed?
- **What has already been killed here?** Mandatory citation of
  `state/decisions/` and `tasks/killed/`.
- Is this answerable from existing data without new compute?

The audit's motivating case: the July 2026 campaign of 5,634 realizations was
already on disk, unanalysed, while the open questions it addressed were being
discussed elsewhere. Step 1 is designed to catch exactly that.

## 2. Parallel investigation

Specialists work **independently and without reading each other's drafts**. The
lead does not tell them the preferred answer. Each returns findings plus
proposals, or an explicit null.

## 3. Falsification gate

Red-team receives the candidate and the raw evidence, not the lead's narrative,
and applies the ten-point checklist in `AGENT_ROLES.md`.

Outcomes:
- `killed` -> task closes, a `negative_result` claim proposal is created and
  **still goes through the human gate**. Negative results are canonical state.
  The audit found the project repeatedly re-deriving things it had already
  disproven because kills were buried in a session log.
- `survives_with_scope_restriction` -> the claim's `scope` and `validity_range`
  are narrowed before proceeding.
- `survives` -> proceed to experiment design.

## 4. Minimal experiment specification

The spec must name the **smallest** test that discriminates. Not the most
thorough, not the most publishable. Required completion of this sentence:

> "If the observable takes value X in range R, hypothesis A is excluded at
> confidence Y, and if it takes value X', hypothesis B is excluded."

A spec that cannot complete it is not an experiment, it is a data-collection
wish. Rejected before the human sees it.

## 5. HPC / expensive-computation gate

**No agent may launch a Ruche campaign.** Full stop. The gate is a human
approval writing `experiments/<EXP-ID>.yaml`, which is immutable thereafter.

### Required fields

```yaml
id: EXP-...
question: >                 # which dispute or open claim this resolves
hypotheses:
  - {id: CLAIM-ID, prediction: }
  - {id: CLAIM-ID, prediction: }   # >=2. A one-hypothesis experiment is rejected.
observable_id: OBS-...
estimator:
parameters:
  L: []
  zeta: []
  lambda: {}
  T_rule: >                 # REQUIRED, and must state T/L explicitly
  N_c: []
  n_real:                   # REQUIRED with a justification
  burn_in:
  seeds: {base:, disjoint_from: []}
required_statistics: >      # target uncertainty on the DECIDING quantity,
                            # not on an intermediate
discriminating_outcome: >   # the "if X then A excluded" sentence
kill_criterion: >           # what result ends this line of investigation
metric_id:                  # must be in DEC-MASTER-METRIC for production claims
cost:
  core_hours:
  wall_estimate:
  partition:
  basis:                    # which cost-model evidence ID this is derived from
pilot:
  required: true            # default true
  spec: >                   # smallest cell, expected runtime, what it validates
  pilot_pass_criterion: >
output:
  path:
  required_metadata: [L, zeta, lambda, T, T_over_L, N_c, n_real, burn_in,
                      seeds, git_commit, environment, authorised_by]
authorised_by: {human:, date:}
```

### Tiered gates

| tier | example | gate |
|---|---|---|
| **T0** read-only analysis of existing data | re-fitting crossings, reading an aggregate | none. Encouraged. |
| **T1** local pilot, < 30 min, single machine | one cell to validate a spec | agent may proceed, must declare budget and register the run |
| **T2** local campaign, hours | an L-scan on the Mac | lead approval + a registered spec |
| **T3** any HPC job | anything on Ruche | **human approval, immutable EXP-ID, pilot required** |
| **T4** > 10k core-hours or > 1 week wall | an L=256 ν tier | human approval **plus** a written cost-benefit against the dispute it resolves |

`register_run.py` refuses to create evidence for a T3/T4 run lacking a valid
`authorised_by`. That is the enforcement point, not the submit script.

## 6. Human verification gate

Two gates, both human-only.

**Gate A, before compute.** Approves an `EXPERIMENT_SPEC`. The question is "is
this the cheapest thing that could change my mind".

**Gate B, before canonical state.** Approves claim and evidence proposals. The
question is "does the evidence actually support the statement, at the stated
scope".

The human may approve, reject with reason, or approve with a narrowed scope.
Rejections are retained with reasons, never deleted.

### What may bypass Gate B

Nothing that changes a claim's status. Three low-risk classes may be merged by
the lead and reported rather than pre-approved:

- registering new evidence with `supports: []` (recording that data exists)
- correcting a path, checksum or metadata gap
- generated files (`INDEX.md`, `ACTIVE_QUESTIONS.md`)

Everything else waits.

## 7. Merge step

Mechanical, and the only writer of `state/`:

1. `validate_state.py` on the proposed state. Hard fail blocks the merge.
2. `check_dependencies.py` cascades `stale` from any status change.
3. Regenerate `INDEX.md` and `ACTIVE_QUESTIONS.md`.
4. Write `history/sessions/<date>-<TASK-ID>.md`.
5. Single git commit referencing the task ID and the approval record.

## 8. Session protocol for ordinary chats

Not every conversation is a research task. Lightweight path:

- Read `RESEARCH_CHARTER.md`, then `HANDOFF.md`. Load claim IDs on demand.
- Do the work in the execution plane.
- **If a finding emerges, emit a proposal before the session ends.** A finding
  that exists only in the conversation is the failure this whole architecture
  targets. If there is no time to write a full proposal, write a stub in
  `proposals/` with the conversation reference and
  `type: chat_only_evidence`, `recovery_priority`.
- Do not edit `state/`. Do not edit `HANDOFF.md` except as the lead, and never
  with a scientific number.

## 9. Worked example: the φ dispute

Concretely, what the system does with the audit's hardest open question.

1. `DISP-PHI-001` holds `CB-PHI-HALF-001` (φ ≈ 1/2, empirical, window-dependent)
   and `CB-PHI-LINEAR-001` (φ = 1, from x_J ≈ 1.04 and from corrected-input
   matching). Both `status: contested`. Neither supersedes the other.
   `ACTIVE_QUESTIONS.md` surfaces it automatically.
2. Reconstruction notes that the x_J scripts were container-side and are lost,
   that the source memos are unrecovered, and that the 2026-08-05 χ₂ scan gave
   p = 0.46–1.40 which neither confirms nor refutes.
3. Theory agent is asked for a prediction that separates the two at accessible
   L. Numerics agent is asked whether existing data already separates them,
   including a window scan and the L-extrapolation that has never been done.
   Literature agent searches for the x_J formulation under other names.
4. Red-team attacks both sides symmetrically.
5. Only if existing data cannot separate them does an experiment spec get
   written, and the pilot requirement forces a small L-extrapolation test before
   any L=192 or L=256 campaign.

The point of the example: under the current architecture this dispute is
invisible, because one side lives in an expired container and the other is the
project headline.
