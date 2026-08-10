---
lifecycle: active
authoritative_for: layout of the knowledge plane and migration status
last_reviewed: 2026-08-10
---

# research/ — the ppsQJ_m2 knowledge plane

Created 2026-08-10 by the Phase 4 migration. Design basis: `audit/2026-08-10/`.

**Read `RESEARCH_CHARTER.md` before doing substantive work.** Then `HANDOFF.md`.
Then only the specific entity files your task touches.

## Layout

```
RESEARCH_CHARTER.md          epistemic and procedural rules. Human-owned.
HANDOFF.md                   navigation and in-flight work. NO scientific values.
README.md                    this file
CHARTER_COMPLIANCE.md        which charter guarantees are mechanically enforced
METADATA_RECOVERY_PLAN.md    how to recover simulation metadata
GIT_MIGRATION_PLAN.md        staging plan for the migration commit
COWORK_AGENT_SPEC.md         design intent for the multi-agent system.
                             v1 is IMPLEMENTED as the `/research` workflow;
                             see `.claude/skills/research/SKILL.md`.

state/                       CANONICAL. Written only by the merge step.
    claims/       28 entities   scientific claims, status, evidence links
    evidence/     16 entities   datasets, analyses, derivations, code, artifacts
    observables/  10 entities   observable and estimator definitions
    disputes/      7 entities   preserved unresolved disagreements
    decisions/     4 entities   methodology decisions and kill records
    sources/      11 entities   literature, with inspection_level

RESOURCE_POLICY.md           compute, model and token policy. Human-owned.
resource_profile.local.yaml  MACHINE-LOCAL compute profile, gitignored.
experiments/                 approved experiment specs. These authorise
                             PREPARATION of an HPC package, never submission.
tasks/{active,completed,killed}/
    TASK_TEMPLATE/           skeleton for a `/research` task directory
proposals/                   agent output awaiting review
runs/                        executed compute
schemas/CLAIM_SCHEMA.yaml    the four claim axes, canonical
templates/REDTEAM_TEMPLATE.yaml   nine Stage 8 attacks, PER CANDIDATE (v2)
templates/TASK_MANIFEST.yaml      phase ledger with frozen artifact hashes
templates/ANALYSIS_SPEC.yaml      estimator + crossing-validity declaration
tools/validate_state.py      integrity checker
tools/validate_redteam.py    Stage 8 completeness checker (schema v2 + legacy v1)
tools/validate_task.py       execution-plane task completeness + PHASE LOCK
tools/validate_resource_policy.py  checks the resource policy is wired in
tools/task_phase.py          open/close/freeze task phases (TASK_MANIFEST.yaml)
tools/find_predecessors.py   duplicate / novelty gate over canonical records
tools/resolve_provenance.py  one-hop direct provenance for a claim
tools/test_workflow_regressions.py  the seven reviewed defects, as tests
data_roots.local.yaml        MACHINE-LOCAL, gitignored. See state/DATA_ROOTS.yaml.
history/legacy/              frozen pre-migration material
history/sessions/            per-session records
```

## Conventions

- **One file per entity**, named for its ID. This keeps git diffs readable and
  lets two agents propose in parallel without a merge conflict on a shared file.
- **IDs are immutable.** Correcting a claim means a new claim plus `supersedes`.
- **No scientific number appears in more than one place.** Prose cites IDs.
- Two axes, never collapsed: epistemic `status` on claims, `reproducibility` on
  evidence.

## Validate

```bash
.venv/bin/python3 research/tools/validate_state.py
```

Checks duplicate IDs, id/filename mismatch, dangling references, the
reproducibility floor for `verified`, missing discriminating evidence, the
dependency staleness cascade, superseded claims cited as live evidence, broken
paths asserted to exist, contested-claim bookkeeping, well-formedness of numeric
claims, and asymmetric `contests` links.

Current, 2026-08-10: **76 entities, 0 errors, 1 warning.** The warning is
`W3` on `CB-MIPT-001`, whose `observable_id` is *declared* unresolved between
`OBS-SHALF-FINAL-001` and `OBS-SHALF-TAVG-001`. A declared gap is a warning by
design; a silent omission would be error `E10`. Schemas: `research/schemas/`.
Charter compliance: `research/CHARTER_COMPLIANCE.md`.

**Do not trust the counts in this file over the validator.** They are a
convenience and they go stale on every merge. `validate_state.py` prints the
live breakdown; that is the authority.

## Critique of this structure, as implemented

Deviations from `audit/2026-08-10/ARCHITECTURE_PROPOSAL.md`, and why.

**Not created: `INDEX.md`, `ACTIVE_QUESTIONS.md` and their generator.** At this
size the directory listing is the index, and `HANDOFF.md` §4 lists the open
disputes by hand. Generating a file that duplicates a seven-row table is empty
complexity. Add the generator when the state exceeds roughly 150 entities or
when an agent must query it programmatically.

**Not created: `register_run.py`, `check_dependencies.py`, `generate_index.py`.**
The cascade check is implemented inside `validate_state.py` as check E6, which
is where it actually gets run. `register_run.py` is premature: there are no runs
yet, and its shape should be determined by the first real run, not guessed.

**`sources/` is populated but only partially inspected.** 11 sources are
registered. Two carry `inspection_level: relevant_sections` (`SRC-JIAN-2023`,
`SRC-KMR-2023`), one is `abstract_only` (`SRC-LMR-2025`), and eight are
`not_inspected`. Validator check `E20` blocks a `supported` claim from leaning
on anything below `relevant_sections`, so the registry is safe to cite from
today — it is just incomplete. **Stage 0 is therefore discharged per source, not
globally.** A `/research` task completes Stage 0 for its own load-bearing
sources; see `.claude/skills/research/SKILL.md` §Stage 0. It does not need the
whole registry inspected first.

**No `worktrees/`.** Git worktrees live wherever git puts them. Creating an
empty directory for them signals nothing.

**Kept despite being empty**: `experiments/`, `tasks/*`, `proposals/`, `runs/`,
`history/sessions/`. These are addressed by the charter and the agent spec, so
their absence would be a broken reference rather than avoided complexity.

## Bootstrap status and known gaps

The state is a **conservative first pass**, not complete.

Status names below are the charter vocabulary (`unsupported`, `provisional`,
`supported`, `contradicted`, `withdrawn`), which the schema and validator now
use. The pre-migration names are retained per claim in
`architecture_status_legacy`.

- Only **3 claims are `supported`**, and each is a statement about artifacts or
  mathematics that the audit checked directly, not a physics result:
  `CB-PARAM-001` (a Jacobian identity), `CASEA-IMPL-001` (Cut A data exists),
  `INFRA-GIT-001` (repository state). **No physics claim is supported.** That is
  deliberate.
- `CB-AMP-001` is `provisional` despite being independently reproduced, because
  it is not L-extrapolated and has no bootstrap. Its observable dependency was
  resolved by the 2026-08-10 audit: it now cites `OBS-BLPROD-001`
  (`definition_verified: true`), with `OBS-BL-001` retained as
  `observable_id_previous`.
- 7 claims are `contested: true` across 6 live disputes. Contestation is an
  orthogonal boolean, not a status; both sides keep their own
  `epistemic_status`.
- Several claims carry `evidence: []` with an `evidence_note` recording that the
  supporting analysis was not located. That is honest, not a defect.
- **The observable audit is done for the load-bearing locators.** `OBS-BL-001`
  is `retired` and superseded by `OBS-BLPROD-001` (our average-of-products) and
  `OBS-BLKMR-001` (KMR's product-of-averages) after the audit found one label
  covering two distinct quantities. `OBS-CMI-001` is `active`,
  `definition_verified: true`. **`OBS-ACTIVITY-001` remains `needs_audit` with
  `definition_verified: false`** — activity and SCGF-adjacent claims still
  cannot be promoted.

## Blockers before the first multi-agent experiment

Derived under the restored charter. See `CHARTER_COMPLIANCE.md` for the
reasoning. Reviewed 2026-08-10 at the `/research` v1 build.

**Knowledge-layer — CLEARED. A read-only research task may now open.**
1. ~~Audit `OBS-BL-001` and `OBS-CMI-001`.~~ **Done.** The audit
   (`proposals/2026-08-10-A-observable-audit.md`) split `OBS-BL-001` into
   `OBS-BLPROD-001` and `OBS-BLKMR-001` and retired it. `OBS-CMI-001` is
   verified. Residual: `OBS-ACTIVITY-001` is still `needs_audit`, which blocks
   activity/SCGF claims only.
2. ~~Add the nine mandated red-team attacks with a validator check.~~ **Done.**
   `templates/REDTEAM_TEMPLATE.yaml` plus `tools/validate_redteam.py`; a missing
   attack is error `R4` and fails the run.

**Reproducibility (block trusting an analysis, not reading state):**
3. ~~Run the T/L check.~~ **Executed** (read-only, T0). Result in
   `proposals/2026-08-10-C-metadata-and-TL.md`; **not yet merged into state**, so
   cite it as provenance, not support, until it passes the human gate.
4. **Partially open.** `SRC-JIAN-2023` and `SRC-KMR-2023` are inspected at
   `relevant_sections`. `SRC-LMR-2025` is `abstract_only` and eight sources are
   `not_inspected`. This is no longer a global blocker: a `/research` task
   discharges Stage 0 for its own load-bearing sources and returns
   `Infrastructure first` if it cannot.

**Execution (block code changes and compute, not reading):**
5. `GIT_MIGRATION_PLAN.md` — the migration commit landed (`6c9c843`) and the
   tracked tree is clean. Remaining: unregistered outputs under `analysis/`.
6. `register_run.py` — **still not written.** Blocks converting a run into
   evidence, not reading.

**HPC (permanently human-submit only):**
7. An approved `experiments/<EXP-ID>.yaml` authorises **preparation** of an HPC
   package, never its submission. **No agent ever submits an HPC job**, at any
   stage or gate. `/research` stops at Human Gate A by construction and never
   launches compute. Policy: `research/RESOURCE_POLICY.md` §4.

## The `/research` engine

Version 1, built 2026-08-10. Version-controlled under `.claude/`:

```
.claude/skills/research/SKILL.md            the charter, operationalised.
                                            The LEAD's procedure.
.claude/skills/research/WORKER_CONTRACT.md  the compact worker-facing
                                            invariants. Workers read this.
.claude/agents/{literature,theory,numerics,red-team}.md
.claude/workflows/research.js      the phase script (A-F), stops at Gate A
.claude/commands/research.md       the /research slash command
.claude/hooks/guard_research.py    PreToolUse guard, see CHARTER_COMPLIANCE
.claude/settings.json              shared deny/ask rules. Tracked.
```

The main session is the research lead; there is no lead subagent. There is no
implementation agent in v1 — experiment execution is a later phase. `/research`
is read-only with respect to `research/state/**`, and that is enforced by a hook
rather than by instructions.

Resource discipline is `research/RESOURCE_POLICY.md`: local-only autonomous
compute, no simulation during `/research`, human-approved local pilots only,
**agents never submit HPC jobs**, and explicit per-role model routing so nothing
silently inherits the strongest model.

### Execution lifecycle

```
/research
   └─ autonomous read-only investigation (T0, local, no simulation)
        └─ HUMAN GATE A
             ├─ local compute proposed:
             │     human approves  →  bounded local Mac pilot (<=10 min,
             │     one job, pinned threads)  →  review  →  next human decision
             │
             └─ HPC necessary:
                   AI prepares and reviews the package
                      └─ READY_FOR_HUMAN_SUBMISSION
                           └─ HUMAN SUBMITS MANUALLY
                                └─ returned data ingested and reviewed later
```

**No branch of this diagram has an agent submitting a job.** Designing
experiments, writing and validating SLURM scripts, building parameter manifests
and estimating CPU/RAM/wall-time are all expected agent work; executing them is
not. Enforced by `guard_research.py` G4 (which matches a scheduler in *command
position*, so preparation stays possible) and by deny rules in
`.claude/settings.json`.

## What is NOT authoritative

`theory/**`, `audit/**`, `history/**`, project memory, project instructions,
manuscripts, and prior conversations. Read for orientation, cite as provenance,
never as support.
