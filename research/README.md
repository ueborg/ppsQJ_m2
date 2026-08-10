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
METADATA_RECOVERY_PLAN.md    how to recover simulation metadata
GIT_MIGRATION_PLAN.md        staging plan for the migration commit
COWORK_AGENT_SPEC.md         design of the future multi-agent system

state/                       CANONICAL. Written only by the merge step.
    claims/       27 entities   scientific claims, status, evidence links
    evidence/     15 entities   datasets, analyses, derivations, code, artifacts
    observables/   6 entities   observable and estimator definitions
    disputes/      7 entities   preserved unresolved disagreements
    decisions/     4 entities   methodology decisions and kill records
    sources/      11 entities   literature, with inspection_level

experiments/                 approved experiment specs. HPC authorisation tokens.
tasks/{active,completed,killed}/
proposals/                   agent output awaiting review
runs/                        executed compute
tools/validate_state.py      integrity checker
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

Current: **70 entities, 0 errors, 0 warnings.** Schemas: `research/schemas/`.
Charter compliance: `research/CHARTER_COMPLIANCE.md`.

## Critique of this structure, as implemented

Deviations from `audit/2026-08-10/ARCHITECTURE_PROPOSAL.md`, and why.

**Not created: `INDEX.md`, `ACTIVE_QUESTIONS.md` and their generator.** With 59
entities the directory listing is the index, and `HANDOFF.md` §4 lists the open
disputes by hand. Generating a file that duplicates a seven-row table is empty
complexity. Add the generator when the state exceeds roughly 150 entities or
when an agent must query it programmatically.

**Not created: `register_run.py`, `check_dependencies.py`, `generate_index.py`.**
The cascade check is implemented inside `validate_state.py` as check E6, which
is where it actually gets run. `register_run.py` is premature: there are no runs
yet, and its shape should be determined by the first real run, not guessed.

**`sources/` is empty and that is a real gap**, not a simplification. Populating
it needs actual source inspection, which is deferred literature work. Until it
is populated, no claim can cite literature as evidence, which is arguably the
correct default given `DEC-CITATION-001`.

**No `worktrees/`.** Git worktrees live wherever git puts them. Creating an
empty directory for them signals nothing.

**Kept despite being empty**: `experiments/`, `tasks/*`, `proposals/`, `runs/`,
`history/sessions/`. These are addressed by the charter and the agent spec, so
their absence would be a broken reference rather than avoided complexity.

## Bootstrap status and known gaps

The state is a **conservative first pass**, not complete.

- Only **3 claims are `verified`**, and each is a statement about artifacts or
  mathematics that the audit checked directly, not a physics result:
  `CB-PARAM-001` (a Jacobian identity), `CASEA-IMPL-001` (Cut A data exists),
  `INFRA-GIT-001` (repository state). **No physics claim is verified.** That is
  deliberate.
- `CB-AMP-001` is `plausible` despite being independently reproduced, because it
  is not L-extrapolated, has no bootstrap, and rests on `OBS-BL-001` whose stored
  definition is unaudited.
- Several claims carry `evidence: []` with an `evidence_note` recording that the
  supporting analysis was not located. That is honest, not a defect.
- `OBS-BL-001`, `OBS-CMI-001` and `OBS-ACTIVITY-001` are `needs_audit` or have
  `definition_verified: false`. **Until `OBS-BL-001` is audited against the
  worker source, no exponent claim can be promoted.**

## Blockers before the first multi-agent experiment

Derived under the restored charter. See the session report and
`CHARTER_COMPLIANCE.md` for the reasoning.

**Knowledge-layer (block a read-only experiment):**
1. Audit `OBS-BL-001` and `OBS-CMI-001` against `pps_qj/parallel/worker_clone_pps.py`.
   Until then every exponent claim points at an unverified definition.
2. Add the nine mandated red-team attacks (charter Stage 8) to the report
   template, with a validator check that each has a verdict.

**Reproducibility (block trusting an analysis, not reading state):**
3. Run the T/L check in `METADATA_RECOVERY_PLAN.md` §5 step 1.
4. Inspect the load-bearing sources, above all `SRC-JIAN-2023`.

**Execution (block code changes and compute, not reading):**
5. Execute `GIT_MIGRATION_PLAN.md`.
6. Write `register_run.py`.

**HPC (block production jobs only):**
7. An approved `experiments/<EXP-ID>.yaml`.

## What is NOT authoritative

`theory/**`, `audit/**`, `history/**`, project memory, project instructions,
manuscripts, and prior conversations. Read for orientation, cite as provenance,
never as support.
