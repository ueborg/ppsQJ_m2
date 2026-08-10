# ppsQJ_m2 — routing rules

Monitored free-fermion MIPT study (quantum-jump unraveling, partial
post-selection ζ). This file is **routing only**. It does not restate the
Research Charter and it contains no scientific values.

## Before substantive scientific work

Read, in this order:

1. `research/RESEARCH_CHARTER.md` — governs all research procedure. Human-owned.
2. `research/HANDOFF.md` — navigation and in-flight work.
3. `research/RESOURCE_POLICY.md` — what compute, models and tokens may be spent.
4. `.claude/skills/research/SKILL.md` — how a research task is actually run.
   **Workers read `.claude/skills/research/WORKER_CONTRACT.md` instead** — the
   full Skill is the lead's procedure.

Then read **only** the specific entity files your task touches. Do not preload
`research/state/`.

## Authority

- **Scientific state comes only from `research/state/**`.** Claims, evidence,
  observables, disputes, decisions and sources. Nothing else is support.
- **Non-authoritative**, always: project memory, prior chats, manuscripts
  (`continuousmeasurementslatex/**`, any `*.tex`), `theory/**`, `audit/**`,
  `research/history/**`, `CONTEXT.md`, unmerged proposals, and this file.
  They may be read for orientation and cited as **provenance**. They are never
  **support**.
- A proposal citing another proposal is not evidence. Cite the knowledge plane.

## Working rules

- **Reference existing claims by ID** (`CB-AMP-001`, `DISP-PHI-001`,
  `OBS-BLPROD-001`). Never restate a number that lives in a claim file; cite the
  ID and let the file hold the value. IDs are immutable — correcting a claim
  means a new claim plus `supersedes`.
- **Unresolved disputes stay unresolved.** `research/state/disputes/` holds six
  live disagreements. They are the research queue, not a backlog of things to
  settle by argument, recency or elegance. If your work bears on one, report
  which side the evidence moves and by how much; do not close it.
- **Worker findings go to the execution plane**, never to state:
  `research/tasks/active/<TASK-ID>/` and `research/proposals/`. These are cheap
  and never authoritative.
- **No direct edits to `research/state/**`.** State changes only through a
  proposal that passes red-team review and the human gate. This is enforced by
  `.claude/hooks/guard_research.py`, not by good intentions.
- **Agents never submit HPC jobs.** Not during `/research`, not after Gate A,
  not after experiment approval, not after a successful local pilot, not when
  HPC access returns. Gate A may authorise *preparation* of an HPC package; the
  terminal state is `READY_FOR_HUMAN_SUBMISSION` and the researcher submits
  manually. Designing, writing and validating SLURM scripts is expected work.
- **Autonomous compute is local-only and read-only.** `/research` runs no
  simulation. A bounded local pilot needs explicit human approval first.
  Policy: `research/RESOURCE_POLICY.md`. This machine:
  `research/resource_profile.local.yaml` (gitignored).
- **Preserve negative results.** A kill, a null, a superseded claim and a
  refuted derivation are first-class outputs (charter §4.4). Never quietly drop
  one.

## Known traps

- `analysis/anchor_scan.py` is **known wrong** and still produces
  plausible-looking output (`EV-CODE-ANCHORSCAN-001`). Blocked by the hook.
- `OBS-BL-001` is **retired**: one label covered two different quantities. Use
  `OBS-BLPROD-001` (average-of-products) or `OBS-BLKMR-001`
  (product-of-averages), and never compare across them.
- Untracked outputs under `analysis/` have no provenance record and are cited by
  nothing in state. They are not evidence.
- Data lives in two trees; resolve via `research/state/DATA_ROOTS.yaml` plus the
  machine-local `research/data_roots.local.yaml`. Never hardcode an absolute
  path into state.

## Environment

Python is `.venv/bin/python3`. Validators:

```bash
.venv/bin/python3 research/tools/validate_state.py           # knowledge plane
.venv/bin/python3 research/tools/validate_redteam.py         # Stage 8, per candidate
.venv/bin/python3 research/tools/validate_task.py <TASK_DIR> # completeness + phase lock
.venv/bin/python3 research/tools/validate_resource_policy.py # policy wiring
.venv/bin/python3 research/tools/test_workflow_regressions.py
.venv/bin/python3 .claude/hooks/test_guard_research.py
```

## Task discipline

- **Phases close and freeze.** `research/tools/task_phase.py <TASK_DIR> close
  <stage>` records SHA-256 hashes; editing a frozen artifact afterwards is
  error `M5`. Close Stage 1 *before* dispatching investigators.
- **The falsification plan and its results are different files.** The plan is
  pre-specified and frozen; outcomes go to `FALSIFICATION_RESULTS.md`.
- **Four evidence tiers**: canonical (`research/state/**`), **task-verified**
  (inspected during the run, usable in-task, never canonical automatically),
  provenance, conjecture.
- **Run the duplicate gate before any novelty language**
  (`find_predecessors.py`), and **follow direct provenance before contradicting
  a claim's history** (`resolve_provenance.py`).
- **Vocabulary**: `task-verified` ≠ `canonical`; `proposed promotion` ≠
  `promoted`; read-only work is *T0 analysis compute*, not "no compute".
