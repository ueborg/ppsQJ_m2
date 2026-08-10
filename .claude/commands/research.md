---
description: Run a charter-compliant research investigation on a question, stopping at Human Gate A
argument-hint: <research question>
---

You are the **research lead** for ppsQJ_m2. The question is:

$ARGUMENTS

Load the `research` skill (`.claude/skills/research/SKILL.md`) now and follow it.
`research/RESEARCH_CHARTER.md` governs; the skill is its operational form. Read
`research/RESOURCE_POLICY.md` for what this run may spend.

**Resource discipline, as lead:**
- You orient **once**. Workers get compact context — question, role, the
  relevant IDs, exact paths, output schema — never the repository.
- Workers load `WORKER_CONTRACT.md`, not the full Skill. The workflow does this.
- Models are routed explicitly: literature/numerics/theory on **sonnet**,
  red-team on **opus** (all sonnet in `historicalValidation` mode). Escalate
  theory to opus only with a concrete reason recorded in `CHARTER.md`.
- Spawn only the investigators that can materially help (`workers: [...]`).
  Record which you skipped and why. A purely analytic question may need no
  numerics worker.
- **No simulation, no HPC submission, ever.** If new compute would change the
  decision, design the smallest calculation and stop at Gate A.
- At each phase boundary ask: *will another model invocation have enough
  expected information value to justify its cost?* Stop when it will not.

Run these phases. Do not skip one because it looks unnecessary.

**Phase A — orientation (you, in the open).**
Mint `TASK-YYYY-MM-DD-<SHORTNAME>`. Copy `research/tasks/TASK_TEMPLATE/` to
`research/tasks/active/<TASK-ID>/`, then initialise the phase ledger:

```
.venv/bin/python3 research/tools/task_phase.py <TASK_DIR> init <TASK-ID> --mode normal
```

Write `CHARTER.md`: the question, at least two competing hypotheses, the
canonical IDs in scope, and the kill criterion — **before** any evidence is
seen. Then discharge Stage 0 *for this question only*: read the relevant
claim/evidence/observable/dispute files, identify the load-bearing sources, and
check each one's `inspection_level`.

**Follow direct provenance** for every claim in scope — one hop, no recursion:

```
.venv/bin/python3 research/tools/resolve_provenance.py <CLAIM-ID> ...
```

Put what it surfaces into the worker `context`. No worker may declare a claim's
history false without it. Write `SOURCE_REGISTER.md` and `PROBLEM_MEMO.md`, then
**close Stage 1 before dispatching anyone**:

```
.venv/bin/python3 research/tools/task_phase.py <TASK_DIR> close stage_1_problem
```

That freezes the memo and the kill criterion. Anything learned later goes to
`agent_reports/` and the results artifacts — never back into the memo.

**Phase B — independent investigation (the workflow).**
Invoke the workflow by path — the named registry does not resolve project
workflow files:

```
Workflow({ scriptPath: ".claude/workflows/research.js",
           args: { taskId, taskDir, question, context,
                   stage: "investigate",
                   mode: "normal",            // or "historicalValidation"
                   workers: ["literature","theory","numerics"],  // subset OK
                   theoryEscalationReason: "" } })               // opus only w/ reason
```

`context` is **facts only**: the canonical IDs in scope and where to read them.
Do not put a preferred answer, a ranking, or a hint into it. Save each raw
report verbatim to `agent_reports/`.

**Phase B2 — optional collaboration (the workflow), at most ONE round.**
First freeze the independent reports:

```
.venv/bin/python3 research/tools/task_phase.py <TASK_DIR> close first_pass_frozen
```

Then decide whether a round is worth it. If there is no unresolved cross-role
dependency:

```
.venv/bin/python3 research/tools/task_phase.py <TASK_DIR> collaborate none --reason "..."
```

Otherwise justify it *before* opening — the ledger requires all four:

```
.venv/bin/python3 research/tools/task_phase.py <TASK_DIR> collaborate open \
   --dependency "..." --roles literature,theory --question "..." --value "..."
```

then run the workflow with `stage: "collaborate"` and `collab: { question,
dependency, asks: [{to, type, fact, ask, refs}] }`. **You extract each atomic
fact or question yourself** — never forward a peer's report or reasoning trace.
The red team is not a participant. Copy
`research/templates/COLLABORATION_LOG.yaml` into the task, record every
exchange, then `close collaboration_closed`.

If a participant changes its mind, record `FIRST_PASS` /
`AFTER_CROSS_EXAMINATION` / `WHY_CHANGED` in the log. **Never rewrite the frozen
report** to make the outcome look consensual. Preserved disagreement is a valid
output. There is no second round.

**Phase C — candidates (you).**
Record the dispatch (`task_phase.py <TASK_DIR> dispatch --worker role=model …
--skip role`). Write `FIELD_MAP.md`, `NOVELTY_MATRIX.md`, `CANDIDATES.md` (3–8
candidates, 11 fields each) and `FALSIFICATION_PLAN.md` — the **pre-specified**
plan, with no results in it. **Preserve disagreement between investigators** —
where they disagree, that is a dispute proposal, not something to average away.

**Run the duplicate gate before using any novelty language.** For every
candidate: `find_predecessors.py "<statement>"`, then record the closest
predecessor and a classification (`replication` / `corroboration` /
`rediscovery` / `provenance repair` / `no predecessor found`) in
`NOVELTY_GATE.md`. Then close the stage — this freezes the candidates, the plan
and the gate:

```
.venv/bin/python3 research/tools/task_phase.py <TASK_DIR> close stage_3_candidates
```

**Phase D — red team (the workflow).**
Invoke the `research` workflow again with `stage: "redteam"` and
`candidates: [...]` — the **bare** candidate statements, no advocacy. The
reviewer must not receive your synthesis. Then confirm:
`.venv/bin/python3 research/tools/validate_redteam.py research/tasks/active/<TASK-ID>/REDTEAM.yaml`

**Phase E — decision gate (you).**
Write `FALSIFICATION_RESULTS.md` (outcomes, **never merged back into the
plan**), `ASSESSMENT_AH.md` (A–H separately, **no aggregate score**),
`SLOP_WARNINGS.md` (explicit verdict on all twelve), and `RESEARCH_MEMO.md`.
Then `RECOMMENDATION.md` with **exactly one** verdict: `Pursue`,
`Reformulate`, `Infrastructure first`, or `Stop`. `Infrastructure first` and
`Stop` are good outcomes here.

Respect the per-candidate verdicts: a killed candidate and a scoped survivor can
coexist, and the summary must match them.

**Phase F — experiment design (you), only if something survived.**
Write `proposed/EXPERIMENT_SPEC.yaml` for the *smallest discriminating* test.
It must complete: "if the observable takes value X in range R, H1 is excluded;
if X', H2 is excluded." State whether it fits the local budget
(`RESOURCE_POLICY.md` §3) or needs HPC. If it needs HPC, the deliverable is a
reviewed package whose terminal state is `READY_FOR_HUMAN_SUBMISSION`.
**Do not execute it, and never submit it.**

**STOP AT HUMAN GATE A.**

Finally write `RESOURCE_USAGE.md` in the task directory (non-authoritative):
roles invoked, models used, retries, approximate usage, local compute executed,
and whether any worker was unnecessary in hindsight.

Close the ledger, then validate and report:
```
.venv/bin/python3 research/tools/task_phase.py <TASK_DIR> close synthesis_closed
.venv/bin/python3 research/tools/validate_task.py <TASK_DIR>
.venv/bin/python3 research/tools/validate_state.py
```
`validate_state.py` must show `research/state/**` **unchanged**. Report using the
§12 six-point format. Do not merge anything into state; do not commit.

**Vocabulary in the report:** `task-verified` is not `canonical`; `proposed
promotion` is not `promoted`; read-only work is `T0 analysis compute`, not "no
compute" — say **no new simulation or production compute**. If the state
fingerprint is unchanged, nothing was promoted, so do not say it was.
