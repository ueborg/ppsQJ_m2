---
lifecycle: active
authoritative_for: compute, model and token resource policy
owner: human
last_reviewed: 2026-08-10
contains_scientific_values: false
---

# Resource Policy — ppsQJ_m2

**Authoritative for how agents may spend compute, model usage and researcher
attention.** The Research Charter governs *what* is good research; this file
governs *what it may cost*. Where they conflict on evidential standards, the
**charter wins** — see §3.0.

This file contains no scientific values and no machine-specific paths. Concrete
hardware numbers live in the gitignored `research/resource_profile.local.yaml`.

---

## 1. Autonomous compute is local-only

**There is currently no usable HPC access.** Every computation an agent may
initiate autonomously must execute locally, on the researcher's current Mac.

The canonical policy never names a machine. It refers to
`research/resource_profile.local.yaml`, which is machine-local, gitignored, and
regenerated per machine from `research/resource_profile.local.yaml.example`.
Read that file to learn what this machine can do; do not assume, and do not copy
its values into `research/state/**`.

If the profile is absent, treat the machine as **unknown** and run nothing
beyond trivial inspection until it is generated.

## 2. `/research` is read-only analysis, not simulation

During `/research`, numerical work uses, in strict order of preference:

1. registered canonical evidence (`research/state/evidence/`),
2. existing aggregates,
3. re-analysis of existing data,
4. analytic or symbolic calculation,
5. tiny diagnostic calculations — **only** where they directly discriminate
   between hypotheses.

**`/research` may never autonomously launch** a new trajectory campaign, a
cloning campaign, a finite-size production sweep, a population-dynamics
campaign, a broad parameter scan, a large Monte Carlo campaign, or a
production-style benchmark.

The numerics worker's default job is to **interrogate existing evidence**. A
5,634-realization campaign once sat unanalysed for six weeks; the bottleneck in
this project has never been a shortage of data.

If new compute would materially improve the decision, `/research`
**designs the smallest useful calculation and stops at Human Gate A.**

## 3. Human-approved local pilots only

A later execution workflow may run a local pilot **only after explicit human
approval**. Default budget, unless the researcher authorises otherwise:

- expected wall time **≤ 10 minutes**;
- **at most one** CPU-intensive calculation at a time;
- **no nested multiprocessing**;
- **no BLAS/OpenMP oversubscription** — pin thread counts explicitly
  (`OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, …), because none are set by
  default on this machine and NumPy will otherwise take every core;
- expected memory **comfortably below physical RAM**, judged against the profile
  and against current swap pressure, not against the RAM figure alone;
- scratch and results stored with provenance;
- **no broad sweep disguised as many individually-small jobs.**

Before any nontrivial local pilot, estimate and record: wall time, memory,
worker/thread count, number of parameter points, expected output size, and
**the scientific decision the calculation can change.** A calculation that
cannot change a decision does not need to run.

If it does not fit comfortably on the current machine: **redesign it, or prepare
it for later HPC. Do not automatically scale it up.**

---

## 4. HPC is permanently human-submit only

**This supersedes all earlier wording suggesting HPC execution becomes permitted
after a gate. It does not.**

**No AI agent may ever submit an HPC or remote-compute job.** This holds during
`/research`, after Gate A, after experiment approval, after a successful local
pilot, and when HPC access returns.

**Agents MAY:** design HPC experiments; generate SLURM scripts; review and
validate SLURM scripts; prepare parameter manifests; estimate CPU, RAM and
wall-time requirements; validate job-array construction; and prepare the exact
submission commands for the researcher.

**Agents MAY NOT execute those commands.**

The terminal state of an HPC-ready workflow is:

```
READY_FOR_HUMAN_SUBMISSION
```

The researcher submits manually. There must never be an HPC-operator agent,
autonomous `sbatch`/`srun`/`qsub`/`bsub`/scheduler submission, autonomous
SSH or remote-shell launch, or an automated submission wrapper.

**Enforcement.** `.claude/hooks/guard_research.py` rule **G4** denies scheduler
and remote-launch commands on the full command string, and
`.claude/settings.json` denies them again by prefix. Reading, writing, grepping
and validating SLURM scripts stays allowed — preparing a package is the
permitted work.

**Correct wording.** The old formulation *"no HPC without Gate A"* is too weak
and must not be reintroduced. The rule is:

> **Agents never submit HPC jobs. Gate A may authorise preparation of an HPC
> package for human submission.**

---

## 5. Model usage is a first-class research resource

**The objective is not minimum token use.** It is to

> maximize decision-relevant scientific information per unit of researcher
> attention, model usage and local compute, subject to the full Research
> Charter.

### 5.0 What efficiency may never buy

Efficiency **never** justifies skipping a load-bearing source, skipping
contradictory evidence, suppressing a serious competing hypothesis, omitting the
independent red team, or weakening evidential standards. If a budget and the
charter conflict, the budget yields and the run reports
`Infrastructure first`.

### 5.1 The lead orients once

The lead resolves canonical scope **once**. Workers receive only the smallest
useful context: the research question, their role, the relevant claim /
evidence / observable / source IDs, the exact paths they need, compact factual
context, and the required output schema.

**Workers do not independently reconstruct the repository** unless the
assignment specifically requires it. Do not preload all of `research/state/**`,
audit history, old HANDOFFs, manuscripts, the full source registry, or unrelated
theory documents into each worker.

### 5.2 Compact worker contract

`.claude/skills/research/SKILL.md` is the **lead's** procedure and is not
loaded by workers by default. Workers load
`.claude/skills/research/WORKER_CONTRACT.md`, which carries the invariants only.
The full charter and Skill remain available when ambiguity genuinely requires
them.

**This is context compression, not rule weakening.** Every invariant in the
contract is enforced exactly as before.

### 5.3 No recursive delegation

Worker agents **never** spawn subagents or workflows. Only the main research
workflow owns delegation. Worker tool lists exclude `Task`/`Agent` and
`Workflow`; see §7 of this file's enforcement notes in `WORKER_CONTRACT.md`.

### 5.4 Explicit model routing

Never let a worker inherit the main session's strongest model by accident.
Aliases (`sonnet`, `opus`) are used rather than pinned IDs so this survives
model updates.

| role | normal `/research` | historical / regression mode |
|---|---|---|
| lead (main session) | researcher's selected model | researcher's selected model |
| `literature` | **sonnet** | **sonnet** |
| `numerics` | **sonnet** | **sonnet** |
| `theory` | **sonnet** by default | **sonnet** |
| `red-team` | **opus** | **sonnet** |

`theory` may be escalated to `opus` **only** with a concrete recorded reason: a
genuinely difficult first-principles derivation, a subtle contradiction, or
multiple serious mechanisms that cheaper reasoning cannot separate. Record the
reason in the task `CHARTER.md`. **Do not use Opus because it is available.**

### 5.4b Collaboration budget

At most **ONE** bounded cross-examination round per task, only after every
first-pass report is frozen, only between the implicated affirmative roles, and
only when the lead has recorded the dependency, the question and the expected
decision value. A collaboration answer uses the **role's configured model**, not
the lead's — a continuation does not inherit Opus. A response is not a retry and
does not license a retry loop. The red team never participates.

### 5.5 Agent count and role economy

v1 is: main session as lead, **up to three** Phase-B investigators
(`literature`, `theory`, `numerics`), and **one** red-team worker afterwards.

**Do not spawn a role because it exists.** Before each Phase-B worker the lead
decides whether that independent context can materially improve source coverage,
mechanistic reasoning, or numerical verification. A purely analytic question may
need no numerics worker; a pure-provenance question may need no theory worker.
Record which workers were skipped and why.

The red team is **mandatory for any substantive surviving claim**. No verifier
swarm in v1.

### 5.6 Concise outputs

Worker reports carry only: findings, relevant IDs, decisive
evidence/calculations, contradictions, unresolved objections, falsifiers, and
the recommended next check.

**Do not forward** long transcripts, verbose tool logs, complete papers,
repository summaries, or exploratory scratch reasoning. Detailed scratch
material stays in the task directory.

Normal Phase B: concise structured reports. **Historical/regression mode:
≤ 1000 words per investigator, normally substantially less.**

### 5.7 Search economy

Literature search is **hypothesis-driven**. Do not browse broadly because web
search exists. Start from registered sources relevant to the claim, then
references that directly bear on the disputed proposition, then terminology
variants only where prior art is genuinely in question. Expand only when the
existing sources do not resolve the task-specific question, and **record why the
broader search was justified.**

### 5.7b External research economy

External search is a first-class capability and a token sink. Both are true.

- **Start from load-bearing registered sources**, then expand only along
  references or terms plausibly capable of changing a candidate.
- **Stop once the task-specific proposition is resolved adequately.** Coverage
  is not the goal; deciding the question is.
- **Literature owns broad prior-art coverage.** Theory and numerics search only
  for their targeted specialty question. Two roles running the same search costs
  twice and buys nothing.
- Interesting but non-load-bearing papers go to `PARKING_LOT.md`, uninvestigated.
- Record enough about the search to judge coverage — queries run, what returned
  nothing, sources inspected — but **never dump search transcripts into agent
  context.**
- The red team is the deliberate exception: its search duplicates nothing
  because it is not told what the affirmative team searched.

### 5.8 Failure and retry

A worker gets **one** normal invocation. If it fails for a clearly identified
transient or mechanical reason, **at most one** retry is permitted, recording
the original failure, why the retry is justified, and whether it reuses
completed work.

Never silently fall back to a generic agent, restart an entire phase, create
multiple replacement workers, or loop on retries.

**If a required project agent is unavailable, return `Infrastructure first`.**
Substituting a generic worker produces output that looks like a specialist
report and is not one. The `general-purpose` fallback that existed in
`/research` v1 was removed for exactly this reason.

### 5.9 Early stopping

Stop when all candidates are killed; when an unavailable load-bearing source
blocks a defensible decision; when canonical evidence already answers the
question; when further investigation is redundant; when local analysis cannot
discriminate the remaining hypotheses; or when the next meaningful action needs
human approval or compute outside the local budget.

**Do not keep spending model usage merely because more investigation is
possible.**

### 5.10 Usage awareness

Do not inspect token usage after every call. **At phase boundaries** the lead
asks one question:

> Will another model invocation have enough expected information value to
> justify its cost?

After a `/research` run, write a small **non-authoritative**
`RESOURCE_USAGE.md` into the task directory: worker roles invoked, models used,
retries, approximate usage where exposed, local compute executed, and whether
any worker was unnecessary in hindsight. **This is never scientific evidence.**

---

## 6. Historical / regression validation mode

A reduced mode for exercising the machinery on a settled case. All workers use
**sonnet**.

| worker | scope in this mode |
|---|---|
| `literature` | only source material directly relevant to the historical attribution or comparison. **No broad field search.** |
| `theory` | only the required parameterization / transformation and interpretation questions. **No open-ended mechanism development.** |
| `numerics` | only registered evidence and the existing audit reproduction needed for the historical claim. **No new simulation. No repository-wide archaeology.** |
| `red-team` | all nine mandated attacks against the resulting reconstruction. **No broad new research.** |

Reports ≤ 1000 words each. The nine Stage 8 attacks are **not** reduced — mode
affects scope and model, never evidential standard.

---

## 7. The execution lifecycle

```
/research
   └─ autonomous read-only investigation (T0, local, no simulation)
        └─ HUMAN GATE A
             ├─ if local compute is proposed:
             │     human approves a bounded local pilot
             │        └─ local Mac pilot within the §3 budget
             │             └─ review  →  next human decision
             │
             └─ if HPC is necessary:
                   AI prepares and reviews the HPC package
                      └─ READY_FOR_HUMAN_SUBMISSION
                           └─ HUMAN SUBMITS MANUALLY
                                └─ returned data ingested and reviewed later
```

**At no point does an agent submit the HPC job.** There is no branch of this
diagram in which one does.
