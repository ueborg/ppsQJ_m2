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

**Priority order when these trade off.** Cost is last, and it is still real:

1. scientific validity;
2. expected information value;
3. researcher attention saved;
4. experimental compute saved;
5. token / model cost.

Token cost never dominates a high-value research decision. It routinely decides
a low-value one, and most of a run is low-value one.

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

### 5.4 Adaptive model routing

**Governing principle.**

> Use the model with the highest expected scientific value for the decision
> being made.

This **replaces** the earlier principle, *"use the cheapest model unless a
stronger model is clearly necessary."* That rule spent researcher attention to
save model budget, which is the wrong trade at a genuine research bottleneck.
Two things remain true and are not in tension:

- **Never let a worker inherit the lead's model by accident.** Every role is
  routed explicitly, by alias, never by pinned ID.
- **Do not make every worker Tier 3.** The 2026-08-10 run that silently put
  every worker on the strongest model is the standing negative example. The
  goal is **selective** use of stronger models.

The machine-readable table is **`research/model_routing.yaml`**. This section is
its prose. `validate_resource_policy.py` cross-checks the YAML, this table, the
four agent definitions and the routing table embedded in
`.claude/workflows/research.js`; a disagreement is an error, not a nuance.

#### 5.4a The three tiers

| tier | alias | for |
|---|---|---|
| **Tier 1** | `sonnet` | routine execution: work dominated by volume, not by difficult inference — repository archaeology, search, provenance extraction, routine literature extraction once sources are identified, parsing logs/CSV/JSON, predefined analyses, profiling, straightforward implementation, mechanical prototypes, repeat benchmarks, ordinary debugging, formatting artifacts, regression validation. **The workhorse.** |
| **Tier 2** | `opus` | difficult scientific reasoning: target-measure correctness, Radon–Nikodym / Feynman–Kac, finite-population bias, hard estimator questions, interpreting unexpected numerics, universality/field-theory transfer, constructing falsifiers, statistically sound experimental design, resolving conflicting evidence, nontrivial control variates, physical-vs-methodological calls, literature+theory+numerics synthesis, serious red-team attacks. |
| **Tier 3** | `best` | the deepest bottlenecks: a fundamentally new algorithm from first principles, an exact or controlled change of measure, a rare-event/particle method specific to the PPS process, an analytic phase boundary, hard RG/field theory, the optimal or approximate Doob/value function, a stubborn theory–numerics contradiction, a new direction after several mechanisms failed, combining partial algorithms into a new architecture, a genuinely new estimator, the deepest surviving candidate after ordinary red team, an important disagreement still unresolved after Tier-2 work, long-horizon open synthesis. |

**Tier 3 is not for mechanical work whose parent task is important.** Not for
benchmarks, not for log parsing, not for reading many PDFs, and not for running
several expensive models as a voting panel.

#### 5.4b Role defaults

Defaults, not ceilings. Every entry may be escalated with a recorded escalation
(§5.4e) and de-escalated when the actual subproblem is mechanical.

| role | economical | **normal** | deep | regression |
|---|---|---|---|---|
| `literature` | tier 1 · `sonnet` | tier 1 · `sonnet` | tier 1 · `sonnet` | `sonnet` |
| `theory` | tier 1 · `sonnet` | **tier 2 · `opus`** | **tier 3 · `best`** | `sonnet` |
| `numerics` | tier 1 · `sonnet` | tier 1 · `sonnet` | tier 1 · `sonnet` | `sonnet` |
| `red-team` | tier 2 · `opus` | tier 2 · `opus` | tier 2 · `opus` | `sonnet` |
| lead | advisory tier 1 | advisory tier 2 | advisory tier 3 | advisory tier 1 |

Escalation triggers per role are enumerated in `research/model_routing.yaml`
under `roles.*.escalate_to_tier_2_for` / `escalate_to_tier_3_for`. In outline:

- **literature** → Tier 2 for difficult prior-art synthesis, conflicting papers,
  whether two constructions are genuinely equivalent, a subtle methodological
  assumption, source-transfer questions needing scientific reasoning. → Tier 3
  only rarely: a major synthesis across fields with strongly differing
  terminology, or an obscure theoretical connection essential to a high-value
  candidate. **Never** to read or download more PDFs.
- **theory** → Tier 1 for routine algebra, verification of a straightforward
  derivation, known formula substitution, mechanical symbolic work. → Tier 3
  for first-principles derivation, new algorithm invention, analytic
  phase-boundary work, hard field theory/RG, exact stochastic control,
  unresolved conceptual contradictions, genuinely novel synthesis. **Theory has
  comparatively easy access to Tier 3.**
- **numerics** → Tier 2 for subtle estimator design, bias/variance reasoning,
  experimental design, nontrivial inference, diagnosing a surprising result,
  deciding whether a scaling is real or an artifact. → Tier 3 only for
  inventing a genuinely new numerical method or a problem that resists ordinary
  reasoning. **Never** to run benchmarks or parse logs.
- **red-team** → Tier 3 when the candidate is potentially load-bearing for the
  paper, when the affirmative team claims a major new theoretical or algorithmic
  result, when a Tier-2 pass left an important disagreement unresolved, when the
  candidate cost substantial compute and a false positive would be especially
  costly, when it involves subtle exactness/bias claims, or when the result
  could redirect a large production campaign. → Tier 1 for routine regression
  red-team work.

**The lead's row is advisory.** The lead is the main session and its model is
the researcher's choice; the engine does not change it mid-session. If the
session model is weaker than the synthesis warrants, **do not redesign
orchestration to change it** — dispatch a Tier-3 advisor worker for that one
high-value step and leave the lead where it is.

#### 5.4c Heterogeneous by design

One `/research` run is expected to look like:

```
lead synthesis        opus       theory derivation   best
literature extraction sonnet     numerical profiling sonnet
estimator analysis    opus       prototype execution sonnet
red team (ordinary)   opus       deep final attack   best
```

A mechanical worker never inherits the lead's expensive model. A difficult
worker is never downgraded because the lead happens to be on Sonnet. **Model
choice is role- and subproblem-specific.**

#### 5.4d No failure is required before escalating

Any rule of the form *"escalate only after the cheaper model has failed"* is
**withdrawn.** It bought a wasted pass and a wasted read of it. The test is:

> Is there a plausible material gain from stronger reasoning **at this decision
> point**?

If yes, escalate. For a genuinely difficult problem, **start** at Tier 2 or
Tier 3 rather than staging a throwaway Tier-1 pass first.

#### 5.4e Escalation records are five lines, not an essay

Before escalating above the role's posture default, record:

```
MODEL_ESCALATION:
  from: sonnet
  to: opus
  role: theory
  question: whether candidate C3 preserves the exact target measure
  decision_at_stake: prototype / kill
```

After the stronger-model result, add one field:

```
  material_value: changed_conclusion | new_derivation | caught_error |
                  confirmed_existing | no_material_gain
```

That is the whole procedure. No approval round-trip, no justification memo.
Records live in `<TASK_DIR>/RESOURCE_USAGE.md` under **Model routing**, which is
non-authoritative and is never scientific evidence. Their purpose is to let us
score routing empirically after several real tasks.

#### 5.4f Effort is not capability

Where the runtime exposes effort, treat it as an independent axis. **Do not pair
Tier 3 with maximum effort by reflex.** Use high effort when substantial context
must be integrated, when a long derivation is required, when a high-stakes
candidate needs adversarial checking, or when the task is genuinely open-ended.
Routine questions asked of a strong model use normal effort.

And the converse, which matters more: **a high-effort Tier-1 search is not a
substitute for a Tier-2/3 reasoning step when the bottleneck is conceptual
rather than informational.** More searching does not fix a wrong derivation.

#### 5.4g Aliases and availability

Use aliases — `sonnet`, `opus`, `best` — not pinned version IDs. Tier 3 routes
to **`best`** specifically so the configuration stays correct where Fable is
unavailable: `best` resolves to the strongest model this installation offers and
degrades to an Opus-class model otherwise.

A Tier-3 request **never crashes a run.** The resolution order is
`best → fable → opus`; if the runtime rejects an alias the router steps down and
logs the substitution, and the tier actually used is what gets recorded.

**Two dispatch paths, one of which is narrower.** Verified against the installed
CLI (2.1.227), whose alias set is `sonnet`, `opus`, `haiku`, `fable`, `best`
(plus `[1m]` and `opusplan` variants):

- **Agent frontmatter and workflow routing accept `best`.** `.claude/agents/*.md`
  takes the alias string as written, and `.claude/workflows/research.js` routes
  it. This is the normal path and it is what §5.4b describes.
- **The Agent tool's own `model` parameter enumerates `sonnet | opus | haiku |
  fable`** in this build — no `best`. If the lead dispatches a worker *directly*
  rather than through the workflow, Tier 3 is requested as **`fable`**, and if
  that is unavailable, as `opus`. Same tier, same record; only the spelling
  differs, because that call site validates against a fixed enum.

Neither path may pin a version ID. If a future build widens or narrows the enum,
`validate_resource_policy.py` P15 is where the supported set is asserted.

Requiring Fable *specifically* — appropriate for a reproducibility experiment
that must pin the model family, not for ordinary research — is done with the
explicit `fable` alias: `roleAlias` in the workflow args, `model: fable` in an
agent definition, or `claude --model fable` for a session. Prefer `best` in
engine routing.

#### 5.4h Tier does not change the standard

A stronger model earns no evidential discount. Tier-3 workers **may** invent —
new hypotheses, new algorithm architectures, alternative formulations, new
falsifiers, reinterpretations of negative results, cross-field combinations —
and are not restricted to auditing weaker models' output. Everything they
produce is still subject to the evidence rules, prediction-before-test, the red
team, the claim-strength audit and Human Gate A. Independent first passes may
deliberately run at *different* tiers (e.g. theory A on `best` via a
first-principles route, theory B on `opus` via an independent route) provided
the independence is **representational or methodological** — not merely two
expensive models voting — and neither sees the other before first-pass freeze.

#### 5.4i Posture

A research question may declare a posture in its prompt:

```
MODEL_POSTURE: economical | normal | deep
```

- **economical** — mostly Tier 1; Tier 2 only for a real scientific bottleneck;
  Tier 3 only with an explicit recorded escalation.
- **normal** — adaptive routing as above. **`/research` defaults to this.**
- **deep** — Tier 2 freely available; Tier 3 encouraged at major theory,
  architecture and synthesis bottlenecks. Appropriate for e.g. analytic
  phase-boundary derivation or fundamental sampler-architecture search.
  **`deep` does not mean "make everything Tier 3"** — mechanical roles stay on
  Tier 1 in a deep run, and a deep run with `literature` on `best` is a routing
  bug.

Historical/regression validation defaults to **economical** and additionally
applies the §6 overrides.

### 5.4j Collaboration budget

At most **ONE** bounded cross-examination round per task, only after every
first-pass report is frozen, only between the implicated affirmative roles, and
only when the lead has recorded the dependency, the question and the expected
decision value. A collaboration answer uses the **role's routed model** for this
run, not the lead's — a continuation does not inherit the lead's model, in
either direction. A response is not a retry and does not license a retry loop.
The red team never participates.

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

### 5.11 Did the stronger model buy anything?

`RESOURCE_USAGE.md` carries a **Model routing** section so that stronger-model
spending can be scored after the fact rather than argued about in advance. Per
run, record:

- Tier-1 (`sonnet`), Tier-2 (`opus`) and Tier-3 (`best`) invocation counts;
- the posture and how it was chosen;
- each escalation (§5.4e) with its `decision_at_stake`;
- each escalation's `material_value` — `changed_conclusion`, `new_derivation`,
  `caught_error`, `confirmed_existing`, or `no_material_gain`;
- any Tier-3 request that the runtime degraded, and to what.

**Exact token accounting is not required** and should not be invented; Claude
Code does not expose it reliably per subagent. Report what is exposed and leave
the rest blank. The question this section answers is not "what did it cost" but
**"did the stronger model buy scientific value"** — after several real tasks the
`no_material_gain` rate is what tells us to tighten or loosen routing. A run of
`confirmed_existing` results at Tier 3 is evidence the escalation triggers are
too loose; a `caught_error` at Tier 3 typically repays a great deal of compute.

---

## 6. Historical / regression validation mode

A reduced mode for exercising the machinery on a settled case. Posture is
**economical** and the §5.4b defaults are overridden: **all four workers use
`sonnet`, and Tier 3 is not available.** Regression mode is the one place where
"cheapest that suffices" is still the right rule — the answer is already known,
so there is no discovery to buy.

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
