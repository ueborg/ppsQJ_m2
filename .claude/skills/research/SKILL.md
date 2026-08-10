---
name: research
description: >
  Operational implementation of research/RESEARCH_CHARTER.md for ppsQJ_m2
  investigations. Load this before opening or working a research task, before
  writing any task artifact, and before assessing whether a candidate direction
  is worth the researcher's attention. Implements the charter stages the
  architecture explicitly deferred to /research - Stage 0 task-scoped source
  reconstruction, Stages 1-3, the Meaningful-Contribution Test A-H, the twelve
  Slop Warnings, the silo-breaking bridge audit, and Stage 9 synthesis. Use for
  any question of the form "what do we actually know about X", "is direction Y
  worth pursuing", "reconstruct the history of claim Z", or "what experiment
  would decide dispute D".
---

# The Research Charter, operationalised

The charter (`research/RESEARCH_CHARTER.md`) is authoritative and this file is
not. Where they differ, the charter governs and this file is the bug. What
follows is the charter turned into steps, checks and artifact formats.

`research/CHARTER_COMPLIANCE.md` records which guarantees are mechanical and
which rest on you doing them. **Everything in this file is the second kind.**
The bookkeeping layer is trustworthy; the judgment layer is you.

**This file is the LEAD's procedure.** Workers do not load it — they load
`.claude/skills/research/WORKER_CONTRACT.md`, which carries the invariants in
about a tenth of the context. Sending the full Skill to every worker was
measured to be one of the two largest sources of wasted context in the first
validation run.

**Resource rules are in `research/RESOURCE_POLICY.md`** and bind this procedure:
compute is local-only and read-only (§§1–2), local pilots need human approval
(§3), **agents never submit HPC jobs at any stage** (§4), models are routed
explicitly per role (§5.4), workers get compact context (§5.1), one retry and no
generic fallback (§5.8), and you stop early when further work cannot change the
decision (§5.9).

---

## 0. The four rules that override everything else

1. **`research/state/**` is read-only.** You propose; the researcher merges. A
   PreToolUse hook enforces this, but the hook is a backstop, not the reason.
2. **"No result" is always a valid return.** Nothing here rewards finding
   something. An investigation that concludes "the existing data cannot decide
   this" has done its job.
3. **A contradiction against the claim you were asked to support must be
   reported.** Suppressing it is the one unrecoverable offence.
4. **Unresolved disputes stay unresolved.** You may report which way evidence
   leans and by how much. You may not close a dispute by argument, by recency,
   by elegance, or because one side is better written.

## 0.05 Phases close, and closing freezes

A task is a **ledger**, not a folder. `TASK_MANIFEST.yaml` records when each
stage closed and the SHA-256 of the artifacts frozen at that point. Close a
stage with `research/tools/task_phase.py`, never by hand:

```bash
task_phase.py <TASK_DIR> init <TASK-ID>
task_phase.py <TASK_DIR> close stage_1_problem      # BEFORE dispatching anyone
task_phase.py <TASK_DIR> dispatch --worker theory=sonnet --skip numerics
task_phase.py <TASK_DIR> close stage_3_candidates   # freezes the falsification PLAN
task_phase.py <TASK_DIR> close redteam_dispatched
task_phase.py <TASK_DIR> close synthesis_closed
```

Editing a frozen artifact afterwards is validator error `M5`. The 2026-08-10
run's `PROBLEM_MEMO.md` — a Stage 1 artifact, written before anyone
investigated — contained "three independent investigators found…". It had been
backfilled, and nothing noticed. A record that can be rewritten to agree with
its own results is not a record.

If a frozen artifact genuinely must change, `task_phase.py … amend` records the
old hash, the new one, the reason and the authoriser. Loud and attributed is
fine; silent is not.

**The plan and its results are separate files.** `FALSIFICATION_PLAN.md` states
what will be attempted and what would kill each candidate, and freezes at
`stage_3_candidates`. Outcomes go to `FALSIFICATION_RESULTS.md`, written
afterwards, never merged back. Check `F1` rejects a results column in the plan.

## 0.055 Independence first, then ONE bounded round

The three investigators start **independently** and their reports are frozen
before anyone sees a peer's work (`close first_pass_frozen`, which refuses if a
dispatched worker produced no report). Only then may the lead open **at most one**
cross-examination round.

Collaboration is **optional and must be justified before it opens** — the
unresolved cross-role dependency, the roles implicated, the concrete question,
and why another invocation has expected decision value. No dependency means
`collaborate none --reason ...`, recorded as `COLLABORATION_NOT_NEEDED`. Do not
run a round because the feature exists.

What it is for: chains no single role can complete. Literature finds result X →
theory checks whether X's assumptions apply to our model → numerics tests the one
consequence that separates the possibilities. Or: numerics finds an estimator
instability → theory asks whether a symmetry predicts that direction →
literature checks whether it is known in prior work. What it is *not* for is
three agents independently rediscovering the same fact, which is what happened
without it.

**Atomic messages only.** You extract the smallest fact or question and send
that, with the IDs behind it. Never forward a peer's report or reasoning trace —
that destroys the independence the first passes bought. Every message needs a
sender, recipients, a type, a requested action, a result class, and something it
traces back to.

**Collaboration never rewrites a first-pass report.** A changed conclusion is
recorded in `COLLABORATION_LOG.yaml` as `FIRST_PASS` / `AFTER_CROSS_EXAMINATION`
/ `WHY_CHANGED`. Continued disagreement is a valid output, and making the record
look consensual in hindsight is not.

**The red team is never a participant** and never sees the round's transcript.

## 0.06 Four evidence tiers

`canonical` (`research/state/**`) · `task-verified` (inspected during *this*
task, recorded in `TASK_EVIDENCE.yaml`, admissible in-task including by the red
team, **never canonical automatically**) · `provenance` (manuscripts,
`theory/**`, `audit/**`, history — origin only) · `conjecture/judgment`.

The task `SOURCE_REGISTER.md` is the task-level authority for source inspection
achieved during the run. Nothing there is promoted; the human merges.

## 0.1 Evidence / Inference / Conjecture / Judgment

Charter §2. Every substantive sentence you write in a task artifact carries one
of these labels, explicitly, in the text:

| class | means | test |
|---|---|---|
| **Evidence** | directly supported by a cited source, derivation, dataset, or executed experiment | can you name the file, ID, or command? |
| **Inference** | follows from evidence but was not itself observed | what is the inferential step, and could it fail? |
| **Conjecture** | plausible, presently unsupported | say so in the same sentence |
| **Judgment** | an assessment of importance, novelty, practicality, or value | whose judgment, on what basis? |

Never present an inference as evidence or a judgment as a fact. In practice the
failure looks like: "the exponent is 1/2" (stated as evidence) when the truth is
"a fit over one window gives 0.5 ± unknown, and the window drifts" (evidence)
plus "so the asymptotic exponent is plausibly 1/2" (inference, contested).

Write it as `[E]`, `[I]`, `[C]`, `[J]` inline. It is ugly and it works.

---

## Stage 0 — task-scoped reconstruction

**Do not attempt global Stage 0.** The registry holds 11 sources of which most
are `not_inspected`. Inspecting all of them before any question can be asked is
how this project stalls. Stage 0 is discharged **per question**.

For the question at hand:

1. **Enumerate the canonical state that bears on it.** Grep
   `research/state/` for the relevant claim, evidence, observable, dispute and
   decision IDs. Read those files in full. Do not preload the rest.
2. **Identify the load-bearing sources** — the ones whose content, if different
   from what we assume, would change the answer. Usually one to three.
3. **Check each load-bearing source's `inspection_level`.**
   - `fully_inspected` or `relevant_sections` → usable as support.
   - `abstract_only` or `not_inspected` → **inspect it now** (the literature
     agent reads the actual PDF under `PAPERS_LIBRARY`), or treat it as
     unavailable.
4. **Refuse to treat an uninspected or unavailable source as evidence.** A
   title, abstract, or search snippet is not evidence for a technical claim
   (charter §4.1). If you cannot open it, say the literature assessment is
   incomplete and name what is missing.
5. **If a missing source prevents a defensible decision, stop and return
   `Infrastructure first`.** That is a successful outcome, not a failure.

Record the result in `SOURCE_REGISTER.md` in the task directory: for each
source, the ID, what was inspected, the date, exactly which claim it supports or
fails to support, and any unresolved interpretation. Anything inspected during
the run is **task-verified**, gets a `TV-*` entry in `TASK_EVIDENCE.yaml`, and
is usable by the rest of the task — including the red team — without waiting for
a merge.

### External research is expected, not exceptional

`research/state/sources/**` and the local PDF library are **not an exhaustive
corpus**. Most relevant literature has never been downloaded here. All four
roles have `WebSearch` and `WebFetch`, with different remits:

- **literature owns broad prior-art coverage** — discovery, closest prior art,
  following load-bearing citations, terminology variants, checking whether a
  purported attribution actually appears in the cited source, and negative-result
  literature.
- **theory and numerics search narrowly**, for a specific theoretical or
  methodological question that can move a candidate. Neither runs a second
  literature review.
- **the red team searches independently** in Stage 8, and is deliberately not
  shown the affirmative team's queries.

Primary sources, in order: journal version → arXiv/author preprint → official
supplementary → primary documentation. **A search-result snippet is discovery,
never evidence**, and neither is an abstract. If the primary source cannot be
opened, record that limitation instead of leaning on the snippet. **A failed
keyword search is not evidence of novelty** — it is evidence about the search.

Everything actually opened is registered as an `EXT-*` entry in
`TASK_EVIDENCE.yaml` and a row in the task `SOURCE_REGISTER.md`, with title,
authors, year, DOI/arXiv, URL, discovery method, inspection level, the exact
sections read, what it establishes, what it does **not**, and
`promotion_status: proposed`. It is task-verified, usable in-task, and canonical
only if the researcher merges it — normally only when it became
decision-relevant.

### Follow direct provenance before contradicting a claim's history

For every claim in scope, run:

```bash
.venv/bin/python3 research/tools/resolve_provenance.py <CLAIM-ID> ...
```

It does **one hop**: cited evidence, `depends_on`, `supersedes`,
`superseded_by`, `contests`, `observable_id`, and — the part that matters — any
**file path named in a prose field** such as `provenance_note`, resolving bare
filenames to their location on disk.

**No worker may declare a claim's recorded history false until those direct
references have been read.** In the 2026-08-10 run, `CB-AMP-096-001`'s own
`provenance_note` says the 0.96 was reinterpreted as an r_c prefactor on
2026-06-10 and names the document. Three workers read the claim, none opened the
document, and a candidate asserted the number "never was an r_c-type prefactor".
It died on that.

This is one hop, deliberately. Do **not** recursively preload project history —
that is the opposite failure and it is what the resource policy exists to
prevent.

**Also reconstruct the code and data**, not only the literature. Which script
produced the number? Does it still exist? Does the stored observable definition
match the formula in the paper? This project has one label that covered two
different quantities for months (`OBS-BL-001`), and one known-wrong analysis
script that still runs (`EV-CODE-ANCHORSCAN-001`).

---

## Stage 1 — problem reconstruction

Artifact: `PROBLEM_MEMO.md`.

Reconstruct the problem **independently of the proposed solution**. Do not start
from a method and reverse-engineer a motivation for it.

Required sections:

- The observed or formal problem.
- The smallest precise research question. If it does not fit in one sentence
  with named quantities, it is not yet precise.
- Why current approaches do not resolve it.
- Which theoretical or operational decision changes with the answer.
- Relevant constraints and information structure.
- **The strongest case that the problem matters.**
- **The strongest argument that the problem is artificial, already solved, or
  unimportant.** Write this one properly. If it is a straw man, the memo is
  worthless.
- What survives that criticism.

## Stage 2 — field, dependency and novelty mapping

Artifacts: `FIELD_MAP.md`, `dependency_graph.json`, `NOVELTY_MATRIX.md`.

A bibliography is not a field map. Map **nodes** (foundational claims, theorems,
assumptions, methods, datasets, benchmarks, software, evaluation conventions,
negative results, open bottlenecks) and **relations** (depends on, generalizes,
contradicts, reinterprets, reuses, empirically validates, assumes, implements,
benchmarks against).

Search under alternative terminology, equivalent mathematical formulations,
adjacent disciplinary language, older terminology, application-specific
terminology, software and benchmark descriptions, and negative-result
literature. **Absence of a phrase is not novelty** (charter §4.2).

Do not speculate about researchers' motives or competence. Cultural and
relational structure is an observable search-and-translation problem, not a
licence for sociological storytelling.

`NOVELTY_MATRIX.md` compares the direction with the closest known work along:
problem definition, information assumptions, mathematical mechanism, guarantee,
empirical evidence, operational constraints, computational cost, reusable
output. One row per comparator.

## Stage 3 — candidates and aggressive refutation

Artifact: `CANDIDATES.md`. **Three to eight candidates. Not more.** Dozens of
superficial variants is the failure mode, not thoroughness.

### The duplicate gate comes first

**Before you may call any candidate new, novel, a finding, or a contribution**,
run the predecessor search over canonical claims, disputes, decisions and
observables — *including* withdrawn, contradicted, superseded and
negative-result records:

```bash
.venv/bin/python3 research/tools/find_predecessors.py "<candidate statement>"
```

Record the closest predecessor for **every** candidate in `NOVELTY_GATE.md`,
and classify: `replication`, `corroboration`, `rediscovery`, `provenance
repair`, or `no predecessor found`. Validator checks `G2`/`G3` enforce that the
record exists and that novelty language is backed by a classification.

Dead records are boosted in the search, not filtered, because they are the ones
most likely to be rediscovered — nobody has them in their working set. In the
2026-08-10 run, candidate C2 was labelled "the finding of the task" while
`METH-EXTRAP-001` (`withdrawn`) already recorded the same content with the same
chi²/dof table. The correct classification was **rediscovery**, and it is not a
lesser outcome: confirming that we already knew something has real value, and
misreporting it as a discovery does not.

Eleven required fields per candidate, none optional:

1. Candidate statement.
2. Strongest affirmative case.
3. Closest known precedent.
4. Strongest **novelty** objection.
5. Strongest **correctness** objection.
6. Strongest **practicality** objection.
7. Strongest **significance** objection.
8. A possible decisive test.
9. Explicit kill criteria.
10. What survives the criticism.
11. Revised or stronger version.

After the first round, **re-pose the surviving problems using what the
refutations taught you.** The second formulation is usually the real one.

## Stage 4 — falsification before scaling

Artifact: `FALSIFICATION_PLAN.md`. Before any large computation:

- Search for counterexamples.
- Test limiting and degenerate cases.
- Construct the smallest analytically transparent model.
- Ask whether the claimed effect can arise trivially.
- Implement the strongest plausible baseline, not a convenient one.
- Identify parameter regimes where the method **should** fail.
- Ask whether the result is an artifact of a definition, a normalization, or the
  simulator.
- Separate mechanism validation from performance optimization.

**A small decisive experiment beats a large benchmark that cannot distinguish
explanations.**

---

## The Meaningful-Contribution Test (charter §5)

Assess **A–H separately**. **Never produce an aggregate score.** A high value in
one dimension does not compensate for a fatal weakness in another; the whole
point of separating them is that they do not trade off.

| | dimension | the question that must be answered concretely |
|---|---|---|
| **A** | Consequential bottleneck | What precise limitation in present knowledge or capability is addressed? What cannot currently be explained, predicted, proved, designed or implemented? What changes if it is solved? "Improving performance" is rejected unless made operational. |
| **B** | Mechanistic contribution | Does it introduce or reveal a *mechanism*? Does it change the causal, operational or mathematical explanation, expose a hidden tradeoff, or move the feasible region? A new label, metric, component or application is not a mechanism. |
| **C** | Discriminability | What observation distinguishes this from existing explanations? What falsifies it? Can competing hypotheses make different predictions? A direction that explains every outcome is not discriminating. |
| **D** | Dependency significance | What does it depend on, and would later work depend on it? Central dependency point, or terminal variant? |
| **E** | Cross-silo value | Which communities or formalisms are connected, what exactly transfers, what obstruction is removed? Analogy-only connections are rejected. |
| **F** | Robustness | Does it survive realistic parameter ranges, alternative baselines and definitions, finite resources? Does it need a narrow tuning regime? |
| **G** | Informative failure | If it fails, what has been learned? Does the failure eliminate a hypothesis or produce a boundary result? |
| **H** | Infrastructure value | Reusable code, data, definitions, benchmarks, tests, dependency maps? Can someone else reproduce and extend it? |

Record as `ASSESSMENT_AH.md`, one section per dimension, each with a verdict and
the reasoning. An unanswerable dimension is recorded as unanswerable — not
skipped, and not filled with something that sounds adequate.

## The twelve Slop Warnings (charter §6)

**Every candidate gets an explicit verdict on all twelve.** Not a summary
judgement: twelve recorded verdicts. Flag it when the main contribution is:

1. Applying an established method to a routine new dataset, model, topology or
   application.
2. Combining two known techniques without identifying a nontrivial interaction.
3. A metric that is a monotone transformation, weighted sum, or rename of an
   existing quantity.
4. Another constraint on a familiar optimization problem, with no change to its
   conceptual structure.
5. Swapping one architecture for another and reporting a small benchmark gain.
6. A theorem whose assumptions largely encode the desired conclusion.
7. A simulation regime constructed mainly because it makes the method look good.
8. Comparison against weak, obsolete, incorrectly implemented, or
   informationally disadvantaged baselines.
9. Treating computational scale as scientific depth.
10. Treating the existence of runnable code as evidence that a research problem
    exists.
11. Claiming silo-breaking novelty from terminology differences alone.
12. Drafting a paper around an artifact before identifying the scientific claim.

**Do not discard a flagged direction silently.** Record why it fails and whether
a stronger reformulation survives. That record is itself a result (§4.4).

Project-specific instances that have already occurred, so recognise them fast:
№6 (three separate derivations of √ζ, each invalidated, each replaced by another
derivation of the same answer), №7 (window chosen after seeing the answer —
`DISP-WINDOW-001`), №9 (a 5,634-realization campaign that sat unanalysed for six
weeks and answered nothing).

## Silo-breaking: the bridge audit (charter §8)

Required **before** any cross-field claim. Artifact: `BRIDGE_AUDIT.md`.

Source field; target field; translation table between definitions and notation;
the object transferred (theorem, method, representation, algorithm, benchmark,
conceptual distinction); assumptions required in the source field; whether they
hold in the target field; the technical obstruction that prevented obvious
transfer; the new result enabled; the closest prior transfer; **reasons the
bridge may be merely terminological**; and a test that would demonstrate
substantive rather than rhetorical integration.

A connection is not accepted because two fields use similar language. It must
alter derivation, prediction, design or implementation.

`DISP-VERTEX-CHIRAL-001` (Ashkin–Teller / Thirring onto the Ising corner) is the
live case. It must not be worked without this audit.

---

## Reporting vocabulary

Four distinctions. The 2026-08-10 run blurred all four, and each one flatters
the run in the same direction.

| do not write | write | why |
|---|---|---|
| canonical | **task-verified** | it was verified in-task; `research/state/**` is unchanged |
| promoted / upgraded | **proposed promotion** | nothing merges without the human gate |
| "no compute was run" | **no new simulation or production compute** | read-only analysis is still T0 analysis compute |
| "the source was promoted" | **the task proposes promoting it** | if the state fingerprint is unchanged, nothing was promoted |

A task whose `research/state/**` fingerprint is unchanged **must not claim that
any source or claim was promoted during it.** Report what the run did, and let
the gate do what the gate does.

## Stage 8 — red team

Handled by the `red-team` agent against
`research/templates/REDTEAM_TEMPLATE.yaml` (**schema v2**: one review per
candidate, each with its own nine attacks and its own verdict of
`killed | survives | survives_scoped | unresolved`; `overall_task_assessment`
must agree with them, checked by rule `R10`). A fatal attack kills the candidate
it applies to and **cannot erase an unrelated survivor** — the v1 single-verdict
form forced the 2026-08-10 report to say "killed" beside prose describing a
survivor.

The reviewer is the one worker allowed to cross role boundaries, and may consume
`task-verified` evidence via `inputs_seen.task_verified`.

Two rules bind the **lead**, not the reviewer:

- The reviewer receives the question, the canonical evidence, the raw
  investigator reports and the bare candidate statements. **It does not receive
  a persuasive lead summary.** `inputs_seen.lead_summary_seen: true` is
  validator error R3.
- All nine mandated attacks must carry a verdict. A missing attack is R4 and
  fails the run. Validate with
  `.venv/bin/python3 research/tools/validate_redteam.py <path>`.

The seven project-specific checks (`extensions.X1..X7`) are additional and can
never substitute for A1–A9.

## Stage 9 — synthesis

Artifact: `RESEARCH_MEMO.md`. Ten required sections: the question investigated;
why it matters; what was previously known; which candidates were eliminated and
why; what survived; the evidence; the remaining uncertainty; the actual
contribution stated without rhetorical inflation; the reusable artifacts; the
next human decision.

**Do not convert the memo into a manuscript.** Prose generation is the final
stage of research, not the process (§4.3), and manuscripts are out of scope.

## Status reporting (charter §12)

Every milestone report uses these six points, in order:

1. What was established.
2. What was refuted.
3. Which assumptions were introduced or removed.
4. Which uncertainties remain.
5. Which files and artifacts changed.
6. Which decision now requires human judgment.

**Activity is not progress.** "Ran 500 simulations" says nothing unless the
state of the scientific question changed.

---

## The decision gate

Every `/research` run ends in **exactly one** of four recommendations. Not two,
not a ranked list, not "pursue but also".

| verdict | when |
|---|---|
| **Pursue** | A candidate survived refutation and red team, A–H is answered with no fatal dimension, and a discriminating experiment exists and is affordable. **`Pursue` never means "run it"** — it means the experiment is worth designing for Gate A. If it needs HPC, the deliverable is a package at `READY_FOR_HUMAN_SUBMISSION`. |
| **Reformulate** | The question as posed cannot discriminate, but a sharper version can. State the sharper version explicitly. |
| **Infrastructure first** | The blocker is a missing source, an unaudited observable, absent metadata, or unregistered evidence — not physics. Name the exact artifact needed. |
| **Stop** | The direction is dead, already answered, or cannot produce an informative failure. Record the kill; it is a `negative_result` proposal, not a deletion. |

`Infrastructure first` and `Stop` are **normal, good outcomes** for this project
at its current maturity. A run that returns `Pursue` every time is broken.

## Negative results

Charter §4.4. A kill, a null, a superseded claim, a refuted derivation and an
abandoned direction are first-class outputs. Never overwrite, hide or quietly
drop one. A negative result is valuable when it eliminates a plausible
mechanism, establishes a boundary condition, reveals that an apparent
contribution is already implicit in existing theory, identifies a regime where a
method cannot work, or prevents duplication.

Killed candidates go to `CANDIDATES.md` with their kill criterion and the
evidence that triggered it, and the task directory moves to
`research/tasks/killed/` rather than being deleted.

## Numeric well-formedness

Any exponent or amplitude you propose requires `parameterization`,
`observable_id`, `fitting_window`, and a `window_sensitivity` scan over **at
least three windows**. **A single-window exponent is not a measurement.** State
whether a crossing was L-extrapolated. Never compare a number computed under one
observable convention with one computed under another — see `OBS-BLPROD-001`
versus `OBS-BLKMR-001`.

A derivation that reproduces an already-known number is a **postdiction** and
cannot raise a claim's support level.

## What you may never do

- Write `research/state/**`.
- Cite a proposal, a chat, project memory, `HANDOFF.md`, a manuscript, or
  `theory/**` as *support*. Provenance only.
- **Submit an HPC or remote-compute job. Ever.** Not after Gate A, not after
  experiment approval, not after a local pilot, not when HPC access returns.
  Prepare the package, reach `READY_FOR_HUMAN_SUBMISSION`, and stop.
- Launch any production simulation. `/research` is read-only, local-only
  analysis and stops at Human Gate A. See `research/RESOURCE_POLICY.md` §§1–4.
- Run `analysis/anchor_scan.py` as evidence-producing analysis.
- Declare a contribution novel or important because a search found nothing
  (§3). Novelty is the researcher's call.
- Invent a missing value. Record the gap and branch on explicit alternatives, or
  ask.
