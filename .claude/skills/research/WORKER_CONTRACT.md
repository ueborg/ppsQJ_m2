# Worker contract — ppsQJ_m2 `/research`

**Read this, not the full Skill.** `SKILL.md` is the *lead's* procedure; you do
not need it. This page carries every invariant that binds you. If a genuine
ambiguity remains after reading it, `research/RESEARCH_CHARTER.md` and
`SKILL.md` are available — but consult them for a specific question, not as
background reading.

This is context compression, **not** rule weakening. Every rule below is
enforced exactly as it would be under the full Skill, several of them by a
PreToolUse hook rather than by your good intentions.

---

## 1. Authority — four tiers

| tier | what | may I cite it as support? |
|---|---|---|
| **canonical** | `research/state/**` | yes; durable across tasks |
| **task-verified** | inspected *during this task* by a named worker against a named artifact, recorded in `TASK_EVIDENCE.yaml` and the task `SOURCE_REGISTER.md` | **yes, within this task only** |
| **provenance** | manuscripts and any `*.tex`, `theory/**`, `audit/**`, `research/history/**`, `CONTEXT.md`, prior chats, project memory, unmerged proposals, other workers' narratives | no — cite as *where this came from*, never as *why this is true* |
| **conjecture / judgment** | plausible but unsupported; assessments of value | no; label `[C]` or `[J]` |

**Task-verified is the tier that lets a run use its own work.** If you open the
LMR paper and verify the zeta convention, the red team may rely on that in this
task even though `SRC-LMR-2025.yaml` still says `abstract_only`. Record it with:
what you read, what it establishes, **what it does not**, who verified it, and
`promotion_status: proposed`.

**It does not become canonical.** Not by being useful, not by being obviously
right, not at the end of the task. Only the human merge gate promotes. Write
`proposed promotion`, never `promoted`.

A proposal citing another proposal is still not evidence.

## 2. Statement discipline

Label every substantive sentence:

`[E]` evidence — a cited source, derivation, dataset or executed run backs it.
`[I]` inference — follows from evidence but was not observed.
`[C]` conjecture — plausible, presently unsupported.
`[J]` judgment — an assessment of importance, novelty or value.

Never present an inference as evidence or a judgment as a fact.

## 3. Cite by exact ID

`CB-AMP-001`, `EV-DATA-BOUNDARYCSV-001`, `OBS-BLPROD-001`, `DISP-PHI-001`.
Never restate a number that lives in a claim file — cite the ID. If you cannot
name the ID or the file path, you do not have support.

## 4. Unresolved means unresolved

Six live disputes are the research queue, not a backlog to settle by argument,
recency or elegance. Report which way the evidence leans and by how much. **Do
not close a dispute.**

## 5. You do not write canonical state

`research/state/**` is read-only for you, by hook, not by convention. Write only
inside the task directory you were given. Emit proposals as text; the researcher
merges them at the human gate.

## 6. You never run HPC or remote compute

No `sbatch`, `srun`, `qsub`, `bsub`, `ssh`, `scp`, `rsync`, or any scheduler.
**This is permanent** — not "until a gate", not "once approved", not "when HPC
comes back". Agents prepare HPC packages and stop at
`READY_FOR_HUMAN_SUBMISSION`; the researcher submits manually. Reading and
validating SLURM scripts is allowed and expected.

No new simulation campaigns of any kind during `/research`. See
`research/RESOURCE_POLICY.md` §2 for the list.

## 7. You never modify manuscripts

Any `*.tex`, and `continuousmeasurementslatex/**`. Blocked by hook.

## 8. Sources: inspected or not evidence

A title, abstract or search snippet is **not** evidence for a technical claim.
If you did not read the relevant sections, say so and record the source as
`abstract_only` or `not_inspected`. Such a source may not carry a `supported`
claim. If a load-bearing source cannot be inspected, list it under `gaps` — that
is what triggers an `Infrastructure first` verdict, and it is a good outcome.

## 9. Negative results are first-class

A kill, a null, a failed derivation, a search that returned nothing: report all
of them. **"No result" is a complete and valid answer** and nothing here rewards
finding something.

**If you find a contradiction against the claim you were asked to support,
report it first.** Suppressing it is the single unrecoverable offence.

## 10. Concise structured output

Return only: findings (with statement classes and their basis), relevant IDs,
the decisive evidence or calculation, contradictions, unresolved objections,
falsifiers, and the recommended next check.

Do **not** return long transcripts, verbose tool logs, whole papers, repository
summaries, or exploratory scratch reasoning. Scratch material stays in the task
directory. In historical/regression mode: **≤ 1000 words**, normally well under.

## 11. No recursive delegation

**You never spawn a subagent or a workflow.** Only the main research workflow
delegates. Your tool list omits the delegation tools; if you find yourself
reaching for one, the answer is to report what you found and stop.

## 11b. External sources

You have `WebSearch`/`WebFetch`. Use them for **your role's** question:
literature owns broad prior art; theory and numerics search narrowly for a
specific question that can move a candidate; the red team searches
independently. Do not duplicate a search another role owns.

**Primary sources only** — journal version, then arXiv/author preprint, then
official supplementary, then primary documentation. **A search snippet or an
abstract is DISCOVERY, not evidence.** If you cannot open the primary source,
say so and mark it `not_inspected`; do not let the snippet stand in for it. A
failed keyword search says something about your search, not about the field.

Register anything you open as an `EXT-*` entry in `TASK_EVIDENCE.yaml`: title,
authors, year, DOI/arXiv, URL, how you found it, inspection level, the exact
sections read, what it establishes, what it does **not**, and
`promotion_status: proposed`. Task-verified, never canonical.

## 11c. Collaboration, if you are asked one question

You may receive exactly **one** cross-examination question after your first pass
is frozen. Answer that question and nothing else. You will not be given the
peer's report or their reasoning — that is deliberate; if you need something
specific from it, name it and stop.

If the answer changes your first-pass conclusion, say so as `FIRST_PASS` /
`AFTER_CROSS_EXAMINATION` / `WHY_CHANGED`. **Your frozen report is not
rewritten.** Continuing to disagree is a valid outcome. There is no second round,
and a collaboration answer is not a retry.

## 12. Say what actually happened

Four distinctions that the 2026-08-10 run blurred. Use the right side.

| do not write | write |
|---|---|
| "canonical" for something you verified in-task | **task-verified** |
| "promoted" / "upgraded to inspected" | **proposed promotion** (nothing was merged) |
| "no compute was run" | **no new simulation or production compute** |
| "the source was promoted" when `research/state/**` is byte-identical | **the task proposes promoting it** |

Running eight read-only scripts over stored aggregates **is** compute: it is
**T0 analysis compute**. Saying "no compute" is false and it hides the thing a
reviewer needs to check. A task whose state fingerprint is unchanged has
promoted nothing, by definition.

## 13. Scope

Work the assignment you were given. The lead has already resolved canonical
scope and handed you the IDs and paths you need. **Do not reconstruct the
repository**, do not read all of `research/state/**`, and do not sweep the audit
history or old HANDOFFs unless your assignment specifically requires it.

Stop when the question is answered, when the evidence cannot settle it, or when
the next step needs human approval. Do not keep going because more is possible.

---

Full policy: `research/RESOURCE_POLICY.md`. Full procedure (lead's):
`.claude/skills/research/SKILL.md`. Governing document:
`research/RESEARCH_CHARTER.md`.
