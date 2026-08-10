---
lifecycle: active
authoritative_for: charter compliance status
last_reviewed: 2026-08-10
---

# Charter compliance

**The question this file answers:** if you launch a research task, which charter
guarantees are actually **enforced by the system**, and which rely only on
**agent obedience**?

Enforcement legend:
- **MECHANICAL** — `validate_state.py` or the schema rejects the violation.
- **STRUCTURAL** — the repository layout or a gate makes violation awkward but
  not impossible.
- **OBEDIENCE** — nothing stops an agent ignoring it. Prose only.

---

## Summary

> **Updated 2026-08-10 (hardening pass).** Changes: Stage 8 moved from
> OBEDIENCE to MECHANICAL; Stage 0 partially discharged (3 sources inspected);
> §10 item 1 strengthened by the observable audit.
>
> **Updated 2026-08-10 (`/research` v1 build).** The workflow the deferred
> requirements were assigned to now exists. §5 (A–H), §6 (twelve slop
> warnings), Stages 1, 2, 3 and 9 move from *not implemented* to **WORKFLOW +
> STRUCTURAL**: each is a required task artifact whose presence and completeness
> `research/tools/validate_task.py` checks. **Their quality is still
> unchecked.** Two invariants that were pure prose became MECHANICAL: no write
> to `research/state/**`, and no HPC or remote execution during a research run,
> both enforced by `.claude/hooks/guard_research.py` plus deny rules in
> `.claude/settings.json`. Charter Appendix B B.1–B.4 were confirmed resolved by
> the researcher on the same date.

| | count |
|---|---|
| Implemented and MECHANICALLY enforced | 13 |
| Implemented, WORKFLOW-enforced (presence and completeness, not quality) | 6 |
| Implemented, STRUCTURAL enforcement only | 7 |
| Implemented, OBEDIENCE only | 9 |
| Partially implemented | 4 |
| Not implemented | 2 |
| Intentionally deferred | 3 |

**The honest headline: the bookkeeping layer is mechanised, the process layer is
now at least checked for completeness, and research judgment remains
unmechanisable.** A validator can confirm that an A–H assessment exists with a
verdict on every dimension and no aggregate score. It cannot confirm that the
verdicts are any good. Assume that asymmetry when delegating.

A third enforcement level is therefore used below:

- **WORKFLOW** — `validate_task.py` fails the task if the record is missing or
  incomplete. Nothing evaluates whether the content is correct.

---

## §1 Mission

| provision | status | enforcement |
|---|---|---|
| Do not maximize hypothesis/simulation/extension count | partial | OBEDIENCE. No counter penalises volume. The T0–T4 compute gate limits simulation count indirectly. |
| Maximize reduction of uncertainty, early elimination | partial | STRUCTURAL. `tasks/killed/` and `state/decisions/DEC-KILLS-001` make eliminations first-class and citable. |
| Reduce, not enlarge, the direction space | not implemented | OBEDIENCE |
| Never optimize for paperability | implemented | STRUCTURAL. Manuscripts are declared non-authoritative and excluded from scope. |

## §2 Epistemic hierarchy

| provision | status | enforcement |
|---|---|---|
| Nine-level priority order | not implemented | OBEDIENCE. Not representable as a check. |
| Evidence / Inference / Conjecture / Judgment | **implemented** | **MECHANICAL** — `statement_class`, validator E13 |
| "Do not present an inference as evidence" | **implemented** | **MECHANICAL** — E13 requires the field; E17 blocks `judgment` + `supported`; E18 blocks `conjecture` + `supported` |

## §3 Human authority

| provision | status | enforcement |
|---|---|---|
| Six reserved decisions | implemented | STRUCTURAL. No agent writes `state/`; two human gates. |
| May not declare novelty because searches found nothing | partial | OBEDIENCE + STRUCTURAL. `sources/` requires `inspection_level`, but there is no novelty field to guard. |
| Missing info: branch explicitly or ask | partial | STRUCTURAL. `unknown_recoverable`, `evidence_note`, `confidence: unassessed` all exist. Nothing forces their use over invention. |

## §4 Non-negotiable rules

| provision | status | enforcement |
|---|---|---|
| 4.1 No fabricated support | **implemented** | **MECHANICAL** — E8 stats every asserted path; E3 resolves every ID. This is what would have caught `paper/main.tex`. |
| 4.1 Title/snippet match is not evidence | **implemented** | **MECHANICAL** — E20 blocks a `supported` claim leaning on a source below `relevant_sections` |
| 4.1 State when literature assessment is incomplete | implemented | MECHANICAL — `inspection_level` required, E20 |
| 4.2 No novelty by vocabulary | not implemented | OBEDIENCE. `search_terms` exists on sources as a prompt, nothing more. |
| 4.3 No premature manuscript production | deferred | OBEDIENCE. Manuscripts are out of scope by instruction. **When they return, this needs a gate.** |
| 4.4 Preserve negative results | **implemented** | **MECHANICAL + STRUCTURAL** — `claim_kind: negative_result`, `tasks/killed/`, `DEC-KILLS-001`; superseded claims are never deleted; `architecture_status_legacy` preserves migration history |

## §5 Meaningful-contribution test (A–H)

**Status: IMPLEMENTED in the `/research` workflow. WORKFLOW enforcement.**
Deliberately not built as static registry machinery. `ASSESSMENT_AH.md` is a
required task artifact; `validate_task.py` check `T6` fails the task if any of
the eight dimensions lacks a filled-in verdict, and check `T6b` fails it if an
aggregate score appears — the charter's prohibition on collapsing A–H into one
number is therefore mechanical. The *content* of each verdict is unchecked.

Partial coverage: **C (discriminability)** is the exception and is strongly
mechanised — `falsifiers` is a required field, `discriminating_evidence` is
required for `supported`, and the `postdiction` role blocks a derivation that
merely reproduces a known number.

**Future component:** a `PROPOSAL_ASSESSMENT.yaml` with eight required fields,
emitted by the research lead before a task opens, validated for completeness but
not for quality.

## §6 Automatic slop warnings

**Status: IMPLEMENTED in the `/research` workflow. WORKFLOW enforcement.** None
of the twelve is checked by static tooling and none will be. `SLOP_WARNINGS.md`
is a required task artifact carrying all twelve as rows; `validate_task.py`
check `T5` fails the task if fewer than twelve carry a verdict. Whether a
verdict is honest is unchecked.
Two have indirect coverage: "a simulation regime constructed because it makes the
method outperform" is partly caught by mandatory `window_sensitivity`, and
"treating computational scale as scientific depth" is partly caught by
`DEC-MASTER-METRIC-001` plus `metric_id`.

**Future component:** a red-team checklist item requiring an explicit verdict
against all twelve, recorded rather than assumed.

## §7 Mandatory research cycle

| stage | artifact | status | enforcement |
|---|---|---|---|
| 0 Repository and source reconstruction | `SOURCE_REGISTER` → `state/sources/` + per-task `SOURCE_REGISTER.md` | **partial** | MECHANICAL for structure + E20; **WORKFLOW per task.** 11 registered; 2 at `relevant_sections` (Jian, KMR), 1 `abstract_only` (LMR), 8 not inspected. **Stage 0 is discharged PER QUESTION, not globally:** a task inspects its own load-bearing sources or returns `Infrastructure first`. |
| 1 Problem reconstruction | `PROBLEM_MEMO.md` | **implemented** | WORKFLOW — required artifact; `T1`, `T2`, and `T8` (statement classes present) |
| 2 Field and dependency mapping | `FIELD_MAP.md`, `dependency_graph.json`, `NOVELTY_MATRIX.md` | **implemented** | WORKFLOW — required artifacts; `T1`, `T2` |
| 3 Candidate generation and refutation | `CANDIDATES.md` (11-field record) | **implemented** | WORKFLOW — `T7` fails outside 3–8 candidates or on a missing required field |
| 4 Falsification before scaling | `FALSIFICATION_PLAN.md` | partial | STRUCTURAL + **MECHANICAL** — required artifact (`T1`); and scaling is now physically blocked rather than discouraged: `/research` may run no simulation, a local pilot needs human approval, and **no agent may submit an HPC job at any stage** (`research/RESOURCE_POLICY.md`, guard rule G4). "Falsification before scaling" is enforced by making agent-initiated scaling impossible. |
| 5 Executable research plan | `EXECPLAN.md` → `tasks/active/<ID>/` | partial | STRUCTURAL |
| 6 Experiment discipline | `EXPERIMENT_SPEC.md` → `experiments/<EXP-ID>.yaml` | partial | STRUCTURAL. Spec fields defined; **`register_run.py` not written**, so "preserve raw data, store configs with outputs, record versions" is unenforced. |
| 7 Claim ledger | `CLAIM_LEDGER.md` → `state/claims/` | **implemented** | **MECHANICAL** — all Stage 7 fields required and validated |
| 8 Independent adversarial review | `RED_TEAM_REPORT.md` → `proposals/<TASK-ID>-redteam/REDTEAM.yaml` | **implemented** | **MECHANICAL** — `research/templates/REDTEAM_TEMPLATE.yaml` requires all nine attacks; `research/tools/validate_redteam.py` fails hard (R4) on any missing attack, and also on incomplete fields (R5), unexplained skips (R6), invalid severity/effect (R7/R8), fatal-without-kill (R9), and reviewer contamination by the lead summary (R3). |
| 9 Synthesis | `RESEARCH_MEMO.md` | **implemented** | WORKFLOW — required artifact; `T1`, `T2`, `T8`. `RECOMMENDATION.md` adds `T3`: exactly one of Pursue / Reformulate / Infrastructure first / Stop |

## §8 Silo-breaking protocol

**Not implemented.** No `BRIDGE_AUDIT.md`. Currently no active cross-field claim,
so this is **intentionally deferred** until one arises. `DISP-VERTEX-CHIRAL-001`
(Ashkin–Teller / Thirring mapping onto the Ising corner) is the case that will
trigger it, and it should not be worked without a bridge audit.

## §9 Open-source and dependency-centered research

**Partially implemented.** `evidence` types `code_implementation` and `test`
exist. `INFRA-GIT-001` records repository state. Not implemented: dependency
tracing, fragile-component identification, duplicate-implementation detection.
OBEDIENCE.

Live instance already recorded: nine `.bak_*` module copies shadowing live
modules, and `analysis/anchor_scan.py` as a known-wrong shared component.

## §10 Theory-specific requirements

**Partially implemented.** Item 1 ("define all objects before use") is
**MECHANICAL** via `state/observables/` and required `observable_id`, and was
materially strengthened on 2026-08-10 by the observable audit
(`proposals/2026-08-10-A`), which traced B_L and CMI from formula to stored
field and found ONE LABEL COVERING TWO QUANTITIES (our average-of-products
versus KMR's product-of-averages). Item 3
("audit every assumption") is MECHANICAL via required `assumptions`. Items 2 and
4–10 are OBEDIENCE.

## §11 Communications / information-theory audit

**Intentionally deferred.** Does not currently bind this project. Three
generalisable checks are carried elsewhere: proxy versus stated phenomenon
(`DEC-MASTER-METRIC-001`, `metric_id`), equal budgets across baselines
(red-team), and costs moved outside the accounting boundary (red-team).

## §12 Research status reporting

**Implemented, OBEDIENCE.** The six-point format is followed in session reports.
Nothing enforces it. "Do not report activity as progress" is unenforceable
mechanically.

## §13 Completion standard

**Partially implemented.** Mechanically checkable subset: claim ledger current
(validator exit 0), negative results preserved, unsupported claims identified
(`epistemic_status`), remaining uncertainty explicit (`confidence`,
`falsifiers`, `state/disputes/`). Not checkable: question precision, substantive
prior-work comparison, alternative explanations considered.

---

## §7 Stage 8 — the nine mandated attacks

Charter requirement, mapped to mechanism. Currently **all nine are OBEDIENCE**:
they are specified in `COWORK_AGENT_SPEC.md` but no red-team report exists and
nothing validates that each was attempted.

| # | charter attack | architecture mechanism | enforced? |
|---|---|---|---|
| 1 | Problem already solved under another formulation | `state/sources/` + literature agent + `search_terms` | OBEDIENCE. Blocked in practice: 0 sources inspected. |
| 2 | Result follows trivially from assumptions | required `assumptions`; `postdiction` role; `registered_before_evidence` | STRUCTURAL |
| 3 | Baseline is disadvantaged | red-team checklist item "alternative estimator or observable"; `state/observables/` makes locators comparable | OBEDIENCE |
| 4 | Gain comes from extra information or resources | `metric_id` + `DEC-MASTER-METRIC-001`; equal-wall-time comparison convention | OBEDIENCE |
| 5 | Theorem fails under dependence, causality, boundary cases | charter §10 items 5–7; no schema field | OBEDIENCE |
| 6 | Experiment measures a proxy, not the phenomenon | `DEC-MASTER-METRIC-001`; proxies are `diagnostic` role only | STRUCTURAL |
| 7 | Contribution disappears under realistic conditions | `scope`, `validity_range`, `window_sensitivity` | STRUCTURAL |
| 8 | Result is statistically or practically negligible | `value.uncertainty_type`; `confidence` | OBEDIENCE |
| 9 | A simpler explanation accounts for the evidence | `contests` / `dispute_id` force the alternative to be named | STRUCTURAL |

**IMPLEMENTED 2026-08-10.** Each attack is a required key with `attempted`,
`finding`, `evidence`, `severity`, `unresolved`, `effect_on_candidate`. A missing
attack is error R4 and fails the run. Self-tested: deleting A5 from a complete
report yields exit code 1.

Charter Stage 8 requirement -> template field -> validation rule:

| charter attack | template key | rule |
|---|---|---|
| already solved under another formulation | `attacks.A1_already_solved_elsewhere` | R4, R5 |
| follows trivially from assumptions | `attacks.A2_follows_trivially_from_assumptions` | R4, R5 |
| baseline is disadvantaged | `attacks.A3_baseline_disadvantaged` | R4, R5 |
| gain from extra information or resources | `attacks.A4_gain_from_extra_information_or_resources` | R4, R5 |
| fails under dependence, causality, boundary cases | `attacks.A5_fails_under_dependence_causality_or_boundary_cases` | R4, R5 |
| measures a proxy, not the phenomenon | `attacks.A6_measures_a_proxy_not_the_phenomenon` | R4, R5 |
| disappears under realistic conditions | `attacks.A7_disappears_under_realistic_conditions` | R4, R5 |
| statistically or practically negligible | `attacks.A8_statistically_or_practically_negligible` | R4, R5 |
| simpler explanation accounts for evidence | `attacks.A9_simpler_explanation_accounts_for_evidence` | R4, R5 |
| review must not rely on affirmative reasoning | `inputs_seen.lead_summary_seen` | R3 |
| (severity must drive the verdict) | `severity` + `verdict` | R9 |

The seven project-specific checks are `extensions.X1..X7` and are explicitly
NON-substitutive: they cannot satisfy A1–A9.

---

## What an agent can and cannot get away with today

**Cannot** (mechanically blocked): submit an HPC, scheduler or remote job at any
stage, gate or approval level, including via `bash -c`, `xargs`, `nohup` or an
`env` prefix; write `research/state/**` by any route; run `git push` or a
destructive git command; modify a manuscript; invoke the known-wrong scan
script; **silently inherit the lead's model** (`validate_resource_policy.py` P3
rejects `model: inherit`); **substitute a generic agent** for a missing project
agent (P5); invent a file path; cite a non-existent claim,
evidence item, observable or source; mark a claim `supported` on chat-only
evidence; mark a judgment or conjecture `supported`; omit statement class, claim
kind, status, or confidence; omit `confidence_basis`; leave a claim depending on
a contradicted claim un-flagged; assert an exponent without parameterization,
observable, fitting window, or a three-window scan; create a one-sided
contest link; lean a `supported` claim on an uninspected source.

**Can** (obedience only, or deferred to the workflow): skip the
Meaningful-Contribution Test; ignore the twelve slop warnings; skip Stages 1, 2,
3 and 9; report activity as progress; assert novelty; propose a direction with no
consequential bottleneck.

**No longer possible:** submitting a red-team pass that omits any of the nine
mandated attacks; an agent submitting an HPC job; a worker running on an
unrouted model.

**Still obedience-only on the resource axis** (`research/RESOURCE_POLICY.md`):
report length, search breadth, early stopping, the lead's role-economy
judgement, and whether a worker's context was actually kept compact. These are
*measured after the fact* in each task's `RESOURCE_USAGE.md`, not prevented.
See `research/tasks/completed/TASK-2026-08-10-AMP096/RESOURCE_USAGE.md` for what
that measurement looks like when the discipline was absent.

## Requirements the future `/research` workflow MUST implement

Explicitly deferred rather than half-built here, so the orchestration is not
constructed twice:

| charter requirement | why deferred | workflow obligation |
|---|---|---|
| §5 Meaningful-Contribution Test A–H | needs a task under evaluation; a static registry cannot host it | require an A–H record before a task opens; forbid a single aggregate score |
| §6 twelve Slop Warnings | same | require an explicit verdict on all twelve at candidate generation |
| §7 Stage 1 problem reconstruction | per-task artifact | emit `PROBLEM_MEMO.md`, including the strongest argument that the problem is artificial |
| §7 Stage 2 field and dependency map | per-question | emit `FIELD_MAP.md`, `dependency_graph.json`, `NOVELTY_MATRIX.md` |
| §7 Stage 3 candidate refutation | per-task | 3–8 candidates, 11 required fields each, kill criteria |
| §7 Stage 9 synthesis | terminal | emit `RESEARCH_MEMO.md`, 10 required sections |
| §8 silo-breaking | triggered by `DISP-VERTEX-CHIRAL-001` | emit `BRIDGE_AUDIT.md` before any cross-field claim |
| §12 status reporting | per-milestone | enforce the six-point format |

**This asymmetry is the current state of the system and should be assumed when
delegating.** The bookkeeping layer is trustworthy. The judgment layer is not yet
mechanised and depends on the charter being read and followed.
