# Architecture proposal

Audit 2026-08-10, Stage 3. Non-canonical. Nothing outside `audit/2026-08-10/` touched.

Companions: `CLAIM_SCHEMA.yaml`, `EVIDENCE_SCHEMA.yaml`, `AGENT_ROLES.md`,
`RESEARCH_WORKFLOW.md`, `MIGRATION_PLAN.md`, `PROPOSED_PROJECT_MEMORY.md`,
`PROPOSED_PROJECT_INSTRUCTIONS.md`.

---

## 1. What the audit says the architecture must prevent

Not abstract failures. These specific ones, all documented in Stages 1 and 2:

| observed failure | structural cause |
|---|---|
| A = 0.96 corrected in June, re-asserted by memory in August | no single owner for a number, and memory is writable prose |
| `paper/main.tex` recorded as existing when it never did | an agent's suggestion entered state with no execution record |
| θ₁ conclusions still resting on Δ_ζ = 1 after Δ ≈ 2 landed | no dependency links between claims, so a retraction does not propagate |
| φ quoted as 0.5 from a window where it drifts 0.19 to 1.02 | fitting window not part of the claim |
| Case A run, aggregated, and simultaneously listed as "not yet run" | status lives in prose in three places |
| 16,344 files of July campaign absent from every document | no registration step between producing data and it existing |
| the [1.5, 3] ν confidence set produced by a script that no longer exists | reproducibility not modelled |
| snapshot gain "largest confirmed lever" while L=48 gives 1.22× | proxy and single-cell results promoted without a discrimination step |
| √ζ re-derived three times after each derivation failed | postdiction accepted as support |

Every one of these is a *representation* failure, not a physics failure. The
architecture below is designed against this list specifically.

## 2. The two planes

**Knowledge plane** (`research/state/`, `research/RESEARCH_CHARTER.md`,
`research/HANDOFF.md`). Durable, small, machine-validated, single-writer.
Nothing enters except through the approval gate.

**Execution plane** (`research/tasks/`, `research/runs/`, `research/proposals/`,
`worktrees/`). Free-for-all for agents. Cheap to create, never authoritative,
never read as truth by another agent without going through the knowledge plane.

The hard rule: **an agent may read the execution plane, but may only cite the
knowledge plane.** A proposal that cites another proposal as support is rejected
at review. This is what stops two agents bootstrapping each other into a
consensus that no evidence supports.

## 3. Proposed tree, with changes to your draft

```text
research/
    RESEARCH_CHARTER.md          # epistemic rules. Rarely changes. Human-owned.
    HANDOFF.md                   # ≤150 lines. Navigation + what is in flight.
    INDEX.md                     # GENERATED. Do not edit.
    ACTIVE_QUESTIONS.md          # GENERATED from claims. Do not edit.

    state/                       # canonical. Written ONLY by the merge step.
        claims/<CLAIM-ID>.yaml           # one file per claim
        evidence/<EV-ID>.yaml            # one file per evidence item
        sources/<SRC-ID>.yaml            # literature
        decisions/<DEC-ID>.yaml          # methodological decisions incl. kills
        observables/<OBS-ID>.yaml        # observable + estimator definitions
        disputes/<DISP-ID>.yaml          # preserved unresolved disagreements

    tasks/
        active/<TASK-ID>/        # charter, findings, sub-agent outputs
        completed/<TASK-ID>/
        killed/<TASK-ID>/        # first-class. Kills are results.

    runs/<RUN-ID>/               # executed compute: config, logs, outputs, manifest
    proposals/<PROP-ID>/         # claim/evidence proposals awaiting review
    experiments/<EXP-ID>.yaml    # approved experiment specs (the HPC gate)

    tools/
        validate_state.py        # schema + referential integrity + path existence
        generate_index.py        # INDEX.md, ACTIVE_QUESTIONS.md
        register_run.py          # run dir -> evidence proposal, with checksums
        check_dependencies.py    # cascade stale flags on retraction

    history/
        sessions/                # per-session narrative records
        legacy/                  # frozen pre-migration material, read-only
```

### Changes from your draft and why

**One file per entity, not one YAML per type.** `claims.yaml` as a single file
guarantees merge conflicts the moment two agents propose claims in parallel,
and it makes git diffs unreadable. Per-entity files give clean diffs, natural
per-claim ownership, and let the validator report failures by file. Cost is
needing a generated index, which `tools/` supplies.

**`observables/` is new and is load-bearing.** The audit found the project has
used at least six locators (B_L as ⟨CMI·S⟩, ⟨CMI⟩, the KMR product ⟨CMI⟩⟨S⟩,
ΔS_L, MI(ends), c_eff) with definitions that changed silently and a documented
20 percent spread in φ across them. If observables have IDs and claims reference
`observable_id`, then changing a definition creates a new ID and every dependent
claim is flagged automatically. Without this, "φ ≈ 0.5" is not a well-formed
statement.

**`disputes/` is new.** You instructed me not to settle the open physics. A
dispute is a first-class object holding two or more competing claims, the
discriminating observable, and what would resolve it. Without it, preserved
disagreement degrades into whichever claim was edited last. The φ = 1/2 versus
φ = 1 tension is `DISP-PHI-001` and neither side is `superseded_by` the other.

**`experiments/` separated from `tasks/`.** An approved experiment spec is the
HPC authorisation token. It must be a distinct, referenceable, immutable object
so that a run can cite the spec it was authorised by. Folding it into a task
directory makes it editable after approval.

**`tools/` is not optional.** A schema with no validator is a style guide.
`validate_state.py` is what makes this architecture different from the current
one: it mechanically enforces the rules that prose currently only requests.

**`INDEX.md` and `ACTIVE_QUESTIONS.md` are generated.** Answering your question
directly: yes, active questions should be derived, not maintained. They are the
projection of claims with `status: open` or membership in an unresolved dispute.
A separately maintained question list is one more thing that can disagree.

## 4. Evidence registry: one registry, typed

You asked whether datasets, results, scripts, derivations and executed
calculations can live in one `evidence` registry rather than in overlapping
`RESULTS_REGISTER` and `DATA_CATALOG`.

**Yes, and they should.** Reasons:

1. The relation that matters is always the same: *claim cites evidence*. Splitting
   the target of that relation across registries means every claim needs to know
   which registry to look in, which is exactly the "which document is
   authoritative" problem in miniature.
2. The audit's hardest cases are hybrids. `agg_caseB_combined.pkl` is a dataset,
   but the amplitude claim rests on a dataset *plus* an analysis script *plus* a
   crossing procedure. In two registries that chain is split. In one, it is a
   list of evidence IDs with typed links.
3. Deduplication. `results/boundary_aggregate.csv` would otherwise appear in the
   data catalogue and again in the results register with different metadata.

The one genuine objection is that a raw dataset and a derivation have almost
disjoint field sets. That is handled by a `type` discriminator with
type-conditional required fields, which `EVIDENCE_SCHEMA.yaml` specifies. Fields
that cannot apply are omitted, not filled with null.

## 5. Two independent axes

The audit's central lesson is that **scientific confidence and reproducibility
are orthogonal**. The ν confidence set is the best-argued negative result in the
project and its generating script no longer exists. The θ₁ code is fully
preserved and its conclusion is dead.

- **Epistemic status** (on claims): `verified` `plausible` `open` `contested`
  `superseded` `refuted`.
- **Reproducibility state** (on evidence): `fully_reproducible`
  `partially_reproducible` `artifact_only` `procedure_only` `chat_only`
  `ephemeral_recovered` `unrecoverable`.

Coupling rule, enforced by the validator: a claim may reach `verified` only if
at least one supporting evidence item is `fully_reproducible` or
`partially_reproducible`. `chat_only` evidence caps a claim at `plausible`
regardless of how convincing the chat is.

## 6. The dependency cascade

This is the single highest-value mechanism and the current system has no
analogue.

Claims declare `depends_on: [CLAIM-ID, ...]`. When any claim's status changes to
`refuted` or `superseded`, `tools/check_dependencies.py` sets every transitive
dependent to `review_status: stale` and lists it in `ACTIVE_QUESTIONS.md`.

Applied retroactively to the audit: `PPS-YZETA-001` (y_ζ = 1) declares
`depends_on: [PPS-VERTEX-001]`. When the vertex dimension moved from 1 to 2.02,
y_ζ = 1 and every θ₁-derived statement about `K_eff ~ L⁰` would have been flagged
the same day instead of surviving ten weeks in memory. Similarly
`METH-EXTRAP-001` (1/√L because ν = 2) depends on `CB-NU-001`, so the moment ν
became unmeasured the extrapolation choice would have been flagged as circular.

## 7. Single-writer table

| information | canonical owner | who may write it |
|---|---|---|
| scientific claims and status | `state/claims/<ID>.yaml` | merge step only, after human approval |
| numerical evidence and provenance | `state/evidence/<ID>.yaml` | merge step only |
| observable and estimator definitions | `state/observables/<ID>.yaml` | merge step only |
| literature and what it supports | `state/sources/<ID>.yaml` | merge step only |
| methodological decisions and kills | `state/decisions/<ID>.yaml` | merge step only |
| unresolved disagreements | `state/disputes/<ID>.yaml` | merge step only |
| epistemic rules | `RESEARCH_CHARTER.md` | human only |
| what is in flight, navigation | `HANDOFF.md` | research lead, no scientific values |
| open questions | `ACTIVE_QUESTIONS.md` | generator only, never hand-edited |
| approved experiment specs | `experiments/<EXP-ID>.yaml` | human approval writes, then immutable |
| executed compute | `runs/<RUN-ID>/` | implementation agent, append-only |
| everything unreviewed | `proposals/`, `tasks/` | any agent, freely |

**No scientific number appears in more than one place.** Prose documents cite
claim IDs. `HANDOFF.md` may say "the boundary amplitude is `CB-AMP-001`". It may
not say "the amplitude is 0.49".

## 8. Anti-slop: mechanisms, not exhortations

Each item from your list mapped to something the validator or the workflow can
actually enforce.

| failure mode | mechanism |
|---|---|
| re-deriving the known answer after derivations fail | derivations carry `predicts:` and `registered_before_evidence: true/false`. A derivation whose support is only that it reproduces an already-known number is `role: postdiction` and **cannot** raise a claim above `plausible`. |
| fitting-window shopping | claims of type `exponent` require `fitting_window` **and** `window_sensitivity`, a scan across at least three windows. A single-window exponent is schema-invalid. |
| proxies treated as production wins | evidence carries `metric_id`. Only metrics listed in `state/decisions/DEC-MASTER-METRIC.yaml` support `domain: production`. A proxy result is `domain: methods`, permanently. |
| agent-invented paths | every path field requires `verified_at` and `sha256` for files. `validate_state.py` stats the path and fails on absence. This alone would have blocked `paper/main.tex`. |
| parameterization confusion | claims of type `amplitude` or `exponent` require `parameterization` (e.g. `lambda_c`, `r_c`). Two claims differing only in parameterization are not competing, and the validator refuses to link them as `contests`. |
| silent observable redefinition | `observable_id` required. Changing a definition mints a new ID and cascades staleness. |
| forgotten negative results | `tasks/killed/` and `state/decisions/` are first-class and indexed. A kill record is required to close a task as killed. |
| archived theory read as current | every `.md` under `research/` and `theory/` carries YAML front-matter `lifecycle: active \| superseded \| historical` plus `superseded_by`. Validator fails on missing front-matter. Agents must not cite `historical`. |
| large HPC before a discriminating test | `experiments/` gate, see `RESEARCH_WORKFLOW.md` §5. A run without `authorised_by: EXP-ID` cannot be registered as evidence. |
| concurrent edits to one conclusion | single-writer table above. Agents cannot write `state/` at all. |
| conclusions living only in chat | a session that produces a finding must emit a proposal. `CHAT` evidence caps claims at `plausible` and appears in `ACTIVE_QUESTIONS.md` under "unregistered findings". |

### The hypothesis / discrimination separation you asked for

Enforced by two schema fields rather than by discipline:

- Every claim has `discriminating_evidence: [EV-ID]`, meaning evidence that
  **could have come out the other way**.
- Evidence has `role: discriminating | supporting | postdiction | diagnostic`.

A claim reaches `verified` only with at least one `discriminating` evidence item
whose reproducibility is not `chat_only`. Everything else is hypothesis, however
elegant. Applied to the audit: the √ζ derivations are `postdiction`, the x_J
measurement is `discriminating`, the entropy-scaling log-to-area result is
`discriminating` and would immediately outrank most of the current ledger.

## 9. What this does not solve

Honest limits, so the system is not oversold.

- It cannot tell you whether the physics is right. It can only ensure that what
  is claimed matches what was executed.
- It adds real friction. Registering a run properly costs minutes. That is the
  intended trade and it should be defended when it is annoying.
- YAML validation catches structure, not meaning. Nothing stops a well-formed
  claim from being wrong. The red-team role and the human gate exist for that.
- It will not survive an owner who bypasses it under deadline. The migration
  plan therefore keeps the mandatory surface deliberately small.
