# Migration plan

Audit 2026-08-10, Stage 3. Non-canonical. Nothing here has been executed.

Principle: **preserve, do not rewrite.** No file is deleted. Historical error is
part of the record. The old HANDOFF is frozen, not reconciled.

---

## Phase 0 — Preservation (do first, independent of everything else)

**0.1 Commit the working tree.** 95 untracked source files, including all of
`analysis/var_reduction/` and `theory/VARIANCE_REDUCTION.md`, plus six modified
files. HEAD equals origin/main so nothing committed is at risk, but three weeks
of work exists on one machine. Stage explicitly, never `git add -A`.

**0.2 Bring the loose artifacts into the repository or a backed-up location.**
The `/tmp` recovery is already at
`audit/2026-08-10/recovered_ephemeral/` with a SHA-256 manifest.

**0.3 Register `results/ruche_pull/`** (16,344 files) somewhere durable, or
confirm the `.tgz` archives are backed up. This is 5,634 realizations of data
that no document mentions.

**0.4 Snapshot the audit.** Tag or archive `audit/2026-08-10/` so the
reconstruction survives the migration that follows.

Phase 0 is not architecture. It is the reason the architecture is worth having.

---

## Phase 1 — Freeze history

**1.1** `git mv theory/HANDOFF.md research/history/legacy/HANDOFF_frozen_2026-08-10.md`

Prepend a header:

```markdown
> FROZEN 2026-08-10. Historical record, 1174 lines, reverse-chronological.
> NOT authoritative. Its bottom half contradicts its top half by construction.
> Scientific claims from this document were reconstructed into
> research/state/claims/ by the 2026-08-10 audit. Cite claim IDs, not this file.
> Retained because the correction chronology is scientifically useful.
```

**1.2** Move the five superseded HANDOFF backups
(`theory/HANDOFF.md.bak_20260617`, `theory/archive/HANDOFF.md.bak*`) into
`research/history/legacy/handoff_backups/`. Two of them contain the boxed
`A = 0.96` result and are currently grep-reachable and indistinguishable from
current material.

**1.3** Move `theory/archive/` to `research/history/legacy/theory_archive/`.

**1.4** Add YAML front-matter to **every** surviving `.md` under `theory/` and
`research/`:

```yaml
---
lifecycle: active | superseded | historical
superseded_by: <path or claim-id or null>
last_reviewed: 2026-08-10
authoritative_for: <topic, or "nothing">
---
```

Proposed initial assignment, from the Stage 1 staleness table:

| file | lifecycle | note |
|---|---|---|
| `theory/AGENTS.md` | superseded | replaced by RESEARCH_CHARTER + instructions |
| `theory/Y_ZETA_DERIVATION.md` | historical | already self-supersedes well. Exemplar. |
| `theory/CURRENT_THEORY_STATUS.md` | historical | banner over-deprecates its own verified §6, §7. Split those into evidence entries first. |
| `theory/NUMERICS_STATUS_AND_PLAN.md` | historical | §1 (why r_c) and §5–7 (run design) are still the best methodology text in the project. Extract to decisions before freezing. |
| `theory/OPEN_ANALYTIC_PROBLEMS.md` | superseded | mine for open claims |
| `theory/VARIANCE_REDUCTION.md` | superseded | §5 retracted; retraction lives only in HANDOFF |
| `theory/CASE_A_IMPLEMENTATION_SPEC.md` | active | still the Cut A reference |
| `theory/theta1_first_principles.md` | superseded | conclusions rest on Δ_ζ = 1 |
| `analysis/anchor_scan.py` | **historical, hazardous** | wrong kernel. Add a refusal header or move to `history/legacy/`. It currently carries no warning. |

**1.5** Rename the nine `pps_qj/*.bak_*` module copies into
`history/legacy/module_snapshots/` so they stop shadowing live modules in grep
and imports.

---

## Phase 2 — Stand up the skeleton

**2.1** Create the `research/` tree from `ARCHITECTURE_PROPOSAL.md` §3.

**2.2** Write `RESEARCH_CHARTER.md` (human-authored, short, rarely changes):
the two planes, the two axes, single-writer, the anti-slop rules, the gates.

**2.3** Write `tools/validate_state.py` **before** seeding any state. A schema
without a validator is a style guide. Minimum viable version:
- YAML parses against the schemas
- every referenced ID resolves
- every `path` with `exists: true` is stat'ed and present
- conditional requirements by `type`
- the `verified` rule: discriminating evidence with acceptable reproducibility
- no two open proposals target one claim ID

**2.4** Write `generate_index.py`, `register_run.py`, `check_dependencies.py`.

**2.5** Write the new `HANDOFF.md`, from scratch, capped at 150 lines: what is
in flight, which tasks are open, pointers. **No scientific values.**

---

## Phase 3 — Seed state from the audit

Order matters. Evidence and observables before claims, because claims reference
them.

**3.1 Observables first.** Define `OBS-` entries for every locator in use:
`B_L` as ⟨CMI·S⟩, ⟨CMI⟩, the KMR product, ΔS_L, `MI_ends_q4`/`q8`, `c_eff`
(marked `deprecated: true` with the reason). Until these exist, no exponent
claim is well-formed.

**3.2 Evidence.** Seed from `02_DATA_INVENTORY.md`, `MISSING_ARTIFACTS.md` and
the recovered `/tmp` manifest. Every entry gets `reproducibility` and
`metadata_gaps`. Expect a large number of `chat_only` and `artifact_only`
entries. That is the honest starting picture, not a failure.

**3.3 Claims.** Port `01_CLAIM_LEDGER.md` and `CLAIM_TIMELINE.md`. Roughly 35
claims. Set `depends_on` deliberately, in particular:
`PPS-YZETA-001 -> PPS-VERTEX-001`, `METH-EXTRAP-001 -> CB-NU-001`,
θ₁ conclusions `-> PPS-VERTEX-001`. Run `check_dependencies.py` immediately and
expect a wave of `stale` flags. Those flags are the audit's findings expressed
mechanically.

**3.4 Disputes.** Create at minimum:
`DISP-PHI-001` (φ ≈ 1/2 vs φ = 1), `DISP-XI-001` (λ⁻¹ vs λ^{−1.5} vs λ⁻²),
`DISP-VERTEX-CHIRAL-001` (chiral vs Ashkin–Teller corner),
`DISP-CASEA-UNIV-001` (Ising for all ζ vs LMR crossover),
`DISP-SNAPSHOT-001` (L=48 against the L-trend).

**3.5 Decisions.** Seed `DEC-MASTER-METRIC` (t_wall·σ²_λc), `DEC-OBSERVABLE`
(⟨CMI⟩ over B_L for exponents), `DEC-CLONING-NOT-THINNING`, and the kill
records for adaptive resampling, full-path IS, multi-ζ reweighting, GLS weights,
trajectory MCMC, and the selection-side sampler programme. Negative results are
canonical state.

**3.6 Sources.** Seed with `inspection_level` honestly set. Most will be
`not_inspected` or `skimmed`. Flag `SRC-CAROLLO` as `attribution_verified: false`
and `SRC-FULGA` with `replica_limit: n->0`.

---

## Phase 4 — Cut over

**4.1** Install `PROPOSED_PROJECT_INSTRUCTIONS.md`.
**4.2** Replace project memory with `PROPOSED_PROJECT_MEMORY.md`. Remove every
mutable numerical conclusion. This is the highest-value single action in the
whole plan: memory is currently the least reliable source in the system.
**4.3** Run one supervised research task end to end, single-agent, to shake out
the workflow before any delegation.

---

## Phase 5 — Multi-agent

Only after Phase 4's supervised task completes cleanly. See the readiness
criteria below.

---

## What is deliberately NOT in this plan

- **Manuscript reconciliation.** Per your instruction, manuscripts are
  downstream artifacts. `MANUSCRIPT_LINEAGE.md` is retained as a record. When
  the Overleaf project is downloaded, reconciling it against claim IDs is a
  separate task.
- **Rewriting history.** No content is deleted or edited for correctness.
  Superseded claims keep `status: superseded`, not deletion.
- **Reconciling the old HANDOFF.** Frozen. The reconstruction already happened.

---

## Minimum viable migration

If the full plan is too much at once, the smallest change set that removes the
worst failure modes, in order:

1. **Phase 0.1** commit the tree. Nothing else matters if the work is lost.
2. **Phase 4.2** replace project memory. It is actively injecting a corrected
   error back into every new conversation.
3. **Phase 1.1 and 1.4** freeze HANDOFF and add lifecycle front-matter. Stops
   archived theory being read as current.
4. **Phase 3.1 and 3.4** observables and disputes. Without observables, exponent
   claims are ill-posed. Without disputes, open disagreements silently resolve
   toward whoever wrote last.

Steps 1 to 4 are perhaps a day of work and capture most of the value. The
registries and validator are the remaining half and are what makes delegation
safe.
