# Charter reconciliation, 2026-08-10

Record of the diff between the **original Research Operating Charter** (authored
by the human researcher, supplied 2026-08-10) and the **reconstructed charter**
that Phase 4B wrote from the Stage 3 architecture.

**The original has authority.** The reconstruction was authored in good faith
after a search of the repository and project knowledge failed to locate the
original, and it said so in its own provenance note. That does not excuse the
result: it substituted architecture for epistemology.

---

## 1. Content in the original that was MISSING from the reconstruction

Almost all of it. Roughly 85 percent of the original by substance had no
counterpart.

| original | status in reconstruction |
|---|---|
| §1 Mission: the do-not-maximize / instead-maximize lists | **absent** |
| §1 "the scarce resource is the researcher's attention"; default is to REDUCE the space of directions | **absent** |
| §1 "never optimize for paperability" | **absent** |
| §2 The nine-level epistemic priority order (truth → significance → validity → discriminating novelty → falsifiability → reproducibility → feasibility → exposition → speed) | **absent** |
| §2 Evidence / Inference / Conjecture / Judgment distinction | **absent** (a different taxonomy was substituted, see §2 below) |
| §3 Human Authority: the six reserved decisions | **partially present** as approval gates, but the substance was missing |
| §3 "may not declare a contribution novel merely because automated searches found no objection" | **absent** |
| §3 Missing-information rule: explicit branches, or ask only when it materially changes the conclusion | **absent** |
| §4.1 No fabricated support, and the full prohibited list | **weakly present** |
| §4.1 "a bibliographic match on title or snippet is not sufficient evidence" | present only inside the evidence schema, not in the charter |
| §4.2 No novelty by vocabulary, and the seven alternative-search channels | **absent** |
| §4.3 No premature manuscript production, and its five prerequisites | **absent entirely** |
| §4.4 Preserve negative results, and the five criteria for value | partially present as "negative results are canonical" |
| §5 Meaningful-Contribution Test, dimensions A–H | **absent entirely** |
| §5 "do not collapse them into a single aggregate score" | **absent** |
| §6 Automatic Slop Warnings, all twelve items | **absent** (a different, project-specific list was substituted) |
| §6 "do not discard such directions silently" | **absent** |
| §7 Mandatory Research Cycle, Stages 0–9 and their named artifacts | **absent** (a different workflow was substituted) |
| §8 Silo-Breaking Protocol and BRIDGE_AUDIT.md | **absent entirely** |
| §9 Open-Source and Dependency-Centered Research | **absent entirely** |
| §10 Theory-Specific Requirements, all ten items | **absent entirely** |
| §11 Communications / information-theory audit | **absent entirely** |
| §12 Research Status Reporting, and "do not report activity as progress" | **absent** |
| §13 Completion Standard, all eleven conditions | **absent entirely** |

## 2. Content MATERIALLY CHANGED in meaning

**2.1 Statement-type taxonomy replaced by a status taxonomy.**
The original requires distinguishing **Evidence / Inference / Conjecture /
Judgment** — a classification of *what kind of statement* is being made. The
reconstruction substituted **verified / plausible / open / contested /
superseded / refuted** — a classification of *how well supported* a claim is.
These are orthogonal, not alternatives. The original's rule "do not present an
inference as evidence or a judgment as a fact" has no counterpart in the status
enum and was lost.

**2.2 Claim-ledger status vocabulary silently renamed.**
Original §7 Stage 7: `unsupported, provisional, supported, contradicted,
withdrawn`. Reconstruction and the implemented schema: `verified, plausible,
open, contested, superseded, refuted`. A mapping is given in Appendix B of the
charter. This is a genuine conflict, not a synonym set.

**2.3 `Confidence` was deleted from the claim record.**
Original §7 Stage 7 requires `Confidence` as a recorded field. Stage 3 removed
it deliberately, arguing it duplicated `status`. **The original has authority and
that removal was not mine to make.** Flagged as an outstanding schema action.

**2.4 The research cycle was replaced rather than implemented.**
The original mandates Stages 0–9 producing `SOURCE_REGISTER.md`,
`PROBLEM_MEMO.md`, `FIELD_MAP.md`, `dependency_graph.json`,
`NOVELTY_MATRIX.md`, `FALSIFICATION_PLAN.md`, `EXECPLAN.md`,
`EXPERIMENT_SPEC.md`, `CLAIM_LEDGER.md`, `RED_TEAM_REPORT.md`,
`RESEARCH_MEMO.md`. The reconstruction described a different pipeline with
different artifacts and did not reference the originals.

**2.5 "Adversarial review" narrowed.**
Original Stage 8 lists nine specific attacks the review must attempt (already
solved elsewhere, follows trivially from assumptions, disadvantaged baseline,
gain from extra information, theorem fails under dependence, proxy measurement,
disappears under realistic conditions, negligible, simpler explanation). The
reconstruction's red-team checklist was project-specific and omitted all nine.

## 3. Architecture-specific ADDITIONS not present in the original

These are useful and are retained, but **only as clearly labelled implementation
extensions**, not as charter provisions:

- the knowledge-plane / execution-plane split
- the authority ordering over `state/`, charter, HANDOFF
- the reproducibility axis and its enumerated states
- the coupling rule tying `verified` to reproducible discriminating evidence
- `depends_on` and the staleness cascade
- `parameterization`, `observable_id`, `fitting_window`, `window_sensitivity`
  well-formedness requirements
- the `postdiction` evidence role
- `contests` / `dispute_id` and the disputes registry
- the T0–T4 compute tiers and the `EXP-ID` authorisation token
- the project-specific anti-slop list
- single-writer state and `validate_state.py`

## 4. Conflicts requiring resolution

| conflict | resolution applied |
|---|---|
| Artifact naming: `CLAIM_LEDGER.md` / `SOURCE_REGISTER.md` versus `state/claims/*.yaml` and `state/sources/*.yaml` | The registries are declared the **machine-readable implementation** of the mandated artifacts, satisfying the requirement. Stated explicitly in Appendix A so the substitution is visible rather than silent. |
| Status vocabulary (2.2) | Mapping table in Appendix B. The original vocabulary is recorded as authoritative intent. |
| `Confidence` field (2.3) | Recorded as an **outstanding schema action**. Not applied in this task, which was scoped to the charter only. |
| Original §11 (communications / information theory) applicability | Retained verbatim in substance, with an applicability note. It does not currently bind this project, but the charter is the researcher's and its scope is not mine to trim. |
| Original §7 Stage 0 requires `SOURCE_REGISTER.md` before implementation | `state/sources/` is currently **empty**. The project is therefore, by the original charter's own standard, not cleared to begin implementation. Escalated in the charter's status note. |

## 5. Assessment

The reconstruction was a competent architecture document and a poor charter. It
encoded how information moves through the repository and almost nothing about
how to judge whether research is worth doing. The original's centre of gravity is
the Meaningful-Contribution Test and the Slop Warnings, which is exactly the
material a system built to prevent low-value output most needs, and it was the
material most completely absent.
