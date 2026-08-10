# Manuscript lineage

Audit 2026-08-10, Stage 2. Resolves the "which manuscript is live" question by
filesystem inspection plus chat provenance, not by mtime alone.

## Four distinct documents exist

| # | document | location | class/format | last modified | status |
|---|---|---|---|---|---|
| 1 | M1 internship report | `~/Downloads/01_M1_Internship/Thesis/m1thesislatex/` | `MastersDoctoralThesis.cls`, 7 chapters | 2026-06-07 18:19 | **completed and submitted** (deadline 2026-06-19) |
| 2 | Continuous-measurement notes | `~/Documents/ppsQJ_m2/continuousmeasurementslatex/` | `article`, `sections/sec1…sec5newnew` | earlier | **source material**, superseded as a deliverable |
| 3 | SciPost article, v1 | `~/Downloads/Partial_Postselection_SciPost_article.tex` | `SciPost.cls`, 4071 lines, 223 KB | **2026-07-22 00:10** | superseded by #4 |
| 4 | SciPost article, restructured | `~/Downloads/Partial_Postselection_SciPost_article_restructured.tex` | `SciPost.cls`, 1984 lines, 89 KB | **2026-07-23 22:54** | **LIVE** |

## The live manuscript

**`~/Downloads/Partial_Postselection_SciPost_article_restructured.tex`.**
Confidence: high.

Evidence:
- Most recent of the SciPost family by 22 hours.
- Self-contained and compilable: has `\documentclass{SciPost}`, an abstract
  section, an embedded `filecontents*` bibliography (`paper_refs.bib`), and a
  full appendix set. It is not a fragment.
- Its section structure is a deliberate **restructure**, not an edit: v1 runs
  Introduction / Continuous measurement / PPS on quantum jumps / Numerical
  methods / Results / **Replica field theory and symmetry classes** / Conclusion.
  The restructured version runs Introduction / Quantum-jump dynamics and
  experimental PPS / **Guided cloning algorithm** / Monitored free-fermion model
  and Gaussian implementation / Results / Discussion and outlook.

**Undocumented scope decision.** The restructure **deletes the entire "Replica
field theory and symmetry classes" section** and promotes the cloning algorithm
to §3. That converts the paper from a theory-plus-numerics article into a
methods-forward one. Given that the field-theory content is exactly the part the
audit finds least secure (`CB-NLSM-001` invalid, x_J route unresolved, chirality
tension open), this may be the right call. It is recorded **nowhere**.

## Provenance of the SciPost draft, and a memory-generation failure

Chat `c49698e2` (2026-07-21) drafted the paper from the SciPost template. That
session ran **without Desktop Commander**, so it could not reach the Mac. Outputs
were written to the container at `/mnt/user-data/outputs/ppsQJ_SciPost_draft.tex`
and `ppsQJ_draft_preview_shimclass.pdf`. The session closed with:

> "Since I cannot touch the repo from here, when you are next in a Desktop
> Commander session I would suggest saving this as `paper/main.tex` …"

That was **a suggestion, never executed**. `git log --all -- 'paper/*'` returns
nothing: `paper/` has never existed in this repository.

Project memory records "SciPost paper draft: `paper/main.tex` in repo".
**Memory promoted Claude's own unexecuted suggestion into a stated fact about
repository state.** This is the cleanest provenance failure in the audit and is
the strongest single argument for the rule that an agent's suggestions must
never be recorded as project state without an execution record.

What actually happened: the draft was downloaded and landed in `~/Downloads`
under a different name on 2026-07-22, then restructured on 07-23.

## Amplitude status per document

| document | statement | verdict |
|---|---|---|
| #1 M1 report, `Chapter3.tex:236` | body text: `A ≈ 0.96`, "prefactor roughly twice the diffusive one" | **wrong**, and the KMR comparison is cross-parameterization. Historical document. |
| #1 `Chapter{5,7}.tex` | outline comments: `A = 0.96 ± 0.05`, `φ = 0.502 ± 0.026` | wrong, comments only |
| #3/#4 SciPost | `\lambda_c(\zeta) \approx 0.50\,\sqrt{\zeta}` | **correct** and matches this audit's 0.494 |

The SciPost abstract is appropriately hedged: the boundary is "empirically
consistent, over the accessible finite-size window, with λ_c(ζ) ≃ 0.5√ζ,
although locator-dependent finite-size drifts prevent a precise exponent
determination", with the reference curve "drawn solid in the Born band ζ ≥ 0.85
where the crossing is reliable and dashed below".

**Therefore memory's "urgent amplitude conflict in the manuscript" is stale.**
It described the 2026-07-21 draft's open TODO, which was closed in favour of
0.50 by 07-22. The only live 0.96 is in a submitted M1 report.

## Claims in the live draft that the audit flags

1. **Abstract, Cut A**: "channel-exchange self-duality pins the transition to
   λ_c = 1/2 for all ζ". Chat evidence validates this only at **ζ ≤ 0.25**
   (crossings 0.49–0.51), with drift to ~0.42 at higher ζ attributed to
   incomplete cells. The `for all ζ` is a theoretical claim presented without
   the numerical caveat. `[P]`, needs the caveat or the completed cells.
2. **Abstract, Cut B**: √ζ consistency. The x_J route (`CHAT_ARCHAEOLOGY.md` §4)
   independently measured x_J ≈ 1.04, which maps to **linear**. Unrefuted and
   uncited.
3. **Bibliography**: the Carollo anchor for λ_c(1) was flagged in the 2026-07-21
   session as unverified and is separately known mis-attributed since
   2026-06-06. Stage 4 item.
4. **Sampling**: the 07-21 session flagged that if any production data came from
   thinned MC rather than cloning, the methods sections are wrong. HANDOFF's
   thinning prototype has an **open bug** and was never shipped, so production is
   cloning, but this should be stated explicitly rather than assumed.

## Preservation risk

The live manuscript is a **loose file in `~/Downloads`**, outside the repository,
outside `01_M1_Internship`, and not in git. Six figure files are placeholders.
