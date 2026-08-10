# Amplitude trace: A = 0.96 vs A ≈ 0.51

Audit 2026-08-10. Non-canonical. No files outside `audit/2026-08-10/` were touched.

## Verdict

`A = 0.96 ± 0.05` is **not** the amplitude of `λ_c = A √ζ`. It originated as a
May-2026 fit that was superseded on 2026-06-07 and formally corrected in
`Y_ZETA_DERIVATION.md` on 2026-06-10. The correct λ_c amplitude is ≈ 0.49–0.51.
The ≈ 0.7–0.9 figure belongs to the `r_c = λ_c/(1−λ_c)` parameterization.

**Current Claude project memory states the correction backwards.** It asserts
A = 0.96 for λ_c and attributes ≈ 0.5 to r_c. That is the inverse of the
established position, and memory presents the inversion *as* the correction.

## Independent reproduction (this audit)

`audit/2026-08-10/scripts/reproduce_amplitude.py`, run against
`~/Downloads/01_M1_Internship/Data/pps_aggregates/agg_caseB_combined.pkl`
(1046 records, guided cloning, L ∈ {32,48,64,96,128,160}, ζ ∈ [0.05, 0.85]).
Wide-pair (L2 ≥ 2·L1) Binder B_L crossings, linear interpolation, median over pairs.

| parameterization | A at fixed φ=1/2 | free power φ | free-power A |
|---|---|---|---|
| `λ_c`             | **0.494** | **0.495** | 0.488 |
| `r_c = λ_c/(1−λ_c)` | 0.692 | **0.681** | 0.874 |

This reproduces the 2026-06-07 conclusion (λ_c A ≈ 0.51, r_c φ ≈ 0.65–0.85)
on the rebuilt guided aggregates, which is an independent dataset from the one
that conclusion was drawn on.

Caveat on status: this is an **audit-grade** reproduction. No bootstrap, no
drift error bars, no correction-to-scaling term, no N_c debiasing. It is
sufficient to adjudicate 0.49 vs 0.96. It is **not** a production measurement
of φ and must not be quoted as one.

## Provenance chain

| date | source | statement |
|---|---|---|
| 2026-05-20 | `theory/archive/SESSION_2026_05_20.md:31` | "A = 0.96 ± 0.05 (not A ≈ 0.5 as previously assumed)" — from a 1/√L extrapolation table giving φ = 0.502 ± 0.026 |
| 2026-05-22 | `theory/archive/NLSM_FRAMEWORK.md:286` | "A = 0.96 ± 0.05 (this session's analysis) is consistent" |
| ~2026-05-24 | `theory/archive/HANDOFF.md.bak:14,73` | boxed result `λ_c(ζ) ~ A√ζ, A = 0.96 ± 0.05` |
| 2026-05-27 | `Chapters/Chapter3.tex:236` | inherits it into manuscript body text |
| 2026-06-07 | `HANDOFF.md:434` (cont.-1 #2) | λ_c = A√ζ gives **A ≈ 0.51** on dense (χ²/dof 0.76) and 0.53 on v2; r_c gives 0.78–0.90 |
| 2026-06-07 | `HANDOFF.md:340` (cont.-2 #1) | debiased (32,64,128) triple: **λ_c = 0.501·√ζ**, φ = 0.523 ± 0.019, R² = 0.986 |
| 2026-06-10 | `Y_ZETA_DERIVATION.md:180` | explicit correction recorded in the source document |
| 2026-08-10 | this audit | λ_c A = 0.494 on the rebuilt guided aggregate |
| current | Claude project memory | reasserts A = 0.96 for λ_c — **regression to the superseded value** |

## Manuscript exposure

Manuscript is **not** at `paper/main.tex`. That path does not exist. Memory is wrong
about the location. The affected document is
`~/Downloads/01_M1_Internship/Thesis/m1thesislatex/`.

| file | line | status | text |
|---|---|---|---|
| `Chapters/Chapter3.tex` | 236 | **body text, compiled into `main.pdf`** | "we find $\lambda_c \sim A \sqrt{\zeta}$ with $A \approx 0.96$, so the square-root exponent persists, but the prefactor is roughly twice the diffusive one" |
| `Chapters/Chapter5.tex` | 8, 46 | LaTeX comment (outline) | "A = 0.96 +/- 0.05"; "sqrt(zeta) ansatz is 28% better" |
| `Chapters/Chapter7.tex` | 13 | LaTeX comment (outline) | "lambda_c ~ 0.96 sqrt(zeta) with phi = 0.502 +/- 0.026" |

`Chapter3.tex` mtime is **2026-05-27**, i.e. eleven days before the correction.
`main.pdf` was last built 2026-06-07 18:19, so the wrong value is in the compiled PDF.

### The secondary error is worse than the number

Chapter 3 line 236 compares our amplitude to KMR and concludes the prefactor is
"roughly twice the diffusive one". The preceding sentence states KMR report
`λ_c ~ A r_c^{1/2}` with A of order one half — that is KMR in an **r_c-type**
parameterization. Comparing our 0.96 against KMR's ~0.5 is therefore a
cross-parameterization comparison. In matched parameterizations the numbers are
λ_c: 0.49 (ours) and r_c: ≈0.87 (ours), so the "factor of two larger than the
diffusive case" claim does not survive once the parameterizations are aligned.

This is a **physics claim in the manuscript narrative**, not only a typo, and it
should be re-derived rather than merely renumbered. Flagged for human decision.

## Related stale memory item

Memory lists as an outstanding urgent action: "amplitude conflict … mislabeling
in §7 of `Y_ZETA_DERIVATION.md`". That fix **landed on 2026-06-10**. The file now
carries both a superseding header banner and an in-line CORRECTED paragraph at
line 180. The memory item is two months stale in addition to being inverted.
