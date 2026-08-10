# SOURCE_REGISTER — TASK-2026-08-10-AMP096         (Charter Stage 0, task-scoped)

Only sources load-bearing for THIS question. Global registry inspection was not
attempted and is not required.

## Load-bearing sources

| SRC-ID | inspection before | inspection achieved | what was actually read | supports_exactly | does NOT establish |
|---|---|---|---|---|---|
| `SRC-KMR-2023` | `relevant_sections` (Appendix B.1 title only) | **full text** | all 1758 lines of SciPost Phys. 14, 031 | B_L = product of trajectory-AVERAGED quantities, confirmed in the body (Sec. 3.2) not only the appendix title; nu = 1.6(7) for the w=0 measurement-only cut; alpha_c = gamma by duality | **No** statement of lambda_c ~ A r_c^{1/2}; **no** occurrence of "diffusiv" anywhere; **no** numerical value for the gamma=0 critical alpha/w |
| `SRC-FULGA-2012` | `not_inspected`, arXiv id UNKNOWN, title recorded as a removed guess | **inspected** | arXiv:1205.1441, 9 pages | identified as Fulga/Akhmerov/Tworzydlo/Beri/Beenakker, PRB 86 054505; nu = 2.06 [1.89,2.20] and 1.93 [1.78,2.24], Table I | nothing about measurement, monitoring, replicas or Born rule — it is a 2D disordered-network localization exponent |
| `SRC-JIAN-2023` | `relevant_sections` | re-read for this point | Jian et al. explicitly equate the forced-measurement (n→0) DIII transition with Fulga's thermal metal-insulator transition (their Ref. 53) | `DEC-CITATION-001` item 3, now verified from both sides | their Born (2.1±0.1) and forced (1.9±0.1) nu are statistically compatible; the class separation rests on x(1), x(2), zeta_1 — not on nu |
| `SRC-LMR-2025` | `abstract_only` | numbers verified | the LMR values previously marked UNVERIFIED | linear scaling of critical unitary strength in (1−zeta); Möbius variable zeta/(1−zeta) | the zeta convention itself (zeta=1 = Born endpoint) is still taken from project notes, NOT verified against the paper |

## Sources that could not be inspected

| SRC-ID | why | blocks a defensible decision? |
|---|---|---|
| the source of lambda_c(1) = 1/2 | **not found**: 182-PDF local library searched, Desktop dir searched, four web queries under alternative terminology. `DEC-CITATION-001`'s standing task remains open. | **No** for this question — the endpoint value is used here only as a numerical cross-check, and the numerics agent measured lambda_c(1) = 0.43–0.49 directly from data. **Yes** for any claim that cites 1/2 as a literature value. |
| `SRC-CAROLLO-2018`, `SRC-FAVA-2023`, `SRC-POBOIKO-2023` | not opened this pass | No — not load-bearing for the amplitude question. `SRC-FAVA-2023` IS load-bearing for `CB-NLSM-001` and remains uninspected. |

## Search log

Local: `PAPERS_LIBRARY` (182 PDFs, titles extracted and filtered),
`DESKTOP_INTERNSHIP`, `DATA_INTERNSHIP/Papers`. Web: four queries under
alternative terminology for the lambda_c(1) = 1/2 value. **Returned nothing**:
every phrasing of "critical measurement rate one half monitored free fermion
post-selection". Absence of a phrase is not novelty (§4.2) and is not recorded
as one.

## Code and data provenance

**The primary document is on disk and tracked in git**, contradicting the claim
file's `evidence_note`: `analysis/lambda_c_phi_analysis.md`, mtime 2026-05-19,
present at commit `6c9c843`. It fits `lambda_c^inf = A*zeta^phi` explicitly in
the lambda_c parameterization and never forms r_c.
