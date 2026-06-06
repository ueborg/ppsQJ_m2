# Master prompt — PPS-QJ theoretical synthesis for M1 internship report

This is the prompt to paste into a fresh chat session. It is written for an
agent that has access to the project folder at
`/Users/catlover1337/Documents/ppsQJ_m2/` and the thesis draft at
`~/Downloads/m1thesislatex/`. The agent should produce a single, long,
structured response that the user can lift into the report directly.

---

> **⚠️ STALE-CLAIM WARNING (added 2026-06-06).** This prompt was written
> against the older "√ζ is closed" framing, which is now **superseded**. For
> Case B the √ζ derivation is invalid for the projector-jump model: the genuine
> cross vertex is **marginal** (Δ≈2, verified), the no-click length is
> **ξ_nc~λ⁻¹** (not λ⁻²), the "König-Brouwer 2014" reference is **non-existent**,
> and the phase-boundary exponent φ is **unresolved**.
> `theory/CURRENT_THEORY_STATUS.md` is the canonical corrected state and
> **overrides anything in this prompt** where they conflict. The Part 1 bullets
> below have been corrected accordingly; if any residual √ζ-as-result phrasing
> survives, treat √ζ as one empirical possibility only, never a derived result.

## CONTEXT THE AGENT MUST LOAD BEFORE WRITING

Read these files in this order, fully, before composing the response.
**Read these three corrected-state files FIRST — they supersede the older docs
wherever they conflict:**

- `theory/CURRENT_THEORY_STATUS.md` — canonical corrected theory state
- `theory/OPEN_ANALYTIC_PROBLEMS.md` — the analytic attempts and what failed
- `theory/NUMERICS_STATUS_AND_PLAN.md` — current data plan and exponent fits

Then the original context set:

1. `theory/HANDOFF.md` — current project state, what's been done
2. `theory/SUMMARY_2026_05_22.md` — comprehensive theoretical+empirical state
3. `theory/qj_pps_theory_summary.md` — 604-line theory summary (the
   long-form derivations)
4. `theory/qj_pps_final_synthesis.md` — the most compact synthesis
5. `theory/ONE_LOOP_RG.md` — the matched-NLSM derivation explicit chain
6. `theory/sec_matching_revised.tex` and `theory/sec_predictions_revised.tex`
   — current LaTeX sections that may be reused
7. `theory/NLSM_FRAMEWORK.md` — Case A vs Case B structural distinction
8. `theory/COLLABORATOR_RESPONSE.md` and `COLLABORATOR_RESPONSE_2.md`
   — peer commentary that constrains the final picture
9. The two key references (read from the project):
   - KMR (Kells-Meidan-Romito SciPost Phys 14, 031, 2023): the Born-rule
     QSD setup our model is the QJ analogue of
   - LMR (Leung-Meidan-Romito PRX 15, 021020, 2025): the PPS extension
     of KMR in QSD; we extend this to QJ

The current thesis draft at `~/Downloads/m1thesislatex/Chapters/` (seven
chapters, .tex files) is the document the response will be inserted into.
The agent should match its tone and structural conventions.

The supervisor is Dganit Meidan. The internship is at ENS-PSL
(CentraleSupélec/SPMS, Université Paris-Saclay). Deadline: **19 June**.

## THE QUESTION, IN FOUR PARTS

The user wants a comprehensive theoretical synthesis covering:

### Part 1 — Case B: ζ-dependence of the MIPT (α + w = 1, γ = 0)

Trace the theoretical story of how the MIPT moves as ζ interpolates between
the fully-postselected limit (ζ → 0) and the Born-rule endpoint (ζ = 1).
Specifically:

- **Setup.** The 1D Kitaev chain with Hamiltonian H = -w Σ (c†_{j+1} c_j +
  Δ c†_{j+1} c†_j + h.c.) at the topological point μ=0, Δ=w, with single
  Bogoliubov-density measurement L_j = d_j† d_j at rate α, and PPS
  parameter ζ ∈ (0, 1]. Quantum-jump unraveling (not QSD).

- **Replica Keldysh action derivation.** Following Le Gal-Schirò
  (arXiv:2511.22506), derive the replicated action with PPS, showing
  explicitly that ζ enters only the cross-replica vertex (the non-Hermitian
  part is replica-diagonal and ζ-independent). This is the structural
  result the whole story rests on.

- **NLSM target and class.** Class DIII via Altland-Zirnbauer; target
  SO(R) in the replica limit R → 1. State the symmetries that pin this
  class for the QJ-PPS-Case-B model.

- **Matched-NLSM attempt (does NOT close — present as a failed derivation).**
  The earlier story derived λ_c(ζ) ~ √ζ from two inputs that are now both
  contradicted: (i) Δ_ζ^UV = 1 is **wrong** — the genuine normal-ordered cross
  vertex :B₊B₋: has Δ ≈ 2.02 (marginal, y_ζ = 0), verified on L=600 in
  `analysis/cross_vertex_dimension.py`; the apparent "Δ=1" came from the raw
  (non-normal-ordered) correlator picking up a dim-1 single-copy admixture
  because ⟨B⟩ ≠ 0. (ii) ξ ~ λ⁻² is **wrong** — the steady-state no-click length
  is ξ_nc ~ λ⁻¹ (verified, `analysis/noclick_spectrum_probe.py`); λ⁻² is only
  the band/dimerization length, not the entanglement-relevant scale. With the
  correct inputs the matching machinery gives φ = 1 (linear) or no clean power,
  not φ = 1/2. The "König-Brouwer PRB 90, 165140 (2014)" citation used for ν = 2
  is **non-existent** (confirmed hallucinated); the real class-DIII ν ≈ 2 source
  is Fulga et al. PRB 86, 054505 (2012), and that is the **n→0 (forced/Anderson)**
  exponent, which Jian-Shapourian-Bauer-Ludwig (arXiv:2302.09094) show is a
  *different* universality class from the Born-rule (n→1) MIPT — so it is not
  transferable. Conclusion to write: √ζ has **no surviving first-principles
  derivation** for this model; present it as one empirical possibility, not a
  prediction the data confirms.

- **The two endpoints.**
  - ζ → 1: λ_c → 1/2 (Carollo et al. PRA 98, 010103, 2018, analytically).
    This is the Born-rule MIPT critical point.
  - ζ → 0: the post-selected limit is deterministic non-Hermitian dynamics.
    What is the critical point there? Discuss honestly — is ν₀ = 2 (DIII
    multicritical) actually the correct identification, or is it a
    crossover rather than a critical point?

- **What changes as ζ varies.** Describe the picture: λ_c(ζ) moves from 0
  at ζ=0 to 1/2 at ζ=1, continuous and monotonic, recovering the Born endpoint.
  The small-ζ exponent is **unresolved** (see corrected bullet above), so do
  not assert √ζ here. The genuine cross vertex is **marginal** (Δ≈2, y_ζ=0,
  verified) — a result worth foregrounding — so small ζ moves the critical line
  without (at leading order) driving a new universality class; but note the
  many-body ν itself is unresolved and the relevant replica limit (n→0 vs n→1)
  is not pinned, so "ν=2 plateau / class-DIII multicritical" must be stated as a
  symmetry-class expectation, NOT a settled exponent.

- **Where the data sits.** The exponent is observable-dependent and NOT
  settled: free FSS fits give φ ≈ 0.56 when fitting λ_c (which saturates toward
  1/2 and biases the fit low) but φ ≈ 0.7–0.85 when fitting the physical ratio
  r_c = λ_c/(1−λ_c). √ζ overshoots and linear undershoots the r_c data; neither
  Möbius form fits well. The earlier "φ ≈ 0.56, consistent with 1/2" headline is
  largely an artifact of fitting the saturating λ_c. ν is scattered around ~2
  with no detectable drift (Spearman −0.07, p=0.88) — consistent with a plateau
  but unable to confirm one. Report the spread honestly; do not collapse it to
  "φ = 1/2 confirmed."

- **What the data cannot yet decide.** The exact value of φ (current
  L ≤ 128 finite-size bias gives φ_eff ≈ 0.76–0.84, trending toward 0.5
  with L but not converged). The slope at ζ=1 (Möbius prediction 1/8
  vs naive NLSM prediction 1/4) requires L = 192, 256 in the decisive
  ζ ∈ [0.10, 0.30] window — campaign currently running on Habrok.

### Part 2 — Case A verification: PPS does NOT shift the MIPT (α + γ = 1, w = 0)

Verify the prediction that λ_c^A = 1/2 for all ζ ∈ (0, 1].

The argument: Case A has two competing on-site measurements (c-density at
rate γ, d-density at rate α) and zero Hamiltonian. The self-duality
α ↔ γ, c ↔ d is exact at the Born rule. PPS weights only by total click
count N_T = N_c + N_d, not by which channel fired — so ζ^{N_T} respects
the self-duality. Therefore the self-dual line α_c = γ_c (equivalently
λ_c^A = 1/2) is preserved for all ζ.

The agent should write this argument out carefully, identify any
loopholes, and discuss what would falsify the prediction (e.g., a
ζ-dependent shift in λ_c^A that breaks the symmetry — which would
imply some non-trivial mechanism not captured by the self-duality
argument). State Case A status accurately: the Gaussian + exact backends are
implemented and validated against exact Fock space (2026-06-06, hard gates
pass), but the production Binder scan + FSS have NOT been run, so λ_c^A = 1/2
and the Ising class remain numerically unverified (spec at
`theory/CASE_A_IMPLEMENTATION_SPEC.md`).

The universality class of Case A is *class D* with self-duality, giving
Ising universality (ν = 1, c = 1/2). This differs from Case B's class DIII
(whose many-body ν is unresolved — not assumed to be 2; see the corrected
Part 1 bullets). The report should make this distinction explicit and explain
why Case A is theoretically the cleaner test: the self-dual line is pinned by
symmetry alone, while Case B's λ_c(ζ) has no closed first-principles derivation
and its exponent is fixed only empirically (and not yet settled).

### Part 3 — Is PPS-QJ doing anything interesting?

The whole project is positioned as the QJ analogue of LMR (QSD-PPS). The
question is whether QJ-PPS gives genuinely new physics or just reproduces
the QSD picture.

Discuss honestly:

- **What's the same as LMR/KMR.** The Lindbladian is identical (KMR's
  γ=0 edge for Case B). Universal critical exponents (ν, λ_c, φ) are
  unraveling-independent at the Lindbladian level; they only depend on
  the symmetry class and the replica field theory. So the *universal*
  predictions are shared between QSD and QJ versions.

- **What's different.** Unraveling-dependent quantities differ:
  - The no-click effective Hamiltonian. In QSD (LMR/KMR), H_eff is the
    Doob-conditioned non-Hermitian generator and gives ξ_ps ~ λ^{-2} as
    a microscopic localization length. In QJ (our model), the distance-3
    Majorana bond gives an effectively gapless H_eff with no single-particle
    length scale — verified by direct diagonalization (see
    `analysis/bdg_actual_vs_simple.png` discussion).
  - The trajectory distribution and the cloning algorithm's behavior.
    The PPS-tilted measure under QJ is structurally different from
    QSD-PPS.
  - Specifically, LMR's main result is the discontinuous universality
    change at ζ* ≈ 0.28 (Ising for ζ < ζ*, monitored for ζ > ζ*). They
    use random unitaries + two measurements, in QSD. Our model has
    deterministic Hamiltonian + single measurement, in QJ — this is a
    *non-overlapping* point of the parameter space. The discontinuity
    at ζ* is specific to LMR's random-circuit + two-measurement model,
    not a generic PPS feature.

- **What's potentially novel in QJ-PPS Case B.** The verified structural
  results are the headline: under QJ the no-click effective Hamiltonian is
  **gapless** (distance-3 Majorana bond, verified) and the PPS cross vertex is
  **marginal** (Δ≈2, verified) — both genuine differences from the QSD operator
  content of KMR/LMR. On top of that, the data show a continuous, monotonic
  λ_c(ζ) with no sign of a discontinuous universality jump, in contrast to LMR's
  ζ*-separated two-class picture. Frame the contribution as "same Lindbladian,
  different unraveling → different operator content and a continuous boundary,"
  with the exponent as the open quantitative question this raises — NOT as a
  "φ = 1/2 + class-DIII-stability" result, which the data do not establish. Flag
  the replica-limit subtlety (n→0 forced vs n→1 Born are distinct classes in
  DIII per Jian et al.; ζ plausibly interpolates), so a single ζ-independent
  class across the whole range is itself not guaranteed.

- **Practical implications.** PPS in experiments is the question of how
  many shots you can afford to throw away. QJ-PPS with continuous
  ζ-dependence means moderate PPS is enough to shift the transition by
  measurable amounts — quantitative experimental relevance. Whether this
  is "useful" depends on whether the shifted transition reveals features
  that the Born-rule transition hides.

- **Honest verdict.** The agent should state clearly: PPS-QJ Case B is
  meaningfully different from KMR/LMR in that the no-click effective
  dynamics is gapless (QJ-specific) and the PPS cross vertex is marginal, and
  the measured phase boundary is continuous and monotonic with no universality
  jump (unlike LMR's discontinuous ζ* picture) — though the boundary exponent
  itself is unresolved (φ observable-dependent, ≈0.56 on λ_c vs ≈0.7–0.85 on
  r_c). Whether this rises to "interesting" in the PRX/PRL sense depends on the
  slope-at-ζ=1 outcome (currently unresolved) and on Case A's prediction holding
  numerically (not yet tested).

### Part 4 — Feedback: a forward-looking discussion

The user noted feedback was considered but not explored. The report
should have a brief, honest section on this:

- **What "feedback" means in this context.** Either:
  (i) Coherent feedback: apply a unitary conditional on the measurement
      outcome (e.g., flip the sign of a hopping term if a click is
      detected), as in Carollo-Verstraete-Lostaglio-style measurement-
      feedback protocols.
  (ii) Adaptive measurement: change measurement strength α based on the
      current entanglement or click history.

- **What would change in the theory.** The replica Keldysh action's
  cross-replica vertex would gain feedback-dependent terms. The simplest
  case (linear-feedback gain → unitary correction) preserves the class-DIII
  symmetry; nonlinear feedback could break it and drive the system to
  class D or further. The MIPT could become a transition between
  feedback-stabilized and feedback-unstable phases.

- **Why we didn't explore it.** The matched-NLSM framework for Case B
  consumed the available analysis time; feedback adds a third dimension
  to the parameter space (now (α, ζ, feedback-strength)) and a separate
  symmetry analysis. Honest position: feedback is the natural next
  research direction after the current PPS-QJ campaign concludes, and
  is mentioned as future work.

- **Relevant prior literature.** Briefly cite the existing
  measurement-feedback MIPT literature (Buchhold-Diehl-Pollmann, Sang-Hsieh,
  etc.) and explain that feedback under QJ-PPS is unstudied.

## OUTPUT FORMAT

A single long response, ~3000-5000 words, structured as four sections
matching the four Parts above, in LaTeX-friendly prose (e.g., use $...$
inline math, $$...$$ display math, and avoid markdown headers that don't
translate to LaTeX). The agent should:

- For each Part, give a clear theoretical narrative followed by a
  precise statement of what's verified, what's conjectured, and what's
  empirically supported.
- Explicitly distinguish three epistemic levels:
  - **Verified analytically + numerically** (e.g., the cross vertex is
    marginal Δ≈2, the action/cross-term structure, ξ_nc~λ⁻¹, λ_c(1) = 1/2)
  - **Predicted analytically, consistent with current data** (e.g., Case A
    self-duality λ_c=1/2; continuous monotonic Case B boundary; symmetry
    classes A=class D, B=class DIII)
  - **Conjectured / unresolved / to be tested** (e.g., the Case B exponent φ
    and many-body ν, the n→0-vs-n→1 replica limit, the slope at ζ=1, Case A
    numerics, feedback effects)
- Include the relevant citations explicitly using \cite{key} format.
  Keys to use: KMR2023, LMR2025, LeGalSchiro2025, Carollo2018PRA, and for the
  class / replica-limit discussion Fulga2012, Jian2023, FavaNahum2023,
  PoboikoMirlin2023. **Do NOT cite "KonigBrouwer2014" — it is non-existent
  (confirmed hallucinated).** These are in
  `~/Downloads/continuousmeasurements(2)/references.bib`.
- Avoid restating what's already in `theory/sec_matching_revised.tex`
  verbatim; instead synthesize and contextualize for the report.
- End with a section "Open questions and future directions" that lists
  the slope test, Case A implementation, Δ_ζ^IR, the crossover function
  derivation, and feedback as the natural next steps.

## NON-GOALS

- The agent should NOT re-derive everything from scratch — read the
  existing documents and synthesize.
- The agent should NOT propose new numerical campaigns; the data is
  already being collected.
- The agent should NOT be overly hedging; the user wants confident,
  precise statements at each epistemic level.
- The agent should NOT write LaTeX boilerplate (preamble, document
  environment) — just the body content suitable for direct insertion
  into the existing chapter structure.

## VERIFICATION CHECKLIST BEFORE THE AGENT RESPONDS

Before composing the response, the agent should verify it has:
- [ ] Read the three corrected-state files FIRST (CURRENT_THEORY_STATUS,
      OPEN_ANALYTIC_PROBLEMS, NUMERICS_STATUS_AND_PLAN), then the original
      numbered files in the CONTEXT section
- [ ] Confirmed it is NOT asserting √ζ as a derived result, NOT citing
      KonigBrouwer2014, and NOT stating ν=2 or φ=1/2 as settled
- [ ] Identified which thesis chapter (1-7) each Part belongs to
- [ ] Noted the verified-vs-conjectured-vs-data status of each claim
- [ ] Checked references.bib for the exact citation keys

If any file is missing or the project state has changed (e.g., new data
arrived from Habrok), the agent should state this at the start of the
response rather than proceeding with stale information.
