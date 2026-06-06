# Open Analytic Problems — QJ-PPS Case B

**Last update: 2026-06-06.** Companion to `CURRENT_THEORY_STATUS.md`.
This document lists the analytic questions that remain open, and records the
analytic routes attempted for the phase-boundary scaling (Section B), so the
failed attempts are not repeated.

---

## A. Problem table

| Problem | Why it matters | Current status | Possible route | Difficulty |
|---|---|---|---|---|
| IR scaling field controlling λ_c(ζ) | Determines the boundary exponent φ | Single-copy mass (Δ=1, relevant) and cross vertex (Δ=2, marginal) identified at the UV; IR fate unknown | One-loop class-DIII NLSM in the joint (λ,ζ) plane; track which operator stays relevant | Hard |
| Many-body ν (=1 vs 2 vs intermediate) | φ = y_λ = 1/ν once y_ζ=1 is fixed; this *is* the √ζ-vs-linear question | Single-particle ξ_nc~λ⁻¹ ⟹ ν=1; literature/old-FSS ⟹ ν=2; r_c data ⟹ φ≈0.8 ⟹ ν≈1.25 | Clean L→∞ FSS on r_c (see NUMERICS); independent IR ν computation | Hard (theory) / Medium (numerics) |
| Marginal cross vertex in the sigma model | Decides whether c/stiffness drift, and whether logs appear | One-loop β_g ∝ (2n−2) → 0 at n→1 [PLAUSIBLE]; IR not done | Two-loop, or map to known class-DIII NLSM β-function in replica limit | Hard |
| Log corrections to r_c(ζ) | A marginal coupling generically gives r_c ~ ζ^φ|log ζ|^p; would explain window-dependent φ_eff | Fit form C gives p≈0.35–0.47 [rough]; but data drift is dominated by the finite-size floor, not cleanly by logs | Derive p from the marginal β-function; test on floor-free data | Hard |
| Derive/refute r_c~√ζ, r_c~ζ, r_c~ζ\|logζ\|^p | These are the candidate boundaries | √ζ derivation invalid (UV inputs refuted); linear motivated by ξ~λ⁻¹ but not derived; log form only fit | See Section B | Hard |
| Is λ_c ≈ ζ/(1+ζ) ever derivable? | It is the linear-r_c, endpoint-fixed form | Equivalent to (y_λ=1, y_ζ=1); motivated but **not** derived; empirically undershoots r_c | Requires proving ν=1 for the many-body transition | Hard |
| y_λ = 1/ν at the IR fixed point | The single missing number | Not computed; ξ_nc~λ⁻¹ is a single-particle proxy only | IR NLSM, or high-quality FSS ν | Hard / Medium |
| Does ξ_nc~λ⁻¹ control the entanglement transition? | If yes, ν=1 and φ→linear; if no, ν may be 2 | [UNRESOLVED] — single-particle length ≠ many-body length in general | Compare ξ_nc to the FSS collapse length at matched (λ,ζ) | Medium |
| QJ-PPS vs QSD-PPS clean comparison | LMR sees a discontinuity at ζ*≈0.28; QJ here does not | Attributed to single-measurement class DIII/D (no U(1) BKT phase) + QJ chirality | Side-by-side replica-action comparison | Medium |

---

## B. Phase-boundary scaling: routes attempted (record)

Goal: derive λ_c(ζ) or r_c(ζ) from the corrected content
P_+P_- = ¼(1 + B_+ + B_- + B_+B_-), with B_++B_- relevant (Δ=1), :B_+B_:
marginal (Δ=2), ξ_nc ~ λ⁻¹.

**The crossover-exponent backbone.** For two scaling fields near the
multicritical point (0,0), the boundary is g_λ ~ g_ζ^{y_λ/y_ζ}, i.e.
λ_c ~ ζ^{φ} with **φ = y_λ/y_ζ**. Everything reduces to (y_λ, y_ζ).

1. **Relevant mass + marginal cross-coupling RG.** y_ζ from the *relevant*
   single-copy mass = 1 (its coefficient ∝ (ζ−1), so ζ tunes it). The marginal
   cross vertex contributes no relevant ζ-eigenvalue. ⟹ φ = y_λ. The cross
   vertex feeds the mass via the V×ε→ε OPE (β_m = m(1 ± κg)), shifting y_λ
   slightly. **Derived:** φ = y_λ. **Not derived:** y_λ itself.

2. **Sigma-model stiffness / Goldstone.** The log phase is the class-DIII NLSM
   Goldstone phase, *not* the free-Dirac point. A relevant single-copy mass
   around free Dirac would localize for any λ>0, ζ<1, falsely giving λ_c=0; the
   data's finite λ_c at ζ<1 is the Goldstone phase, which the weak-coupling
   expansion does not contain. **Conclusion: the transition is non-perturbative;
   weak coupling around free Dirac cannot fix λ_c(ζ).** This is why all the
   naive balances below fail.

3. **BKT-like separatrix.** A marginal coupling generically gives a BKT length
   ξ ~ exp(c/g) and multiplicative logs r_c ~ ζ^φ|log ζ|^p. But the bare cross
   coupling g ~ λζ *vanishes* where the log phase lives (small λ), so it cannot
   be the stabilizer; the marginal logs are a sub-leading dressing, not the
   leading mechanism. **Plausible:** logs exist. **Not derived:** p.

4. **Crossover scaling with ξ_nc.** Transition at ζ·ξ(λ_c) ~ const with a
   *relevant* ζ-coupling (y_ζ=1): with ξ ~ λ⁻¹ ⟹ ζ/r ~ const ⟹ **r_c ~ ζ
   (linear, φ=1)**; with ξ ~ λ⁻² ⟹ ζ/r² ~ const ⟹ **r_c ~ √ζ (φ=1/2)**. So the
   boundary exponent directly probes the many-body length: φ = 1 ⟺ ξ~λ⁻¹,
   φ = 1/2 ⟺ ξ~λ⁻². The single-particle result ξ_nc~λ⁻¹ argues for φ=1, but the
   *many-body* length is the unresolved quantity. **This is the cleanest framing
   of the open question.**

5. **Physical-ratio scaling r = α/w.** If criticality is set by r reaching a
   ζ-dependent threshold r_c(ζ), and if that threshold were linear (r_c ~ ζ),
   then λ_c = Aζ/(1+Aζ), endpoint-fixed to ζ/(1+ζ). This is route 4 with ξ~λ⁻¹.
   Motivated, **not derived**, and empirically the data undershoots it (φ≈0.8<1).

6. **Endpoint constraint.** λ_c(1)=1/2 (Carollo) fixes the prefactor of any
   one-parameter form (A=1 in both √ζ/(1+√ζ) and ζ/(1+ζ)) but cannot fix the
   exponent.

7. **ζ/(1+ζ) status.** = linear r_c, route 5; corresponds to (y_λ=1, y_ζ=1).
   **Motivated interpolation, not a derivation.**

8. **Log-corrected r_c ~ ζ|log ζ|^p.** Follows *if* the marginal cross vertex is
   marginally relevant (sign of β_g at the relevant replica index). Interpolates
   between linear and √ζ for p>0; fits the data better than either pure power.
   **Plausible from marginality; p not derived; and the observed φ_eff window-
   drift is more likely the finite-size floor than a clean log.**

**Net analytic position.** Two defensible scenarios bracket the data:
- LINEAR: (y_λ=1 from ξ_nc~λ⁻¹, y_ζ=1) ⟹ φ=1, r_c~ζ, λ_c~ζ/(1+ζ). Undershoots slightly.
- SQRT: (y_λ=1/2, ν=2, class-DIII, y_ζ=1) ⟹ φ=1/2, r_c~√ζ. Overshoots.
The data (φ≈0.8 on r_c) lies between, closer to LINEAR, plausibly LINEAR dressed
by marginal logs. **No clean power law is derivable at present.** The decisive
analytic input is y_λ = 1/ν at the actual IR fixed point.


---

## C. Analytic attempts after corrected operator content (2026-06-03)

Actual derivation attempts, not a list. Setup: free critical hopping (c=1) UV
fixed point; two perturbations — relevant single-copy mass m ∝ λ(1−ζ) (the
energy operator B, Δ=1) and marginal cross vertex g ∝ λζ (:B₊B₋:, Δ=2). The
boundary exponent obeys φ = y_λ / y_ζ.

### Attempt 1 — relevant mass + marginal coupling RG
**Assumptions.** Free-Dirac UV; OPE V×ε→ε (coeff b) and V×V→V (coeff a).
**Equations.**
$$\frac{dg}{d\ell} = a\,g^2,\quad a = -A(2n-2),\ A>0\ (\text{Ising }\epsilon\times\epsilon=1);\qquad
\frac{dm}{d\ell} = (1 + b\,g)\,m,\ b>0.$$
**Result.** At n→1, a→0 ⟹ g(ℓ)=g₀ (frozen); m(ℓ)=m₀e^{(1+bg₀)ℓ}, relevant for
any g₀. ξ_m ~ m₀^{−1/(1+bg₀)}.
**Why it fails as a boundary.** m is relevant for any m₀≠0 ⟹ localization for
any λ>0, ζ<1 ⟹ predicts λ_c=0 for ζ<1, contradicting data (λ_c>0).
**Obstruction (located precisely).** A finite λ_c requires the log phase to be a
stable phase. Weak coupling around free Dirac does not contain it — the relevant
mass always wins. The log phase is the class-DIII NLSM **Goldstone phase**, a
separate fixed point not perturbatively connected to free Dirac; equivalently
the n→1 replica limit is singular and the single-copy mass that looks relevant
does not localize the *entanglement* (it gaps single-particle modes, but the
entanglement transition is governed by the inter-replica coupling).
**Predicts.** Nothing (degenerate); only confirms the obstruction.

### Attempt 2 — marginal-log calculation
**Assumptions.** Keep a<0 (g marginally irrelevant at finite n); y_m(g)=1+bg.
**Equations.**
$$g(\ell)=\frac{g_0}{1+|a|g_0\ell},\qquad
m(\ell)=m_0\,e^{\ell}\,(1+|a|g_0\ell)^{\,b/|a|},\qquad
\xi \sim m_0^{-1}\,|\log m_0|^{-b/|a|}.$$
With m₀ ∝ λ(1−ζ): the localization length carries a **multiplicative log**.
**Result / prediction.** The marginal vertex dresses any power law with a log:
$$\boxed{\,r_c \sim \zeta\,|\log\zeta|^{p}\,},\qquad p>0\ \text{if}\ b>0.$$
Local effective exponent φ_eff = 1 − p/|log ζ| → 1 as ζ→0. Since b>0 (V enhances
the mass via the ε×ε=1 OPE), the log enhances localization at small ζ, bending
the *leading-linear* law down to an effective 0.7–0.85 over the measured window.
**Falsification.** Clean (L-extrapolated) data with φ_eff **rising toward 1** as
ζ→0 supports this; φ_eff flat at 1/2 refutes it (favors √ζ/ν=2). The exponent p
is **not** derived (it is b/|a|, set by IR data).

### Attempt 3 — sigma-model stiffness
**Assumptions.** Class-DIII NLSM; λ ↔ mass/inverse-stiffness, ζ ↔ cross-replica
vertex.
**Result.** At n→1 the cross-vertex β-function vanishes (Attempt 1), so the
marginal vertex does **not** renormalize the NLSM stiffness at one loop; ζ enters
the stiffness only at higher order or through the mass. Whether it stays
marginal (Δ_ζ^IR=2 ⟹ logs) or becomes relevant (Δ_ζ^IR=1 ⟹ √ζ re-emerges) is
the open IR question. **Does not independently fix φ;** consistent with Attempt 2.

### Attempt 4 — no-click length vs many-body length (decisive framing)
**Assumptions.** The many-body entanglement-transition length tracks the no-click
localization length (as established for QSD free-fermion MIPTs, where ξ~λ^{−2}
gives ν=2). Transition at ζ·ξ(λ_c) ~ const.
**Equations.**
$$\xi\sim\lambda^{-1}\ (\text{verified, QJ}) \Rightarrow \zeta/r_c\sim\text{const}\Rightarrow r_c\sim\zeta\ (\phi=1);$$
$$\xi\sim\lambda^{-2}\ (\text{QSD band length, NOT this model}) \Rightarrow \zeta/r_c^2\sim\text{const}\Rightarrow r_c\sim\sqrt\zeta\ (\phi=\tfrac12).$$
**Result.** φ=1 iff the many-body length equals the verified ξ_nc~λ^{−1}. **The
QJ steady-state length being λ^{−1} (not λ^{−2}) is exactly why QJ-PPS should
give linear, not √ζ** — the √ζ was inherited from the QSD λ^{−2} length, which
this model does not have.
**Why it might still fail.** The many-body FSS length need not equal the
single-particle no-click length; if a parametrically longer λ^{−2} length
controls the transition (e.g. generated by the marginal vertex), √ζ survives.
**Falsification.** Compare ξ_nc(λ) (from `noclick_spectrum_probe.py`) to the FSS
collapse length at matched (λ,ζ): equal ⟹ ν=1 (linear); parametrically larger
⟹ ν=2 (√ζ).

### Attempt 5 — physical-ratio / rate balance
**Assumptions.** Transition set by α/w=r at a ζ-threshold; rates entangle ~w,
no-click localize ~α(1−ζ), click measure ~αζ.
**Result.** Every naive balance (m>w; ξ_nc~ξ_g; etc.) gives a vertical line or
the wrong sign, because the relevant mass localizes at any value (Attempt 1).
Rate balance confirms **r is the natural variable** but does not derive f(ζ).

### Attempt 6 — endpoint constraint
λ_c(1)=1/2 ⟹ r_c(1)=1 fixes the prefactor A=1 in any one-parameter form. It
**cannot fix φ**. Useful only to pin the prefactor once the exponent is set
elsewhere.

### Attempt 7 — candidate-form verdict
- **r_c=√ζ:** needs ξ~λ^{−2} (refuted) or Δ_ζ^IR=1 (refuted at UV). NOT derived; overshoots data.
- **r_c=ζ:** = (y_λ=1, y_ζ=1), i.e. many-body length = ξ_nc~λ^{−1}. MOTIVATED (most natural given verified no-click length), not rigorously derived; undershoots slightly.
- **r_c=ζ|log ζ|^p:** r_c~ζ dressed by the marginal vertex (Attempt 2). Best-motivated; p not derived; fits data best.
- **λ_c=√ζ/(1+√ζ):** Padé of √ζ; not derived; overshoots.
- **λ_c=ζ/(1+ζ):** Padé of linear r_c; motivated, not derived; undershoots slightly.

### Attempt 8 — strongest defensible prediction
**r_c ~ ζ|log ζ|^p with p>0** — equivalently an effective exponent in (1/2, 1)
that drifts toward 1 as ζ→0 (up to logs). Justification: (i) the verified
ξ_nc~λ^{−1} gives leading **linear** (φ=1) via the same length-tracking logic
that gives QSD its √ζ from λ^{−2}; (ii) the verified **marginal** cross vertex
generically multiplies this by a log, bending the effective exponent below 1
toward the observed 0.7–0.85; (iii) **NOT √ζ**, which requires the λ^{−2} length
this model does not have. Honest caveats: the leading φ=1 assumes the many-body
length tracks the single-particle no-click length (true for QSD, unproven here),
and p is not derived. **The single missing rigorous input is y_λ=1/ν at the IR
fixed point** — attackable by a one-loop class-DIII NLSM in the joint (λ,ζ)
plane, or empirically by the L-extrapolated r_c fit.


---

## D. Replica field-theory routes explored (2026-06-06)

Three routes attacked the *universality* of the von Neumann (n→1) boundary,
using the corrected operator content (Δ_B = 1 relevant single-copy mass,
Δ_{:B₊B₋:} = 2 marginal cross vertex). All three converge on ONE obstruction;
none yields a closed-form ν or φ. Recorded so they are not repeated.

### D1 — Two-loop PCM-SO(N) β-function [done; inconclusive]
β(t) = (C₂/8π)t² + (C₂²/128π²)t³, C₂ = N−2 (dual Coxeter of SO(N)). One-loop
∝ C₂, two-loop ∝ C₂² (positive-definite, group-manifold curvature — UNLIKE the
O(N) vector model, whose two-loop is ∝ C₂). At the replica limit C₂<0 the two
terms OPPOSE → a formal fixed point at t* = −16π/C₂ ≈ 50 (N=1), but this is at
STRONG coupling where two loops is invalid. Formal β'(t*) = 2 ⟹ ν = 1/2, but
this is a two-term-truncation artifact (C₂-independent, prefactor-contingent)
and is inconsistent with the data's ν ~ 2. Net: no reliable perturbative fixed
point; if anything a weak hint of a *conventional* (not BKT) point, unreliably.
Caveat surfaced: the Z₂ vortices the BKT picture needs come from π₁(SO(N))=Z₂ at
N≥3, but SO(1) is trivial — the vortex/BKT argument is shaky at the replica
limit. [verified structure / plausible prefactor]

### D2 — Exact integrable PCM mass gap continued in N [explored; BLOCKED; dropped]
Idea: the SO(N) PCM is integrable (N>2), exact TBA mass gap ξ ~ exp(8π/(C₂ t));
continue in N. BLOCKED: (i) the asymptotically-free mass-gap formula assumes
C₂>0 (mass generated flowing weak-UV→strong-IR); at C₂<0 the flow reverses and
the formula gives ξ→0 at weak coupling, opposite of the log phase. (ii) SO(N) at
N→1 is SO(1)={1}, a trivial target — "N→1 PCM" is a formal β-continuation, not a
sigma model. Nuggets kept: N=2 (SO(2)=U(1)=free boson) is the KT organizing
point, and the replica limit is a finite deformation below it; and IF the
continuation were the O(N) VECTOR model (not the PCM), N→1 = O(1) = Ising
(ν=1, c=1/2, closed form) — but that conflicts with the data's ν~2 and Case B is
PCM, not vector. So the candidate field theories do not cleanly predict the
observed ν. Dropped per low yield.

### D3 — LMR-style one-loop K-matching for the QJ distance-3 operator [done; clean negative]
Redo of LMR App-G with the CORRECT operator content (qj_bosonization_calculation.md
used the wrong linear-jump operator). Relative Choi mode φ_σ, Luttinger K_σ,
K₀=1 (free fermions). The cross vertex :B₊B₋: is the cosine, MARGINAL at K₀=1
(Δ=2, verified). One-loop self-coupling β_g ∝ −A(2n−2)g² (the project's verified
(2n−2) factor, A>0 from Ising ε×ε=1): marginally IRRELEVANT for Rényi n≥2,
EXACTLY marginal at n=1. KEY RESULT: the cross vertex does NOT drive a
transition ⟹ there is NO QJ analogue of LMR's ζ*≈0.28. LMR's ζ* needs a RELEVANT
cross vertex (their diffusive QSD measurement); the QJ projector-click vertex is
marginal. This is the sharp QJ-vs-QSD distinction, and it matches the QJ data
(no sharp ζ*). The MIPT is instead driven by the relevant single-copy mass
(Δ=1, ∝ λ(1−ζ)) — non-perturbative (relevant ⟹ perturbative λ_c=0; Attempt 1).
K_σ(ζ) = 1 − O((ζr)²): small non-universal shift, no marginality crossing at
n→1, confirming no von Neumann ζ*. [verified inputs / plausible]

### Convergent conclusion
D1, D2, D3 triangulate ONE obstruction: at n→1 the relevant structures
degenerate (the (2n−2) factor → marginal vertex; the SO(1) trivial target; the
reversed-flow mass gap), so every perturbative or exact-continuation route gives
either nothing or a Rényi-n≥2 feature that washes out at von Neumann. The von
Neumann MIPT boundary is irreducibly non-perturbative and single-copy-mass-
driven; no closed-form ν or φ is extractable by these methods. This is a robust
negative result, not the failure of any single calculation. Remaining
non-perturbative handles: lattice numerics; or the integrable continuation done
at the level of exact correlators rather than the AF mass-gap formula (hard,
separate). Routes 3–4 of the original list (instanton fugacity, solvable limits)
are now judged unlikely to beat this and are deprioritised.


### D4 — Exact-correlator / integrable continuation; the U(1)/BKT criterion (2026-06-06) [done; route exhausted analytically]
Plan: continue exact SO(N) PCM correlators (not the mass-gap formula) in N to the
replica limit. Outcome below; this route is now closed at the analytical level.

TARGET (inherited, not separately derived): principal chiral SO(N), orthogonal
class, monitored Majorana (Fava-Piroli-Swann-Bernard-Nahum, PRX 13 041045). Case B
INHERITS it — the manifold is fixed by the AZ class (Majorana reality → orthogonal)
and the replica/Choi structure; the hopping H, the distance-3 measured bond, and the
PPS weight are all Majorana-bilinear / replica-symmetric, so they set the COUPLINGS
(bare stiffness t₀(λ), cross-vertex g₀∝ζr), not the target. ζ^{N_clicks} is
SO(N)-invariant ⟹ tuning ζ moves a coupling inside the PCM, does not reduce the coset.
[symmetry-level argument, not a line-by-line Choi-action HS derivation]

COULOMB-GAS / U(1) CRITERION: a BKT essential singularity (ν=∞) needs an exactly
marginal operator sustaining a LINE of fixed points = a U(1) / marginal line. In the
O(n)/SO(n) family this lives only at n=2 (SO(2)=U(1)=free boson); the thermal ν(n) is
FINITE for n<2 and diverges only as n→2⁻. The physical replica limit is n→1 (Born),
where SO(1) is trivial — NO U(1), no marginal line. ⟹ BKT DISFAVORED; the boundary is
a CONVENTIONAL power-law transition. This REVISES the early-June (messages 1–3) BKT
lean: the marginal cross vertex that motivated BKT is a SPECTATOR (D3: irrelevant for
n≥2, marginal at n=1), not the transition driver, so it does not produce BKT.

WHY THE ROUTE TERMINATES: at n→1 the PCM target degenerates and there is no standard
exact-correlator continuation of the SO(N) PCM to n<2 to invoke (the O(n) VECTOR/loop
model is solved for −2≤n≤2 but is a DIFFERENT model). The transition is the
relevant-mass-driven, non-perturbative novel Born-rule orthogonal class (Jian et al.),
with no closed-form exponent. The route settles the QUALITATIVE question (conventional,
not BKT) but not the exponent — that is now a numerical question.

ENDPOINT REFRAMING (useful for the data): the ζ-line interpolates FORCED (ζ→0, n→0,
the Anderson/Fulga class, ν≈2) and BORN (ζ=1, n→1, novel class, ν unknown). So the
measured "plateau ν~2" at small ζ likely reflects proximity to the FORCED (n→0)
endpoint, NOT the Born exponent; ν need not be constant along the line. NUMERICS: fit
ξ ~ |t|^{−ν} (conventional), NOT the BKT essential singularity; downgrade "ν=3.1 at
ζ=0.02 = BKT" to large-ν / forced-endpoint + finite-size.

### Updated convergent conclusion (supersedes the §D conclusion above)
D1–D4 agree: the von Neumann (n→1) boundary is non-perturbative and
relevant-single-copy-mass-driven, in the novel Born-rule orthogonal class, with NO
closed-form ν by any of these methods. BKT is now DISFAVORED (not the leading scenario
it was in early-June notes): it is the n=2 feature, the Majorana Z₂ class lacks the
requisite U(1), and the marginal cross vertex is a spectator. The boundary is most
likely a CONVENTIONAL power-law transition with a large, possibly
forced-endpoint-contaminated, ν. The analytical program for the boundary universality
is exhausted; the exponent is a numerical question (fit power-law, not essential
singularity). Remaining theory work would be a fully rigorous Choi-action HS derivation
to confirm the SO(N) target and rule out a reduced coset — confirmatory, not expected
to yield an exponent.


### D5 — CORRECTION to D4: the transition CHARACTER is ζ-dependent (2026-06-06)
D4's "conventional favored over BKT" was OVER-STATED; this supersedes it. Pushing on
WHERE the relevant operator goes: the single-copy mass (B₊+B₋, Δ=1) has coefficient
∝ α(ζ−1), so it VANISHES EXACTLY at ζ=1 (Born) [verified, CURRENT_THEORY_STATUS].
Consequences:

- ζ=1 (Born): the relevant mass is absent; the ONLY nontrivial replica operator is the
  MARGINAL cross vertex (Δ=2), carrying the standard tree-level KT flow
  dg/dℓ = 2(1−K_σ)g (the (2n−2) self-coupling is subleading and vanishes at n=1). A
  marginal operator tuned across marginality by λ is a KT transition. This is the
  Poboiko-Mirlin scenario (monitored free fermions: log phase as a marginal/critical
  regime ending at a KT-type point). ⟹ at the Born point BKT is NOT disfavored; it is
  arguably the natural expectation. (D4 had this backwards.)
- ζ<1: PPS switches the relevant single-copy mass back ON (coeff ∝ λ(1−ζ)). A relevant
  operator ⟹ CONVENTIONAL power-law transition.

NET (supersedes D4): the transition CHARACTER is ζ-dependent — KT-like at ζ=1,
conventional for ζ<1 — controlled by the single-copy mass that vanishes at ζ=1. PPS
CONVERTS the marginal/KT Born transition into a conventional one. [PLAUSIBLE; from
verified operator content + PM scenario; not proven] TESTABLE: ξ-divergence should be a
power law ξ~|t|^{−ν} for ζ<1 and cross to an essential singularity
log ξ ~ (λ_c−λ)^{−1/2} as ζ→1. NEEDS: (i) the actual monitored-Majorana Born (ζ=1)
universality from the literature / Carollo follow-ups; (ii) confirmation the cross vertex
bosonizes to a relative-field cosine with this KT flow; (iii) completeness of the
two-operator content at ζ=1 (no other relevant operator missed).

POST-SELECTED ENDPOINT (ζ→0) — sharpened to a LARGE-DEVIATION question. ζ^{N_clicks}
tilts the click number. Whether ζ→0 selects ZERO click density (→ the single
deterministic no-click state: area-law for λ>0, ξ~λ⁻¹ from the no-click probe, "ν=1") or
NONZERO click density (→ forced/Anderson-like, possibly ν≈2) is the sign of the click
density ρ_click(ζ) = d(SCGF)/d(log ζ) as ζ→0. Resolvable with the project's tilted-
ensemble / Jack-Sollich machinery; the L_ζ / θ₁ spectral work is the entry point.
ρ_click(ζ→0)→0 ⟹ deterministic endpoint; finite ⟹ singular/forced.

LOG-CORRECTION STRUCTURE (closed-form, from the marginal operator): the
marginally-irrelevant cross vertex (n≥2) gives 1/log L corrections to the Rényi-n
entanglement with coefficient ∝ (2n−2); at n=1 it is exactly marginal (one loop) ⟹ a
NON-universal constant shift to the log-coefficient (a marginal modulus), NOT a 1/log L
term. Prediction: S_{n≥2} approaches scaling with 1/log L corrections; S_vN's leading
correction is not of that form. Testable on the existing Rényi-2/3 data.

### Status of the analytic program (2026-06-06, final for now)
No closed-form von Neumann exponent is obtainable by any route tried (D1–D5); the
replica non-analyticity (forced n→0 vs Born n→1 distinct classes) is the obstruction.
The defensible analytic deliverables are STRUCTURAL: (a) the ζ-dependent character
(KT-like at ζ=1, conventional for ζ<1); (b) the large-deviation framing of the ζ→0
endpoint; (c) the (2n−2) log-correction prediction. The exponents are a numerical
question.
