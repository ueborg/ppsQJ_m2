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


### D6 — External review (2026-06-06): withdraw "KT at ζ=1"; finite-ν Born class; ζ→0 order-of-limits; Carollo citation fix
An external LLM review, pressure-tested by web-verification, corrected the following. This
is the STABLE conclusion; the BKT↔conventional↔KT oscillation in D4/D5 was over-reaching
analytics and is settled here.

1. CITATION FIX [VERIFIED by search]. "Carollo et al. PRA 98, 010103 (2018)" is NOT the
   λ_c(1)=1/2 source — it is Carollo-Garrahan-Lesanovsky-Pérez-Espigares, "Making rare
   events typical in Markovian open quantum systems," a quantum-Doob / large-deviation
   paper. λ_c(1)=1/2 should be relabeled NUMERICALLY PINNED (true source TBD — possibly a
   model-specific self-duality). Silver lining: that Doob paper IS the correct citation for
   the PPS / tilted-ensemble framing (ζ^{N_clicks} = a Doob tilt).

2. WITHDRAW "KT at ζ=1" (D5 was overstated). [VERIFIED] Jian-Shapourian-Bauer-Ludwig
   (arXiv:2302.09094) establish the generic 1D monitored-Majorana Born transition as a
   FINITE-ν "novel universality class beyond Anderson localization" (principal chiral SO(n)
   NLSM, n→1, SO(n)×SO(n) symmetry; forced is n→0 = Anderson), Z₂-defect-driven; they
   extract a finite ν (their App. C method). A finite ν is NOT a KT essential singularity.
   Structural points (correct): a marginal UV operator does not imply an IR KT fixed line;
   Poboiko-Mirlin is the U(1) class (and finds a crossover, not KT) — wrong to lean on.
   CORRECTED STATEMENT: ζ=1 is special ONLY because the relevant single-copy mass
   (∝α(ζ−1)) vanishes; its universality is the unsolved, FINITE-ν Born n→1 orthogonal/DIII
   class; ζ<1 turns a relevant mass on and is conventional. ⟹ finite-ν conventional-type
   throughout, NOT KT.
   CAVEAT: Jian is a RANDOM circuit; Case B is a DETERMINISTIC Hamiltonian. Same DIII /
   SO(n)→1 structure argues for the same (finite-ν novel) class, but random-vs-deterministic
   can matter (the PM crossover debate) — plausible, not automatic. No closed-form ν either
   way (Jian's is numerical, reportedly ν_Born≈2.1, ν_forced≈1.9 — UNVERIFIED, read Table I
   of 2302.09094 before citing).

3. ζ→0 ENDPOINT — order of limits (correct, sharper). At fixed finite L,
   ρ_click(ζ)=ζ·θ_{1,L}+O(ζ²)→0; and θ₁^SCGF=0 by fermion parity ⟹ O(ζ²), even more
   strongly →0. A nonzero limiting click density requires the SINGULAR order
   lim_{ζ→0}lim_{L→∞}; lim_{L→∞}lim_{ζ→0}=0. The right question is this order-of-limits /
   s→∞ dynamical-transition test, not "is ρ_click(ζ→0) zero." No closed-form ρ_click(ζ) for
   the interacting Choi problem (recycling term non-quadratic after doubling); θ₀ (no-click)
   is Gaussian-accessible, ∂_ζθ|_0 needs the left/right eigenoperators + recycling insertion.

4. Unraveling-independence: "MIPT exponents are unraveling-independent" is too strong
   (unraveling-induced transitions exist; ref to VERIFY: Eissler-Lesanovsky-Carollo
   arXiv:2406.04869). Strengthens the QJ-vs-QSD distinction.

STABLE CONCLUSION (settle here): NO closed-form exponent; a FINITE-ν, conventional-type
transition (ν≈2.1 FIXED along ζ∈(0,1], the n→1 Born value — see D8, which SUPERSEDES this
"drift 1.9↔2.1" reading: the forced n→0 point is off the PPS line); KT / essential singularity NOT
established and probably wrong; ζ=1 special only via the vanishing single-copy mass; ζ→0 an
order-of-limits question. The exponents are a numerical question. The operator-content audit
(single-copy mass ∝α(ζ−1) relevant + vanishing at ζ=1; :B₊B₋: marginal Δ≈2) was
independently re-confirmed in this review.


### D7 — Literature anchors from Jian (2302.09094) and the Arrow-of-Time paper (2604.20828) (2026-06-06)
Two papers read in full (uploaded PDFs). Neither gives a closed-form Born exponent —
both reinforce "no closed form" — but both are useful.

JIAN ET AL. Table I [VERIFIED from the PDF] — generic 1D Majorana DIII circuit:
  Born (n→1):   ν=2.1±0.1, X₁=1.00±0.02, X_typ=2.66±0.05, x⁽²⁾=−1.80±0.04, ζ₁=0.39±0.02
  Forced (n→0): ν=1.9±0.1, X₁=1.02±0.03, X_typ=3.53±0.04, x⁽²⁾=−5.7±0.2,   ζ₁=0.30±0.04
  (X₁ = avg of squared Majorana correlator G=⟨iγγ⟩², G~R^−2X₁; X_typ, x⁽²⁾ = 1st & 2nd
   cumulants of log G vs log chord-distance; ζ₁ = S(L/2)~ζ₁ log L.)

KEY POINTS:
1. ν does NOT discriminate Born vs forced (Δ≈0.2, within error) — the paper says so
   explicitly. ⟹ the project's flat ν~2 and UNMEASURABLE drift (Spearman −0.07) are
   EXPECTED, not a failure. [SUPERSEDED by D8: the reason is that the whole PPS line stays in
   n→1 (ν≈2.1 fixed), NOT that "both endpoints are ≈2" — the forced n→0 point (1.9) is off the
   line.] State this with the citation; stop apologizing for the flat ν.
2. The DISCRIMINATORS are the MULTIFRACTAL exponents X_typ (2.66 vs 3.53) and x⁽²⁾
   (1.80 vs 5.7) of the squared Majorana correlator. NUMERICS PIVOT: compute the 1st and
   2nd cumulants of log G(r) (G=⟨iγ_pγ_{p+r}⟩² on the final time slice) vs log chord-
   distance; slopes are −2X_typ and −2x⁽²⁾. Cheap add to the Gaussian covariance backend.
3. COMPARISON TEST for Case B ζ=1 (Born): does it give ν≈2.1, ζ₁≈0.39, X₁≈1.00,
   X_typ≈2.66? If yes ⟹ strong evidence Case B Born = Jian's novel DIII n→1 class.
   CAVEAT: Jian is a RANDOM staggered circuit; Case B is a DETERMINISTIC Hamiltonian +
   single fixed-bond measurement + PPS. Both DIII; Jian argues universality is AZ-class-
   only (⟹ should match), but clean-vs-random is exactly the PM-debate axis — so it's a
   test, not a given. Forced endpoint is solid: Jian ν_forced=1.9 matches Fulga DIII
   Anderson, confirming the project's Fulga reference.
4. Confirms the stable conclusion: finite-ν, Z₂-defect-driven, novel-class-beyond-
   Anderson, NO closed form, NOT KT. The ε=2−n RG for DIII is deferred to (unpublished)
   follow-up in the paper, so no closed-form ν is available even there.

ARROW-OF-TIME (Hurvitz-Kochol-Fleurov-Sela, 2604.20828; Meidan acknowledged) — NOT
analytics for Case B (its one exact result, AoT↔bond percolation, α_specific-heat=−2/3,
AoT~|g−g_c|^{8/3}, is the LARGE-q Haar universality, NOT free fermions). But:
- NEW LOCAL THERMODYNAMIC OBSERVABLE that plugs into existing project machinery. No-click
  AoT rate (their Eq. A7): Q/t → γL − 4Γ_min, with Γ_min = Im part of the longest-living
  eigenvalue of the non-Hermitian H_eff — i.e. the project's no-click activity / θ₁ object.
  Nonanalytic at the MIPT. Computable for Case B's H_eff (the two non-Hermitian SSH chains)
  ⟹ a cheap diagnostic for the ζ→0 endpoint. CAVEAT: their exact Γ_min relation assumes a
  TR structure (Ising); check it survives the Bogoliubov-density measurement.
- AoT = Q₀ − 2 S_SM, S_SM = n→1 entropy of the measurement-outcome distribution — the same
  tilted-ensemble partition function the project's SCGF / L_ζ tools compute. Connects the
  AoT to the project's large-deviation/activity framework.
- A free-energy / specific-heat probe (bulk thermodynamic), complementary to ν and the
  entanglement exponents — in principle a different exponent cut (novel-DIII, not −2/3).
- FUTURE DIRECTION + supervisor-group bridge; sidesteps postselection overhead (relevant
  to the PPS focus).


### D8 — Sharpening: ν is FIXED (≈2.1) along the PPS line; forced n→0 is OFF-line (supersedes the "drift 1.9↔2.1" phrasing in the D6 STABLE CONCLUSION and in D7.1)
Re-reading Jian's replica logic + the project's own marginal-cross-vertex result resolves an
internal inconsistency in these notes: D6.3 correctly says ζ→0 ≠ forced n→0, but the D6
"STABLE CONCLUSION" and D7.1 then describe the ζ-line as interpolating forced(1.9)↔Born(2.1).
That conflation is the mistake ChatGPT flagged. Correct picture:

- Replica limit is set by trajectory WEIGHTING: forced = equal weights = n→0; Born = Born-rule
  weights = n→1 (Jian, verified). PPS reweights the BORN ensemble by ζ^{N_clicks} — a tilt of
  Born weights, NOT a switch to equal weights. ⟹ the whole PPS line ζ∈(0,1] stays in n→1.
  Field-theory echo: "ζ enters only the cross-replica vertex" (project finding) = ζ modifies a
  vertex in the n→1 theory; it does not change the replica limit n→1 → n→0.
- ⟹ ν ≈ 2.1 (the n→1 Born value) is FIXED for all ζ∈(0,1]; only λ_c(ζ) moves (driven by the
  relevant single-copy mass ∝α(ζ−1)). The forced n→0 point (ν≈1.9) is OFF the PPS line.
  ζ=0 (strict) is singular: only the no-click trajectory survives, the replica trick
  degenerates → the AoT/Γ_min non-Hermitian problem (D7). So ζ→0 is a singular limit WITHIN
  n→1, not a smooth approach to the forced point.
- ⟹ NO true ν-drift is expected. The project's flat ν (Spearman −0.07) is because the line
  never leaves the n→1 class — NOT because "both endpoints are ≈2." The earlier predicted
  drift ≲0.4 ∝ ζ^{1+p} is a MARGINAL correction-to-scaling (from the ζ-dependent marginal
  cross vertex), not a true exponent change; unmeasurable either way, but the clean thesis
  statement is "ν fixed at the Born value, λ_c(ζ) the only ζ-dependent quantity."
- QJ-vs-QSD origin (REVISED 2026-06-10; "QSD relevant vs QJ marginal" RETRACTED): the sharp
  difference is record statistics → vertex weight [V]: a Gaussian record reverse-HS-resums to
  a weight-≤2 vertex −γ[Σ_μ(M^μ)² − (ζ/2R)(Σ_μM^μ)²]; a point-process record gives the full
  2R-fold ζΓΠ_μn^μ. Both are Δ=2 (tree-marginal) at the common Majorana anchors; LMR's
  ζ*≈0.28 is an R=2 finite-coupling AT-line drift (rate ∝(2n−2)) that dies at n→1, and the
  mechanism is equally available to QJ at R=2 — it is NOT a field-theory reason the
  unravelings differ at vN. (The Rényi-k≥2 ζ* test remains live, sharpened: the feature
  should strengthen monotonically with Rényi index and vanish toward vN; its location is
  non-universal and need not sit at 0.28.)
- Jian's argument (i)+(ii) that Born = n→1 (Born probabilities + POVM normalization → constant
  partition function) survives the PPS tilt: still Born-based, still POVM-normalized after the
  tilt, so still n→1.

EPISTEMIC STATUS: [PLAUSIBLE, strongly motivated, data-consistent] — NOT [VERIFIED]. Rests on
PPS preserving the n→1 limit (LMR structure + the marginal-cross-vertex result) and on Case B
realizing Jian's class at all (standing random-vs-deterministic caveat). Does NOT change the
D6/D7 conclusion (no closed form, finite-ν, Z₂-defect, not-KT); it fixes ONLY the ζ-dependence
of ν: FIXED at the Born value, not drifting.

RE-VERIFICATION: Table I numbers in D7 independently re-confirmed from the freshly-uploaded PDF
(ν 2.1/1.9, X₁ 1.00/1.02, X_typ 2.66/3.53, −x⁽²⁾ 1.80/5.7, ζ₁ 0.39/0.30). The D6 "ν UNVERIFIED"
note is obsolete — it was already verified in D7; D6 was written from a web-fetch that did not
reach the table.


### D9 — Foster-Guo-Jian-Ludwig (arXiv:2510.23706, Oct 2025): the published ε=2−R expansion for the class-DIII MIPT (strongest analytic anchor of the uploaded papers)
Same authors as Jian 2302.09094 (Jian, Ludwig) + Foster, Guo. This IS the "separate
follow-up work" 2302.09094 deferred — the Fu-Kane-style ε=2−R expansion for the class-DIII
MIPT. Primarily about INTERACTING monitored Majorana fermions (volume-law via a dangerously-
irrelevant mass); the part relevant to this (non-interacting) project is the NON-INTERACTING
ε-expansion + operator content.

1. SEMI-ANALYTIC ν (NOT closed-form). Anchor at R=2: SO(R) field → single compact boson,
   action → sine-Gordon S~∫(K/2)(∇φ)²−(M/32)cos4φ (K=λ/4); cos4φ has Δ_M=4/πK, dual vortex
   Δ_V=πK; at K_c=2/π both =2 (a genuine KT point). Deform R=2−ε to reach the physical MIPT
   R→1 (ε=1). Noninteracting FP (their Eq 15) has a single relevant direction ⟹
       ν = (1+√(33−8x))⁻¹ · (8/ε),   x = WZNW level-deformation param (q=8−xε).
   NOT parameter-free: ε=1 is not small (uncontrolled) and x is FIT — x=3 reproduces the
   numerical ν≈2.1 (at ε=1 the formula gives exactly 2). Authors say "not fully controlled."
   ⟹ "no first-principles closed-form ν" STILL holds; what is new is a published RG scaffolding
   reproducing the correct flow topology + a ν consistent with the numerics.
2. KT QUESTION SETTLED (vindicates the D6 withdrawal of "KT at ζ=1"). KT lives at the R=2
   ANCHOR (marginal cos4φ + vortex), NOT at the physical R→1 transition: at R→1 the stiffness
   deviation y_K acquires a positive (relevant) dimension ε/4 ⟹ FINITE-ν defect-unbinding, not
   KT. The project's marginal cross-vertex (Δ≈2) is the echo of the R=2 KT anchor — consistent.
3. CAVEAT-KILLER for "random-vs-deterministic". Foster's model is a DETERMINISTIC-HAMILTONIAN
   monitored system (clean p-wave/Kitaev SC, their Eq 6/9: J, Δ, μ; monitor LOCAL density;
   randomness only from measurement outcomes) — exactly Case B's setup type (vs Jian's random
   circuit) — and they argue it is the SAME SO(R) PCM class-DIII universality. ⟹ Case B (clean
   Kitaev + monitoring) sitting in Jian's class is now on much firmer ground. Their fine-tuned
   J=μ=0 → two decoupled Majorana chains (enhanced class-D) — rhymes with the project's no-click
   "two decoupled non-Hermitian SSH chains" (worth checking the correspondence).
4. RE-CONFIRMS the discriminators + EXPLAINS X₁=1. Their tests (ii)-(iv) = c_ent=0.39 (=Jian
   ζ₁), X_typ=2.66, X₁=1. X₁=1 explained: noninteracting SO(R)×SO(R) ⟹ two Noether currents of
   dim exactly 1; absorbing boundary at the final time slice leaves one ⟹ G~R⁻². Interacting
   DIII (no continuous sym) ⟹ X₁≠1. So X₁=1 is the FREE-FERMION signature; the project (non-
   interacting) should give X₁≈1 at ζ=1 — if so, no hidden interaction generated by QJ/PPS.

ANALYTIC AVENUE OPENED (most promising route from this paper): place the project's ζ cross-
vertex in their R→1 RG (Eq 5) and compute analytically whether ζ is relevant/marginal/
irrelevant — exactly the D8 question (is universality fixed along the ζ-line?). The SO(R)_q
operator spectrum (Δ_{M,p} their Eq S28; spinor/vortex Eq S29) is theory-intrinsic, so the
ζ-vertex must be one of those primaries; the numerical Δ≈2 narrows the candidates. CAVEAT:
their O_M is interaction-induced, the project's ζ-vertex measurement-induced — identify the
operator by representation/dimension, NOT by origin.

DOES NOT TRANSFER: the interacting/volume-law content (dangerously-irrelevant mass; level-
repulsion test (v)) — the project is non-interacting (log-vs-area, no volume law); and PPS (ζ)
is not in their model; their measured operator is the LOCAL density vs the project's distance-3
Bogoliubov bond (same class DIII, so universality should still match — verify via c_ent=0.39,
X_typ=2.66, X₁=1).

BOTTOM LINE: strongest analytic anchor of the three uploaded papers — not a closed form (still
nobody's), but a published ε-expansion ν, a clean KT resolution, a deterministic-Hamiltonian
model that nearly kills the random-vs-deterministic caveat, and a concrete route to pin the
ζ-dependence analytically.


### D10 — Numerics-needs audit for D7/D8 + analytic synthesis: φ reduces to y_ζ^IR; the decisive test is Δ_B at λ_c (2026-06-06)

NUMERICS AUDIT (traced cloning.py → worker_clone_pps.py → observables/spectrum.py):
- The .npz persists B_L + CMI tripartition (S_AB,S_BC,S_B,S_ABC), S_renyi_2/3, n_T,
  chi_k, S_var, covar_Sk, per-realization arrays of those, and corr_decay_r/mean.
- final_covs (covariance matrices) are computed in-run for B_L then DISCARDED — NOT saved.
  ⟹ nothing covariance-based is post-processable from existing output; re-runs required.
- D8 (ν fixed along the line): NO new numerics — reinterprets the existing flat-ν data.
- D7 discriminators (X_typ, x⁽²⁾, correct X₁, c_ent): NEW observable + re-runs. Two issues
  with corr_decay: (1) WRONG OBJECT — single_particle_correlation returns C_ij=⟨c†_i c_j⟩
  (docstring assumes NO pairing, drops anomalous ⟨cc⟩; but the Kitaev topological point HAS
  pairing), then translation_averaged_correlation_decay takes mean_i|C_{i,i+r}| (abs, not
  squared, not Majorana), linearly averaged over clones+realizations. Jian/Foster need
  G(r)=⟨iγ_pγ_{p+r}⟩² = (Γ_{p,p+r})² — the squared Majorana covariance element, which is the
  COMPLETE object (includes pairing) and is literally Γ², already in the evolved covariance.
  (2) ONLY THE MEAN survives — X_typ needs ⟨log G⟩, x⁽²⁾ needs Var(log G) across clones; the
  distribution is averaged away. FIX (small code): per clone compute G=Γ², log G; aggregate
  to ⟨log G⟩→X_typ, Var(log G)→x⁽²⁾, linear mean→correct X₁; store the von Neumann S(L/2)
  (renyi_entropies_batched already computes the n=1 column but only S_renyi_2,3 are saved).
  Re-run at λ_c(ζ=1), ideally PBC + chord distance (project is OBC w/ L=48,96 Friedel; Jian
  uses PBC). c_ent (=0.39 test): cheapest — store S(L/2), fit vs log L over L∈{32,64,128}.

ANALYTIC SYNTHESIS (Foster framework + Section 6/8 operator content):
- CRITICAL: the project's Δ_B≈1 (single-copy mass) and Δ≈2 (cross vertex) are measured at the
  NO-CLICK critical point (deterministic gapless H_eff, two non-Hermitian SSH chains), NOT at
  the MIPT FP (strong coupling λ~O(1)). These are bare/UV dimensions; IR relevance at the MIPT
  is open (= Section 9's Δ_ζ^IR question). Same status as Foster's R=2-anchor dimensions.
- Foster SETTLES the transverse exponent: single relevant direction = stiffness = λ, y_λ≈1/2
  (x=3 ↔ ν≈2.1). RESOLVES the project's y_λ=1-vs-1/2 ambiguity (Section 8): the ν=1 branch
  read the single-particle ξ_nc~λ⁻¹ as the correlation length, but Section 9 itself warns it
  needn't be the many-body one — and it isn't. Many-body transverse ν≈2.1, y_λ≈1/2.
  (Caveat: Foster's y_λ=1/2 is x=3 fitted to Jian's numerical ν≈2.1, not first-principles.)
- ⟹ the boundary exponent φ = y_λ/y_ζ collapses to ONE unknown: y_ζ at the MIPT (cross vertex
  marginal ⟹ no relevant ζ-eigenvalue; y_ζ set by the single-copy mass). The data brackets it
  GIVEN y_λ=1/2: φ≈0.56 (on λ_c) ⟹ y_ζ≈0.9; φ≈0.8 (on r_c) ⟹ y_ζ≈0.6. So the single-copy
  mass is RELEVANT but mildly renormalized below its no-click value 1.
- DECISIVE CHEAP TEST: measure Δ_B at λ_c(ζ=1) (re-run cross_vertex_dimension.py on the
  SAMPLED critical state at λ_c, not the no-click state). Prediction: Δ_B(λ_c) = 2 − y_ζ ≈
  1.1–1.4 (vs Δ_B(no-click)≈1.0). If confirmed ⟹ φ pinned, picture closed. (Needs sampled
  critical covariances at λ_c → same re-run/persistence issue.) More central to the project's
  actual open question (the φ exponent) than the D7 multifractal exponents.
- ANALYTIC STOP POINT: deriving y_ζ^IR from first principles = "identify the SO(R)_q primary +
  track to R→1." The single-copy mass is a dim-1 fermion-bilinear/dimerization operator at the
  free-fermion anchor (K₀=1) — the Noether/bilinear sector (Foster's X₁=1 argument), so NOT
  Foster's interaction-mass O_M (Δ=2). Obstruction: the project's anchor (free-fermion K₀=1)
  and Foster's (R=2 boson, K_c=2/π) differ, so the operator dictionary must be built across a
  CHANGE OF ANCHOR before the R→1 deformation applies — genuinely a Meidan-scale calculation,
  not closeable in-chat. The numerical Δ_B(λ_c) sidesteps it.
- √ζ STATUS: not revived as proven, but status changed. Section 10 killed √ζ partly via y_λ=1
  ⟹ φ=1; Foster replaces that with y_λ=1/2. Clean √ζ (φ=1/2) needs y_ζ=1 exactly; data wants
  y_ζ≈0.6–0.9 ⟹ φ intermediate (neither clean √ζ nor linear). So √ζ unproven & probably not
  exact, BUT the "φ=1 linear" reading was using the wrong y_λ.

FULL PICTURE: ζ=1 is the class-DIII Born MIPT (finite ν≈2.1, Z₂-defect, novel-class, no closed
form). Along ζ∈(0,1] the limit stays n→1 (D8) ⟹ transverse ν≈2.1 fixed, only λ_c(ζ) moves;
forced n→0 (1.9) off-line; ζ→0 singular (λ_c→0). Boundary φ=y_λ/y_ζ, y_λ≈1/2 fixed (Foster),
y_ζ = single-copy mass dim at the MIPT = the one open number (data ⟹ ≈0.6–0.9). Two re-runs
close the gaps: Δ_B at λ_c (φ/y_ζ) and multifractal X_typ,x⁽²⁾ at λ_c(1) (Case B Born = Jian's
class). Neither post-processable (covariances not persisted).
