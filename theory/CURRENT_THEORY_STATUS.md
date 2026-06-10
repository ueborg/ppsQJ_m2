# Current Theory Status — QJ-PPS Case B

> ★ SUPERSEDED (2026-06-10) — this file is dated 2026-06-03 and is stale. Read
> `HANDOFF.md` top block for the canonical state. Key reversals since: the ζ=0 anchor
> is solved exactly (Fermi-step critical line, EP at λ*=4/5, ν₀=1); **ξ_ps~λ⁻² is
> REFUTED** (no scale has that exponent; the λ⁻¹ scale is the SELECTION length ℓ_λ, not
> a state ξ); the boundary is a CORNER-MATCHING law λ_c=A√ζ (φ=1/2 [P], gated by the
> area-phase ξ run), NOT a y_λ/y_ζ ratio; the Case-A Ising tag belongs to the ζ=0
> endpoint, with the Born line SU(2)₁ (ν=2/3 [P]); QSD/QJ are BOTH marginal at the
> anchor (the "QSD relevant" dichotomy is retracted). Treat anything below about the
> boundary derivation or no-click localization length as historical.

**Canonical theoretical state. Last update: 2026-06-03.**
Supersedes the "√ζ is closed" framing in the older `HANDOFF.md` TL;DR and in
`ONE_LOOP_RG.md`. Companion docs: `OPEN_ANALYTIC_PROBLEMS.md`,
`NUMERICS_STATUS_AND_PLAN.md`. Verified-fact provenance:
`analysis/cross_vertex_dimension.py`, `analysis/noclick_spectrum_probe.py`.

This document distinguishes three epistemic levels throughout:
**[VERIFIED]** (exact/numerical), **[PLAUSIBLE]** (theory-motivated, unproven),
**[UNRESOLVED]** (open).

---

## 1. Model

1D Kitaev chain at the topological point (μ=0, Δ=w), single Bogoliubov-density
measurement, quantum-jump (QJ) unraveling, partial post-selection (PPS) with
parameter ζ ∈ (0,1]. Control parameter λ = α/(α+w), with α+w=1, so the natural
physical ratio is r = α/w = λ/(1−λ).

## 2. Jump operators (the load-bearing correction)

The measurement is a **projector**, not a linear mode operator:
$$L_j = \sqrt{\alpha}\,P_j, \qquad P_j = d_j^\dagger d_j = \tfrac12(1+B_j),$$
with d_j the Bogoliubov-Kitaev mode and B_j = i γ_{2j} γ_{2j+3} a Majorana
bilinear. The earlier linear-jump form L_j = √α d_j (used in
`qj_two_replica_derivation.md`) is **wrong** for this model and any scaling
argument built on it is invalid. The code's `apply_projective_jump` confirms
the projector: it projects onto the +1 eigenspace of i γγ.

## 3. QJ vs QSD

The Lindbladian is unraveling-independent; QJ (discrete projector clicks) and
QSD (diffusive) differ only in trajectory sampling. The universal MIPT exponents
are unraveling-independent, but the microscopic operator that ζ couples to, and
its scaling dimension, are **not** — and that is where the QJ-PPS story differs
from the QSD-PPS story of KMR/LMR. In QJ-PPS the genuinely new replica coupling
is marginal (Section 6); this is the structural feature of this project.

## 4. Choi/Keldysh replicated structure

The doubled (ket = +, bra = −) generator is
$$\mathcal L = -i(H_+ - H_-) - \tfrac{\alpha}{2}(P_+ + P_-) + \zeta\alpha\,P_+P_-.$$
ζ enters **only** the recycling (cross) term. This is exact and structural.

## 5. Corrected operator content

Expanding the recycling term with P = ½(1+B):
$$P_+P_- = \tfrac14\bigl(1 + B_+ + B_- + B_+B_-\bigr).$$

Collecting coefficients in 𝓛 (the −α/2(P_++P_-) decay also contributes to the
single-copy piece):

- **Identity** — normalization/trace, coefficient ζα/4 − α/2.
- **Single-copy mass** B_+ + B_- — coefficient **α(ζ−1)/4**. Relevant (Δ=1).
  **Vanishes at ζ=1.** This is a renormalization of the no-click mass, not a
  new stochastic coupling.
- **Cross-replica** B_+B_- — coefficient **ζα/4**. Because ⟨B⟩ ≠ 0, the *raw*
  operator B_+B_- splits as
  $$B_+B_- = \langle B\rangle^2 + \langle B\rangle(\delta B_+ + \delta B_-) + {:}B_+B_-{:},$$
  i.e. a dim-1 single-copy admixture plus the genuine dim-2 normal-ordered
  vertex :B_+B_:. The dim-1 admixture is what makes the raw correlator look
  relevant; the genuine cross vertex is :B_+B_:.

## 6. Scaling dimensions — VERIFIED

Exact free-fermion ground-state computation (L=600, pure-state check
max|MM^T − I| = 1.3e-14), `analysis/cross_vertex_dimension.py`:

- ⟨B⟩_bulk ≈ −0.637 (nonzero).
- Single-copy connected correlator C(r) = ⟨B_0 B_r⟩_c ~ r^{-2.018}
  ⟹ **Δ_B = 1.01** (Ising energy operator).
- Genuine cross vertex ⟨:B_+B_:(r) :B_+B_:(0)⟩_c = C(r)² ~ r^{-4.036}
  ⟹ **Δ_{:B_+B_:} = 2.02 (MARGINAL, y_ζ = 0)**.
- Raw (non-normal-ordered) ⟨B_+B_- B_+B_-⟩_c = 2⟨B⟩²C(r) + C(r)² ~ r^{-2.015},
  i.e. apparent Δ=1 from the dim-1 piece. **This is the most likely origin of
  the project's earlier "Δ_ζ = 1" claim** (measuring a single bilinear or the
  raw correlator, both ~ r^{-2}, not the genuine cross vertex ~ r^{-4}).

## 7. No-click length — VERIFIED

`analysis/noclick_spectrum_probe.py`. The Majorana Hamiltonian decouples exactly
into two non-Hermitian SSH chains (sublattices E = {0,3 mod 4}, O = {1,2 mod 4};
inter-sublattice coupling = 0). Two distinct scales exist:

- **Dimerization/band length** ξ_dim ~ λ^{-2} (the gap of each SSH chain).
- **Steady-state correlation length** ξ_nc ~ λ^{-1}, prefactor ≈ 4–5, selected
  by the slowest-decaying modes at k* ~ λ (NOT the band ground state).

The entanglement-relevant length is **ξ_nc ~ λ^{-1}** (xi·λ flat ≈ 4–5 over
λ ∈ [0.10, 0.40]; xi·λ² falls monotonically). The "ξ ~ λ^{-2}" used by the
matched-NLSM is the band length, not the steady-state length. Consequence:
λ_c(ζ=0) = 0, no floor.

## 8. PLAUSIBLE (theory-motivated, not verified)

- **One-loop marginality at n→1.** With the Ising fusion ε×ε = 1 and replica
  combinatorics, β_g ∝ −(2n−2) g², which vanishes at the von Neumann limit
  n→1. So the cross vertex is one-loop marginal at n=1. The sign at finite n
  (Rényi) and higher loops are not pinned. (This is the clean version of
  `qj_chiral_vertex_result.md`.)
- **Crossover-exponent relation.** φ = y_λ / y_ζ for the boundary λ_c ~ ζ^φ.
  The relevant ζ-bearing operator is the single-copy mass (y_ζ = 1); the cross
  vertex is marginal and contributes no relevant ζ-eigenvalue. Then φ = y_λ.
  If y_λ = 1 (suggested by ξ_nc ~ λ^{-1}, i.e. ν=1) ⟹ φ = 1 (linear). If
  y_λ = 1/2 (ν=2, class-DIII literature) ⟹ φ = 1/2 (√ζ). The data sits between
  (see NUMERICS doc).

## 9. UNRESOLVED

- The many-body FSS exponent ν (1? 2? intermediate ≈1.25?). The single-particle
  ξ_nc ~ λ^{-1} need **not** equal the many-body correlation length.
- Whether :B_+B_: stays marginal at the IR (class-DIII NLSM) fixed point or
  renormalizes (the Δ_ζ^IR question).
- The actual λ_c(ζ) exponent: free fits give φ≈0.56 on λ_c and ≈0.7–0.85 on the
  physical ratio r_c. Not settled.
- Small-ζ asymptotics: masked by a finite-size floor in current data.

## 10. Why the old √ζ derivation is invalid

The matched-NLSM derivation required **both** Δ_ζ = 1 (relevant cross vertex)
**and** ξ ~ λ^{-2}. Both are now contradicted:
- the genuine cross vertex is marginal (Δ = 2, Section 6);
- the steady-state no-click length is λ^{-1}, not λ^{-2} (Section 7).

With the correct inputs, the same crossover-scaling machinery gives φ = 1
(linear) or no clean power, not φ = 1/2. Independently, the project's own
`PHASE_BOUNDARY_DERIVATION.md` already documents that no first-principles
derivation of √ζ closes (every attempt produces the wrong-sign 1/√ζ). The
empirical "C ≈ 1" support quoted there is from a fit with the exponent **fixed**
at 1/2, which does not test the exponent.

## 11. What can / cannot be claimed

**CAN claim:** the genuine PPS cross vertex is marginal (Δ≈2); the steady-state
no-click length is ξ_nc ~ λ^{-1}; the Born endpoint λ_c(1) = 1/2 (Carollo,
recovered in data); and the previous √ζ *derivation* via Δ_ζ=1 + ξ~λ^{-2} is
invalid for this projector-jump model.

**CANNOT claim:** that √ζ is proven; that λ_c = ζ/(1+ζ) is proven; that the
phase-boundary exponent is settled; or a value for the many-body ν. √ζ survives
only as one empirical possibility, and current free-exponent evidence (φ≈0.56 on
λ_c, ≈0.8 on r_c) does not favor it over an intermediate/near-linear exponent.
