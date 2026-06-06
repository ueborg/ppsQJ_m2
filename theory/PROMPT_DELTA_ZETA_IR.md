# New-chat prompt: Compute Δ_ζ^IR at the class-DIII NLSM fixed point

## Context

This is a theoretical physics calculation, self-contained. No code needed.

## The model

A 1D class-DIII (Altland-Zirnbauer classification) MIPT described by:
- Replica Keldysh action: S = S_nH + S_QJ^(ζ), with
  S_QJ^(ζ) = -iζγ ∫ dt Σ_x Π_r V_{x,r} (cross-replica vertex)
- After HS decoupling: SO(R) NLSM on target SO(R), derived in
  Le Gal-Schirò arXiv:2511.22506 §5.3.3
- One-loop NLSM action: S[U] = (1/g_B) Tr(∂_μ U^T ∂_μ U), U ∈ SO(R)
- One-loop β-function (Le Gal-Schirò): dg_R/d ln L = ((R-2)/8π) g_R²
  In replica limit R→1: dg_R/d ln L = -(1/8π) g_R² (asymptotic freedom)

## What is established

The cross-vertex operator V_ζ = Π_r V_{x,r,+} V†_{x,r,-} has:

**UV dimension (free-Dirac lattice fixed point):**
    Δ_ζ^UV = 1   (engineering dimension; confirmed by lattice cross-Choi 2-point function)
    y_ζ^UV = 2 - Δ_ζ^UV = 1   (relevant at free-Dirac UV)

**IR dimension (class-DIII NLSM fixed point):**
    Δ_ζ^IR = ???   ← THIS IS WHAT WE WANT TO COMPUTE

The matched-NLSM derivation of λ_c ~ √ζ uses y_ζ^UV = 1 for the UV-to-crossover
running. Whether anomalous IR corrections change the result depends on Δ_ζ^IR.

## The key insight from a collaborator

An external collaborator (see theory/COLLABORATOR_RESPONSE_INTEGRATION.md) argued that:

    ∂S_NLSM/∂ζ|_{ζ=0}

is NOT a dimension-1 primary at the NLSM level. Instead, ζ likely renormalizes
the marginal NLSM stiffness coupling, so ∂S_NLSM/∂ζ ~ Tr(∂_μ U^T ∂_μ U)
which has tree-level dimension 2 (marginal in 2d). Therefore Δ_ζ^IR ≈ 2.

**The question is whether this is correct, and if so, what exactly Δ_ζ^IR is.**

## The calculation requested

**Step 1: Identify O_ζ^NLSM.**

Starting from the HS-decoupled replica Keldysh action (Le Gal-Schirò Eq. (13)-(17)),
write:
    S_total(ζ) = S_NLSM(g_B(ζ)) + (ζ-independent pieces)

where g_B(ζ) is the bare NLSM coupling as a function of ζ. Compute:
    O_ζ^NLSM[U] = ∂S_total/∂ζ|_{ζ=0}

At leading order, this should be proportional to ∂g_B/∂ζ × (kinetic NLSM term).
Check what ∂g_B/∂ζ is: from Le Gal-Schirò, g_B ∝ (measurement rate), and ζ
multiplies the cross-replica vertex, so ∂g_B/∂ζ ~ g_B/ζ ≠ 0.

**Step 2: Classify O_ζ^NLSM.**

Is it:
(a) A multiple of the kinetic term Tr(∂_μ U^T ∂_μ U) — i.e., a stiffness renormalization?
(b) A WZW coefficient?
(c) A separate relevant operator?

If (a): Δ_ζ^IR = 2 (marginal, tree level). Under one-loop RG, the anomalous dimension
is γ_ζ = d(ln K_ζ)/dℓ where K_ζ is the coefficient, evaluated at the NLSM fixed point.

**Step 3: One-loop anomalous dimension.**

For the SO(R) NLSM in the background-field RG expansion, the kinetic term
Tr(∂_μ U^T ∂_μ U) does not acquire anomalous dimensions at one loop (it only
renormalizes the coupling g, not through a separate operator). Therefore if O_ζ ~ kinetic
term, then γ_ζ = 0 at one loop, so Δ_ζ^IR = 2 exactly at one loop.

But check: is there a subleading operator in ∂S/∂ζ that IS a separate relevant primary?

**Step 4: Physical consequence.**

If Δ_ζ^IR = 2 (marginal):
- Inside the NLSM regime (ξ_λ^cross < ℓ < ξ_crit), ζ runs marginally,
  not as a relevant operator.
- The formula ζ(ℓ) ~ ζ·(ℓ/a)^{y_ζ^IR} = ζ·(ℓ/a)^0 = ζ (constant)
  would mean ζ stops running inside the NLSM regime.
- Then the critical condition involves only the bare ζ, not renormalized.
- Matching: ζ_eff(ξ_λ) = ζ (constant) ~ K* → independent of λ → WRONG.

This is a contradiction. If ζ is truly marginal in the NLSM regime, then the
matched argument breaks down because ζ wouldn't grow to K* at the matching scale.

**Step 5: Resolution.**

How is this resolved? One possibility: the ζ-running outside the NLSM regime
(from a to ξ_λ^cross, governed by y_ζ^UV = 1) is sufficient. Once ζ_eff reaches
K* at the matching scale, criticality is declared, and there's no further running
needed inside the NLSM. The NLSM is only entered AFTER the critical condition is
satisfied, not before. Therefore the marginal nature of ζ at the IR NLSM FP
doesn't contradict the matched argument — the critical condition is a UV-to-crossover
statement, not an IR statement.

**Formalize this argument.** Is it rigorous, or does it require additional assumptions?

## References

- Le Gal-Schirò arXiv:2511.22506: Full Keldysh-replica derivation of DIII NLSM from QJ
  monitoring, with explicit saddle-point and gradient expansion (§5.3.3)
- König-Brouwer PRB 90, 165140 (2014): ν=2 for class-DIII multicritical NLSM
- Senthil-Marston-Fisher PRB 60, 4245 (1999): Background-field RG for class DIII SO(N) NLSM
- Hikami-Larkin-Nagaoka Prog. Theor. Phys. 63, 707 (1980): One-loop RG for symplectic class

## What a successful answer looks like

- Explicit identification of O_ζ^NLSM[U] in the Le Gal-Schirò notation
- Classification of this operator (type a/b/c above)
- One-loop anomalous dimension γ_ζ at the DIII NLSM fixed point
- Whether Δ_ζ^IR = 1 or 2 or something else
- Whether the matched-NLSM derivation of λ_c ~ √ζ is unaffected by this result
  (the "Step 5 resolution" above, formalized)
