# New-chat prompt: Derivation of the crossover function λ_c(ζ) = √ζ/(1+√ζ)

## Context

This is a theoretical physics question in a self-contained form. No code
access is needed. All context is below.

## The model

1D free-fermion Kitaev chain, single Bogoliubov-mode density measurement L_j = d†_j d_j
at rate α, hopping H at rate w, with α+w=1 and λ≡α. PPS parameter ζ ∈ (0,1].
Class DIII in the Altland-Zirnbauer classification.

## What is established

The MIPT critical line λ_c(ζ) has been derived in the **small-ζ limit** via a
matched-NLSM argument with two inputs:

1. UV cross-vertex dimension: Δ_ζ^UV = 1 → y_ζ = 1 (free-Dirac UV fixed point)
2. Class-DIII multicritical correlation exponent: ν = 2 → y_λ = 1/2

Linearised RG at the multicritical point (λ,ζ) = (0,0) with the two relevant operators:
```
dλ/dℓ = y_λ λ = (1/2) λ
dζ/dℓ = y_ζ ζ = (1)  ζ
```
The crossover scale ξ_λ^cross ~ λ^{-2} is where λ(ℓ) ~ 1. At that scale,
ζ_eff ~ ζ·λ^{-2}. Critical condition ζ_eff ~ K* (universal NLSM coupling) gives:

    λ_c ~ √ζ    (small ζ, universal)

The Born-rule endpoint is Carollo's result: λ_c(ζ=1) = 1/2 (Phys. Rev. A 98, 010103).

## The phenomenological interpolation that fits the data

The closed-form:
    λ_c(ζ) = √ζ / (1 + √ζ)

equivalently:
    α_c/w = √ζ

has no free parameters, recovers λ_c(0→0) = 0 and λ_c(1) = 1/2 exactly, and fits
the numerical data within ~10% over the full range ζ ∈ [0.03, 0.85].
Numerically: φ = 0.56 ± 0.05 (free power-law fit; consistent with 1/2 at 1.3σ).

## The question

Can the closed-form λ_c(ζ) = √ζ/(1+√ζ) be **derived** from a two-parameter RG
separatrix calculation, rather than remaining a phenomenological ansatz?

## The setup for this calculation

The linearised RG equations near (0,0) with eigenvalues (y_λ, y_ζ) = (1/2, 1) give
the critical separatrix as the manifold in the (λ,ζ) plane that flows TOWARD the
multicritical point under reverse-RG (or away from it under forward-RG). The leading
result of the linearised RG is just λ ~ √ζ (from the scaling argument above).

However, the FULL separatrix — including the global connection between the
multicritical point at (0,0) and the Born-rule fixed point at (λ_c=1/2, ζ=1) —
requires either:

(A) **Exact separatrix from 1-loop RG flow equations**: integrate
    dλ/dζ = (y_λ/y_ζ) (λ/ζ) + nonlinear corrections
    to find the separatrix λ(ζ) globally, not just near (0,0).

(B) **Padé / Borel resummation**: the small-ζ expansion λ_c = C·√ζ + corrections
    and the large-ζ expansion λ_c = 1/2 - a(1-ζ) + ... can be matched via a
    Padé approximant. Check if the simplest [1,1] Padé in √ζ reproduces
    √ζ/(1+√ζ) exactly.

(C) **Self-consistency / mean-field argument**: if there is a self-duality or
    mean-field argument that pins the form of the separatrix.

## What to investigate

1. **Try approach (B) first (quickest).** Write the small-ζ expansion:
   α_c/w = C√ζ + c_1 ζ + c_2 ζ^{3/2} + ...
   and the large-ζ expansion near ζ=1:
   α_c/w = 1 + a(ζ-1) + b(ζ-1)^2 + ...
   (Note: at ζ=1, α_c/w = 1, equivalently λ_c = 1/2.)
   Determine whether a Padé approximant in √ζ anchored at both limits gives
   √ζ/(1+√ζ) with C=1 and all higher coefficients zero, or whether it requires
   corrections.

2. **Try approach (A).** The one-loop NLSM β-function for class DIII SO(R) in
   the replica limit R→1 is: dg/dℓ = -(1/8π)g² + O(g³) (asymptotic freedom,
   Le Gal-Schirò arXiv:2511.22506). Write the coupled flow equations for (g,ζ)
   where g = g_B = (bare NLSM coupling) ~ λ/(1-λ)·ζ (approximately). Find the
   separatrix g_c(ζ) and translate back to λ_c(ζ).

3. **Self-consistency check.** The form λ_c = √ζ/(1+√ζ) is equivalent to
   the natural variable α_c/w = √ζ, which says the critical condition is
   simply (α_c/w)² = ζ, i.e., α²/w² = ζ. Is there a physical interpretation
   of this as a self-consistency condition? For example: at criticality, the
   PPS weight per-unit-time ζ·(click rate) ~ ζ·α balanced against the kinetic
   energy scale w² gives α²/w² ~ ζ?

## Useful prior results

- The empirical slope: C^fit ≈ 0.91 ± 0.10. Close to C=1 but not exactly pinned.
- The formula √ζ/(1+√ζ) has C=1 exactly. If C were ≠ 1 this formula would need
  a prefactor, but C≈1 makes it plausible that the formula is exact.
- The formula can be rewritten as: the critical point satisfies
  (λ_c)² + λ_c(√ζ - 1) - ... this does not factorise simply.
- Alternative: at the Born-rule end ζ=1, the critical condition is λ_c = 1/2, i.e.,
  α_c = w_c. If the general critical condition is α_c = w_c·√ζ (a "measurement rate
  = hopping rate times √ζ" statement), that gives √ζ/(1+√ζ) exactly. What physical
  mechanism would give this?

## What a successful answer looks like

Either:
- A derivation from approach (A) or (B) that gives √ζ/(1+√ζ) exactly (or
  identifies the corrections to it and shows they are subleading), OR
- A clear argument for why C=1 is non-universal and √ζ/(1+√ζ) is only
  an approximation with no rigorous derivation, specifying what the leading
  correction would be and at what ζ it becomes visible.
