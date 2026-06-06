# Integration of collaborator response on Problems 1 and 2

**Date:** 2026-05-25

The collaborator's response to `COLLABORATION_PROMPT.md` is essential and changes the
theoretical framing in important ways. Below I integrate it with a critical numerical
check on the BdG/microscopic length scale question.

## Summary of collaborator's findings

**Problem 1 (IR dimension of $\mathcal{O}_\zeta$ at class-DIII NLSM FP):**
The collaborator argues that $\partial S_{\rm NLSM}/\partial\zeta$ is NOT a dimension-1
primary. Rather, $\zeta$ likely renormalizes the marginal NLSM stiffness coupling
$\mathrm{Tr}(\partial_\mu U^T \partial_\mu U)$ (tree-level dimension 2). Therefore:
$$\boxed{\Delta_\zeta^{\rm IR} = 1\text{ is NOT established by the Le Gal–Schir\'o NLSM.}}$$
What we know is just $\Delta_\zeta^{\rm UV} = 1$ from the microscopic cross-Choi.

**Problem 2 (microscopic BdG length for distance-3 Majorana bond):**
For the exact $d_j = (\gamma_{2j} - i\gamma_{2j+3})/2$ used in the code, the collaborator
derived the Bloch dispersion (assuming standard nearest-neighbor hopping):
$$q(k) = 2w\cos k + (i\alpha/2) e^{2ik}$$
and at the Fermi point $k = \pi/2$, the imaginary gap is $\Delta_{\rm BdG} \sim \alpha/2$, giving
$$\boxed{\xi_{\rm BdG} \sim 1/\alpha, \text{ NOT } 1/\alpha^2.}$$
The transfer-matrix root analysis confirms $\xi_{\rm transfer} \sim 4w/\alpha$.

## Critical numerical check: actual code's bond structure

Inspecting `pps_qj/gaussian_backend.py:11-19`, the actual code has TWO bonds per unit cell:
- Distance-3 bond $(\gamma_{2j}, \gamma_{2j+3})$ with strength $w$ (from hopping) and $-i\alpha$ (from measurement)
- Distance-1 bond $(\gamma_{2j+1}, \gamma_{2j+2})$ with strength $-w$ (from hopping, no measurement)

I verified by direct inspection at $L=6$: there are 5 distance-1 bonds AND 5 distance-3 bonds in the hopping matrix. The collaborator's BdG analysis used only distance-1 hopping (giving $2w\cos k$); the actual code has both.

Repeating the Bloch analysis for the **actual** model (with both bond types) and the **respondent's simplified** model gives:

| Model | min$_k$ |Im(E)| scaling with α |
|---|---|
| Respondent's simplified (only distance-1 hopping) | $\alpha^{1.000}$ → ξ ~ $α^{-1}$ |
| Actual code (distance-1 + distance-3 hopping) | $\alpha^{-0.005}$ — **essentially constant** |

The actual code's no-click effective Hamiltonian is **effectively gapless**: there exist
momenta where $\mathrm{Im}(E(k)) = 0$ exactly. This means the no-click correlations decay
algebraically rather than exponentially, and there is NO single-particle localization length
in the conventional sense.

See `analysis/bdg_actual_vs_simple.png` for the comparison.

## Consequence

Neither $\xi \sim \lambda^{-2}$ (earlier hypothesis) nor $\xi \sim \lambda^{-1}$
(respondent's analysis for simplified model) describes the actual code's single-particle
no-click physics. The actual model appears gapless at the single-particle BdG level.

**The $\lambda^{-2}$ scale in the matched-NLSM argument must therefore be interpreted
as a purely field-theoretic crossover scale** ($\xi_\lambda^{\rm cross} \sim \lambda^{-1/y_\lambda} = \lambda^{-2}$ from
class-DIII multicritical $\nu = 2$), with no microscopic single-particle counterpart in
this specific model.

## Final honest theoretical position

The √ζ scaling rests on:
1. $\Delta_\zeta^{\rm UV} = 1$ — **microscopically established** ✓ (engineering + lattice cross-Choi)
2. $\nu = 2$ for class-DIII multicritical NLSM — **literature input** (König-Brouwer 2014; consistent with our FSS data in middle range)
3. Linearized RG at the multicritical point $(\lambda, \zeta) = (0,0)$ with the two relevant operators
4. Critical condition reached at the crossover scale $\xi_\lambda^{\rm cross}$ (assumption; IR running of ζ inside NLSM not parametrically large)

This is a **matched field-theory hypothesis**, not a derivation from microscopic BdG.
It is consistent with the data ($\phi = 0.56 \pm 0.05$, within 1.3σ of 1/2).

## What's now closed vs. open

| Claim | Status |
|---|---|
| $\Delta_\zeta^{\rm UV} = 1$ | Established (cross-Choi lattice) |
| $\nu = 2$ at class-DIII multicritical | Literature input, data-consistent |
| Matched-NLSM gives $\phi = 1/2$ | Robust if criticality reached at matching scale |
| $\Delta_\zeta^{\rm IR} = 1$ (no anomalous IR dim) | **NOT proven**; ζ likely renormalizes stiffness |
| Microscopic ξ ~ λ^{-2} from no-click BdG | **Not supported** for this model — actual model gapless |
| Microscopic ξ ~ λ^{-1} from no-click BdG | True for SIMPLIFIED model (distance-1 hopping only), but not the actual code |
| Closed-form $\lambda_c = \sqrt\zeta/(1+\sqrt\zeta)$ | **Phenomenological** — not derived from one-loop NLSM |

## Action items for the LaTeX thesis section

The current `sec_matching_revised.tex` already correctly distinguishes UV from IR dimensions
and presents $\xi_\lambda^{\rm cross}$ as an RG-defined crossover length. The "Validity"
subsubsection already lists the assumptions. So no major rewrite is needed — the
thesis section is already in the right form.

The only adjustment: in the "Validity" subsubsection, the bullet point about microscopic
single-particle counterpart should be **strengthened** to say explicitly that the actual
distance-3 bond structure gives an EFFECTIVELY GAPLESS no-click h_eff (not just "doesn't
match the standard SSH unit cell"). This is a stronger and more correct statement.
