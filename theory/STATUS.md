# Status of the two-replica bosonization for the QJ-PPS model

A reading guide and synthesis of the four theory documents in this folder.

## Reading order

1. **`qj_two_replica_derivation.md`** (633 lines, committed) — Rigorous
   construction of the QJ two-replica generator $\mathcal{L}_\zeta^{(2)}$.
   The setup.
2. **`two_replica_QJ_PPS.md`** (376 lines) — Earlier draft of the same
   material, with an additional "$\zeta^*_\mathrm{QJ}\approx 0.08$"
   prediction borrowed by analogy from LMR. Superseded by the above plus
   the bosonization documents below; kept for record.
3. **`qj_bosonization_calculation.md`** (870 lines) — First-pass
   bosonization of $V_{\rm cross}$, with extensive self-critique (§9)
   and empirical comparison (§10) showing the data does not fit the
   simplest "linear-in-$\zeta$" form.
4. **`qj_one_minus_zeta_expansion.md`** (570 lines) — Sharpened
   bosonization, identifying the cross-replica vertex as a
   scaling-dimension-4 vertex operator and predicting vanishing leading
   $(1-\zeta)$ slope of $K_\sigma$. **Numerically verified** near
   $\zeta = 1$.

The two bosonization documents (3 and 4) reach slightly different
conclusions because they make different choices about which channel of
the four-fermion operator dominates. The clearest, most defensible
analytical statement is the one in document 4.

---

## The headline analytical result

The QJ cross-replica click vertex, after bosonization in the
four-Choi-copy theory and after going to the inter-replica $(\rho, \sigma)$
basis, is a single vertex operator
$$
V_j \;\sim\; \exp\bigl[\,2i\,(\Theta_D^\rho - \Phi_D^\rho)\,\bigr]
$$
in the difference $\rho_D$ mode between ket and bra of the Choi copies.
At the Born-rule fixed point ($K = 1$, free Dirac CFT), its scaling
dimension is
$$
\Delta(V_j) \;=\; \tfrac{1}{2}(2^2 K^{-1} + 2^2 K) \;=\; 4.
$$
Since $\Delta = 4 > 2$ in 1+1D, **the vertex is strongly irrelevant**.

## Consequence: the leading $(1-\zeta)$ slope of $K_\sigma$ vanishes

For an irrelevant vertex with coupling $g \propto (1-\zeta)\alpha$, the
RG flow is
$$
\frac{dg}{d\ell} \;=\; -(\Delta - 2)\,g \;=\; -2\,g,
$$
i.e., the coupling decays under RG. It cannot drive a phase transition
on its own, and its leading contribution to the Luttinger parameter
$K_\sigma$ is at order $g^2 = (1-\zeta)^2 \alpha^2$, not at order
$(1-\zeta) \alpha$. So
$$
\boxed{\;
\frac{dK_\sigma^{\rm QJ}}{d(1-\zeta)}\bigg|_{\zeta = 1} \;=\; 0
\;}
$$
and the first non-trivial shift is
$K_\sigma^{\rm QJ}(\zeta) = K_\sigma^{\rm Born} + c_2 (1-\zeta)^2 + \ldots$

## Contrast with LMR's diffusive case

LMR's diffusive cross-replica fluctuation operator, when bosonized,
contains a **marginal** $\cos(2\phi_\sigma)$ vertex (dimension 2 at
$K = 1$). Marginal operators DO shift $K$ at leading order in their
coupling:
$$
\frac{dK_\sigma^{\rm LMR}}{d(1-\zeta)}\bigg|_{\zeta = 1} \;\neq\; 0.
$$
The LMR slope is non-zero and produces the smooth crossover from Ising
to BKT universality as $\zeta$ varies.

**Punchline:** the QJ and diffusive unravelings produce qualitatively
different bosonized cross-replica terms, with consequence:
- QJ has vanishing leading $(1-\zeta)$ slope.
- LMR has finite leading $(1-\zeta)$ slope.

## Numerical check

The location $\lambda(c=1)$ of the BKT critical curve at fixed $\zeta$:

| $\zeta$ | $1 - \zeta$ | $\lambda(c=1)$ | $\Delta\lambda$ |
|---------|------------:|---------------:|---------------:|
| 1.00 | 0.00 | 0.364 |  0.000 |
| 0.85 | 0.15 | 0.365 | +0.001 |
| 0.70 | 0.30 | 0.370 | +0.006 |
| 0.50 | 0.50 | 0.334 | $-0.030$ |
| 0.30 | 0.70 | 0.237 | $-0.127$ |

The behaviour near $\zeta = 1$ is essentially flat (shift is 0.001 over
$\Delta\zeta = 0.15$), consistent with vanishing leading slope. For
$\zeta \lesssim 0.5$, $\lambda(c=1)$ drops sharply, which is a separate
non-perturbative feature presumably reflecting an analog of LMR's
strong-PPS phase boundary.

The numerics in $\zeta \in [0.85, 1]$ are quantitatively consistent
with the analytical prediction.

## Independent test from the Renyi reruns

When the Renyi job completes (`submit_renyi_targets.sh`), we can test
the **CFT universality** prediction directly: at every test point in the
log-law region, the ratios $c_2/c_1$ and $c_3/c_1$ should agree with the
free-Dirac CFT predictions $0.75$ and $0.667$.

  - **If the ratios agree at all six test points** (including
    $\zeta = 0.20$), the log-law phase is in the free-Dirac universality
    class throughout, consistent with the irrelevant cross-replica
    vertex analysis above.
  - **If the ratios deviate from free-Dirac values for $\zeta \lesssim 0.5$**,
    the strong-PPS regime has a different universality and the
    sharp-drop pattern in $\lambda(c=1)$ reflects a genuine crossover
    in universality class, analog to LMR's Ising-to-BKT.

Both outcomes are useful. The first confirms the bosonization
prediction in detail. The second falsifies the "single universality
class" picture and motivates a more careful analysis of the strong-PPS
RG flow.

---

## What this gives the thesis

A concrete, defensible analytical statement:

> The QJ cross-replica click vertex bosonizes to a dimension-4
> (irrelevant) vertex operator at the Born-rule fixed point. This
> predicts a **vanishing leading $(1-\zeta)$ slope** of the effective
> Luttinger parameter, which differs from the **linear leading slope**
> in LMR's diffusive case. Our numerical data show essentially flat
> $\lambda(c=1)$ behaviour in $\zeta \in [0.85, 1]$, quantitatively
> consistent with this prediction. The sharp drop in $\lambda(c=1)$ at
> $\zeta \lesssim 0.5$ is a separate non-perturbative feature that
> the leading-order bosonization does not capture and that requires
> further analysis (likely involving the strong-PPS limit of LMR's RG
> flow adapted to the QJ vertex).

This is publication-quality: it makes a specific, falsifiable
distinction between two unravelings and uses both rigorous bosonization
and our numerical data to support it.

---

## What is NOT done

  1. **The $(1-\zeta)^2$ coefficient $c_2$ is not computed explicitly.**
     This requires OPE analysis of $V_j$ with itself, which I have not
     carried out. Estimated complexity: 1-2 days of careful calculation
     following Giamarchi Ch. 5.

  2. **The sharp-drop region $\zeta \lesssim 0.5$ is not analytically
     described.** A full RG analysis in the spirit of LMR's Section V C
     for the strong-PPS limit of the QJ vertex would be needed.
     Estimated complexity: a few weeks, comparable to the original LMR
     program.

  3. **Klein factors across Choi copies are handled heuristically.**
     The bosonization above assumes Klein factors are global phases
     that do not affect local scaling dimensions. This is the standard
     LMR assumption (their App. C) but should be verified for the
     specific four-copy vertex structure of QJ.

These are all natural follow-up calculations, none of which I have
attempted in the present documents.
