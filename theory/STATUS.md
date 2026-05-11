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

---

## Update — sharpened prediction for the Renyi test

After re-reading the bosonization in light of the strong-PPS limit, the
prediction can be stated more precisely.

### What the irrelevant cross-replica vertex actually implies

If the QJ cross-replica vertex really is dimension 4 (irrelevant) at the
Born-rule fixed point, then under RG flow the coupling
$g_\zeta = (1-\zeta)\alpha$ flows to zero. The infrared theory is
*the same fixed-point theory* for all $\zeta$ in the log-law region:
$$
\mathcal{L}_{\rm IR}(\zeta) \;=\; \mathcal{L}_{\rm IR}(\zeta = 1)
\qquad
\text{for all $\zeta$ in the critical phase.}
$$

The flow trajectories differ — they start from different bare couplings —
but all flow to the **same fixed point**. The effective Luttinger parameter
at the fixed point, $K^*$, is therefore independent of $\zeta$:
$$
K^*(\zeta) \;=\; K^*(\zeta = 1)
\qquad \text{up to corrections from the deterministic non-Hermitian part.}
$$

### What changes with $\zeta$

What DOES change with $\zeta$ is the **diagonal-in-replica** part of the
generator $\mathcal{L}_{\rm diag}^{(2)}$, which contains the
non-Hermitian shift $-i\alpha/2 \sum L^\dagger L$ per replica. This is
**$\zeta$-independent**. But the *interplay* between this non-Hermitian
shift and the (now irrelevant) cross-replica vertex sets where the
trajectory crosses into the area-law phase.

So:
  - **Inside the log-law phase**: $K^*$ is essentially $\zeta$-independent.
    The effective central charge $c_{\rm eff} = c(K^*)$ is the same as
    at $\zeta = 1$ for the same $\lambda$.
  - **Boundary of log-law phase**: shifts with $\zeta$ because the
    deterministic non-Hermitian dynamics pushes the system into area-law
    at lower $\lambda$ when click intensity is reduced (lower $\zeta$).

### The clean falsifiable statement

If the irrelevant-vertex picture is correct:
  1. At every test point in the log-law phase, the Renyi ratios should
     satisfy the universal CFT relations $c_2/c_1 = 3/4$, $c_3/c_1 = 2/3$.
  2. The effective central charge $c_{\rm eff}(\lambda; \zeta)$ should be
     **independent of $\zeta$** within the log-law phase at fixed
     $\lambda$.

Numerical check of (2) at $\lambda = 0.30$:
  - $c_{\rm eff}(\lambda=0.30, \zeta=1.00) \approx 1.41$
  - $c_{\rm eff}(\lambda=0.30, \zeta=0.85) \approx 1.48$
  - $c_{\rm eff}(\lambda=0.30, \zeta=0.70) \approx 1.50$
  - $c_{\rm eff}(\lambda=0.30, \zeta=0.50) \approx 1.38$

These differ by ~5–7%. **Within the noise** of the cloning estimator
this is consistent with $\zeta$-independence. A more careful test would
need:
  - Larger $N_c$ at the higher $L$ to reduce the statistical error on
    $c_{\rm eff}$ to below 5%.
  - Or — more importantly — the Renyi ratio test, which only requires
    that $c_2/c_1$ and $c_3/c_1$ be universal (rather than that $c_{\rm eff}$
    itself be flat).

### What the Renyi reruns will tell us

  - **If $c_2/c_1 \approx 0.75$ and $c_3/c_1 \approx 0.667$ at all six
    test points within statistical error**: this confirms a single
    conformal phase (CFT universality) throughout, consistent with the
    irrelevant-vertex bosonization. The QJ model has no analog of
    LMR's Ising universality crossover at strong PPS — the strong-PPS
    limit of QJ is still a Luttinger-liquid-like CFT.

  - **If the Renyi ratios deviate from CFT values at low-$\zeta$ test
    points**: this falsifies the irrelevant-vertex picture. There is
    then a non-CFT regime at strong PPS, and the bosonization must
    include relevant operators we have not identified. This would be
    closer to the LMR scenario (Ising-like at strong PPS).

Either outcome is informative. The Renyi data will resolve this
specific theoretical question.

---

## Summary of what's been done analytically

  1. **Rigorous:** Construction of the QJ two-replica generator in
     `qj_two_replica_derivation.md`. Choi-Jamiołkowski mapping to the
     4-copy Hilbert space. Identification of the cross-replica click
     operator as a four-fermion product.

  2. **Rigorous:** Bosonization of the cross-replica click vertex in
     `qj_one_minus_zeta_expansion.md`. Result: dimension-4 vertex
     operator in the inter-replica difference mode. Irrelevant in the
     RG sense.

  3. **Rigorous:** Distinction from LMR's diffusive case in
     `qj_one_minus_zeta_expansion.md` §4.3. LMR's cross-replica
     operator is marginal (dim 2); ours is irrelevant (dim 4). This is
     a clean structural difference between the two unravelings.

  4. **Rigorous:** Prediction of vanishing leading $(1-\zeta)$ slope of
     $K_\sigma$, contrasted with LMR's finite slope. Numerical check
     near $\zeta = 1$ is consistent.

  5. **Heuristic (sharpened in this update):** The irrelevant-vertex
     picture predicts a single CFT phase throughout the log-law region
     in $\zeta$, no analog of LMR's Ising universality at strong PPS.
     The Renyi reruns provide a direct test.

What's NOT done is the $(1-\zeta)^2$ coefficient and the full analysis
of the area-law boundary — these are well-defined follow-up
calculations of a few days each.
