# Addendum to theory summary: Scenario C — click-recycling

This addendum responds to a perturbative scaling analysis performed by an
external AI agent (using the comprehensive theory summary as input).  The
analysis is structurally sound and proposes a sharp testable prediction
that differs from Scenarios A and B in the main summary.

## The proposed mechanism

Using the structural decomposition

$$
\mathcal{L}_\zeta = \mathcal{L}_0 + \zeta\mathcal{J}
$$

(no-click non-Hermitian fixed point plus dilute click recycling), the
small-$\zeta$ phase boundary is set by the dimensional competition between:

- the no-click gap $m_{\rm ps} \sim v_F/\xi_{\rm ps} \sim \alpha^2/w$, and
- the click vertex strength $g_{\rm click} \sim \zeta\alpha$.

The dimensionless competition parameter is

$$
\mathcal{R}(\lambda, \zeta) = \frac{g_{\rm click}}{m_{\rm ps}} \sim \frac{\zeta\,w}{\alpha} = \zeta\,\frac{1-\lambda}{\lambda}.
$$

Criticality at $\mathcal{R} \sim 1$ gives

$$
\boxed{\;\lambda_c(\zeta) \sim \frac{C\zeta}{1 + C\zeta}, \qquad \lambda_c \sim C\zeta\text{ for small }\zeta.\;}
$$

Setting $C = 1$ to match Carollo's $\lambda_c(\zeta = 1) \approx 0.5$:

| $\zeta$ | predicted $\lambda_c^\infty$ |
|---|---|
| 0.05 | 0.048 |
| 0.10 | 0.091 |
| 0.143 | 0.125 |
| 0.30 | 0.231 |
| 0.50 | 0.333 |
| 1.00 | 0.500 |

## Differences from Scenarios A and B

- **Scenario A** (LMR-Ising persistence): $\lambda_c^\infty = 0.5$ for all $\zeta > 0$.
- **Scenario B** (finite RG fixed point): genuine 2D phase diagram with separatrix at $\zeta_c \approx 0.143$.
- **Scenario C** (click-recycling): $\lambda_c^\infty(\zeta) \to 0$ linearly as $\zeta \to 0$; no finite-$\zeta_c$ separatrix; recovers $\lambda_c(1) = 1/2$.

Scenario C is essentially a refined version of Scenario B with an explicit
functional form for $\lambda_c(\zeta)$, but without a finite-$\zeta$ RG fixed
point.

## Finite-size scaling collapse prediction

The combined no-click crossover and TD limit give

$$
\lambda_c(L, \zeta) \sqrt{L} = F(\zeta\sqrt{L})
$$

with $F(x \to 0) \to C_0$ (no-click crossover) and $F(x \to \infty) \to C_1\,x$
(TD click-recycling).

The crossover scale $x^* \sim O(1)$ sets the apparent finite-$\zeta$ separatrix in
finite-$L$ data: $\zeta^*(L) \sim 1/\sqrt{L_{\rm eff}}$.

## Consistency with existing data

The empirical drift-zero at $\zeta \approx 0.143$:
- $(L_1, L_2) = (32, 64)$: $L_{\rm eff} \approx 45$, $\zeta\sqrt{L_{\rm eff}} \approx 0.96$.
- $(L_1, L_2) = (64, 128)$: $L_{\rm eff} \approx 90$, $\zeta\sqrt{L_{\rm eff}} \approx 1.36$.

Both are $O(1)$ — consistent with the FSS prediction that the apparent
separatrix is at the no-click-to-linear-click crossover scale, not a true RG
fixed point.

Under Scenario C, the drift-zero $\zeta$ should DRIFT with $L$:
- $(L_1, L_2) = (128, 192)$: $L_{\rm eff} \approx 156$, predicted drift-zero at $\zeta \approx 0.08$
- $(L_1, L_2) = (192, 256)$: $L_{\rm eff} \approx 222$, predicted drift-zero at $\zeta \approx 0.067$

## Empirical tests

1. **Scaling collapse on existing data** (immediate, ~1 hour):
   run `analysis/test_fss_collapse.py`. Plot $\lambda_c\sqrt{L}$ vs
   $\zeta\sqrt{L}$ for all $L \le 128$. If curves overlay, Scenario C
   supported. If they don't, look elsewhere.

2. **Drift-zero shift at L = 192, 256** (after FST runs complete):
   - Scenario A: drift-zero $\zeta$ drifts toward 0 or 1.
   - Scenario B: drift-zero pinned at $\zeta \approx 0.143$.
   - Scenario C: drift-zero drifts toward smaller $\zeta$ as $\sim 1/\sqrt{L_{\rm eff}}$.

3. **Perturbative computation of $\theta_1$**: compute the dominant
   eigenvalue shift $\theta_1 = \langle\!\langle \ell_0, \mathcal{J} r_0\rangle\!\rangle / \langle\!\langle \ell_0, r_0\rangle\!\rangle$
   from no-click BdG eigenvectors. Gives the susceptibility
   $\chi_{\rm click}(\lambda)$ in the scaling argument; tests whether
   $\chi_{\rm click}(0) \ne 0$ (predicting linear $\lambda_c \sim \zeta$).

## Caveats and uncertainties

1. **The dimensional argument is not rigorous.** Multiplicative constants
   are dropped; the functional form $\lambda_c = C\zeta/(1+C\zeta)$ is an
   interpolation, not a derivation.

2. **The $C = 1$ choice** to match $\lambda_c(\zeta=1) = 0.5$ is fitted to
   data, not derived. The argument predicts linear $\lambda_c \sim C\zeta$
   with $C = O(1)$; the precise value of $C$ requires computing OPE
   coefficients or doing perturbation theory in $\zeta$.

3. **The assumption $\chi_{\rm click}(\lambda \to 0) \ne 0$** is plausible
   from the non-vanishing form factor of $d_j$ but is not proven. If
   $\chi_{\rm click}(\lambda) \sim \lambda^p$ with $p > 0$, then
   $\lambda_c^{1-p} \sim \zeta$ — different scaling.

4. **The competition argument balances scales but does not derive the
   universality class.** Whether the transition is Ising, BKT, or
   something else is not addressed.

## Status

This is a working hypothesis. The single most efficient next step is the
FSS collapse test on existing data. If the collapse holds, Scenario C is
strongly supported and should be the leading framing for the thesis. If it
fails, return to Scenarios A and B and the L = 192, 256 runs become more
informative.

The relevant script is `analysis/test_fss_collapse.py`.
