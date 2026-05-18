# QJ-PPS: final theoretical synthesis (May 2026)

This document consolidates the current best understanding of the
QJ-PPS phase boundary. It supersedes earlier theory files
chronologically while preserving the analytical chain.

## The one-paragraph summary

The QJ-PPS chain has a non-Hermitian no-click fixed point with
localization length $\xi_{\rm ps} \sim \lambda^{-2}$ (BdG analysis,
single-particle). Postselection introduces clicks as a perturbation
on top of this. The perturbation is a **defect fugacity per
localization volume**: criticality occurs when one expected click
fits inside one localization region, $\zeta \xi_{\rm ps} \sim 1$,
giving the thermodynamic phase boundary
$$
\boxed{\;\lambda_c(\zeta) \sim A\sqrt{\zeta}, \qquad A \approx 0.5\;}
$$
with Born-rule endpoint $\lambda_c(1) \approx 0.5$ matching Carollo.
The finite-size scaling is two-parameter with $y_\lambda = 1/2$ from
localization and $y_\zeta = 1$ from extensivity of the click vertex.

## Microscopic derivation of $y_\zeta = 1$

The click vertex in the two-replica generator is
$$
\mathcal{G}_{\rm click} = \zeta\alpha \sum_{j=1}^{L-1} d_j^{(+a)} d_j^{*(-a)}
$$
a sum over bonds. At the no-click fixed point with finite $\xi_{\rm ps}$,
correlations decay exponentially. First-order perturbation theory in
$\zeta$ gives
$$
\theta_1(\lambda, L) = \sum_j \frac{\langle\!\langle \ell_0 | d_j^{(+a)} d_j^{*(-a)} | r_0 \rangle\!\rangle}{\langle\!\langle \ell_0 | r_0 \rangle\!\rangle}
$$
with $\ell_0, r_0$ the dominant left/right modes of $\mathcal{L}_0^\dagger$.
Since the matrix elements are local and the modes have finite
correlation length, $\theta_1 \sim L$ to leading order. This is **the
derivation of $y_\zeta = 1$** — no fit, no ansatz, just locality and
extensivity.

The TD critical condition is then
$$
\frac{|\theta_1(\lambda_c)|}{L} \cdot \zeta \sim O(1)
$$
i.e., $\zeta \xi_{\rm ps}(\lambda_c) \sim 1$, giving $\lambda_c \sim \sqrt{\zeta}$.

## Two-parameter FSS

The general scaling ansatz near $(\lambda, \zeta) = (0, 0)$:
$$
B_L(\lambda, \zeta) = \mathcal{B}\bigl(\lambda L^{1/2},\, \zeta L,\, u L^{-\omega}\bigr)
$$

The finite-size critical line:
$$
\lambda_c(L, \zeta) = L^{-1/2}\, F(\zeta L) + L^{-1/2-\omega}\, G(\zeta L) + \cdots
$$

In the TD limit:
$$
F(x) \sim \begin{cases} C_0 & x \to 0\,, \text{ (no-click crossover)} \\ A\sqrt{x} & x \to \infty\,, \text{ (TD power law)} \end{cases}
$$

The crossover scale where these two regimes meet: $x \sim O(1)$, i.e.,
$\zeta L \sim O(1)$.

## Empirical status

From the L≤128 aggregate (B_L crossings):

- $y_\lambda = 1/2$: established from BdG (independent of data)
- $y_\zeta$: empirical extraction gives $0.7 \le y_\zeta \le 1.2$
  across methods; best estimate $y_\zeta \approx 1$, $\phi \approx 1/2$
- Born-rule asymptote: $\lambda_c(\zeta=1) \approx 0.50$ from B_L
  crossings, matching Carollo
- Original linear Scenario C ($\phi = 1$, $y_\zeta = 1/2$):
  **excluded** (collapse residual 28% worse than $\phi = 1/2$)
- Original Scenario A ($\lambda_c = 0.5$ for all $\zeta > 0$):
  **excluded** (small-ζ data decreases with L)

## The decisive empirical test (L=192, 256)

When the FST data arrives, run
$$
\texttt{python analysis/test\_yzeta1\_collapse.py --add-fst <path>}
$$

Visual check: do the L=192, 256 [square markers] fall on the
L≤128 [circle markers] curve in the plot of $\lambda_c\sqrt{L}$ vs
$\zeta L$?

- **Yes** → $y_\zeta = 1$ framework confirmed, thesis result
  $\lambda_c(\zeta) \sim A\sqrt{\zeta}$ is solid.
- **No** → reassess. Possible alternatives: $y_\zeta$ has logarithmic
  corrections, or genuine multi-scale structure not captured by the
  simple two-parameter ansatz.

## Open theoretical questions (priority-ordered)

### 1. Direct numerical computation of $\theta_1(\lambda, L)$ — HIGH PRIORITY

This is the simplest non-trivial numerical check. From the no-click
BdG framework:

a. Construct $\mathcal{L}_0$ as a matrix on the two-replica Hilbert
   space (Choi vectorization).
b. Find dominant right/left eigenvectors $r_0, \ell_0$.
c. Compute the matrix element $\theta_1 = \langle\!\langle \ell_0 | \mathcal{J} | r_0 \rangle\!\rangle / \langle\!\langle \ell_0 | r_0 \rangle\!\rangle$.
d. Verify $\theta_1(\lambda, L) \sim L \cdot \mathcal{K}(\lambda\sqrt{L})$ — i.e., the
   FSS collapse with $y_\zeta = 1$.

**Parity caveat**: if non-degenerate PT vanishes by Choi parity,
extend to the near-degenerate manifold (synthesis's $K_{ab}$ matrix
formulation). But try non-degenerate first.

The no-click theory is Gaussian (free fermions under non-Hermitian
evolution), so the dominant modes have efficient representations.
This calculation is concrete and tractable.

### 2. Bosonized derivation of $y_\zeta = 1$

The bosonization framework laid out in earlier theory documents
should give the click-vertex scaling dimension at the no-click
fixed point. The no-click theory is gapped (localized), so the
relevant calculation is the operator content of the click vertex
in the dual lattice picture. This is a longer-term goal.

### 3. Universality class

The two-parameter scaling gives $\lambda_c(\zeta) \sim \sqrt{\zeta}$
but doesn't fix the universality class of the transition. Carollo
suggests Ising at $\zeta = 1$. Whether this persists or changes as
$\zeta \to 0$ requires direct analysis of critical exponents along
the line. Could be done from FSS fits of correlation functions
once L=192, 256 data is in.

## Thesis-level statement

The current state supports the following thesis result:

> "The QJ-PPS model exhibits a measurement-induced phase transition
> with a thermodynamic critical line $\lambda_c(\zeta) \sim A\sqrt{\zeta}$
> for small $\zeta$, reaching $\lambda_c(\zeta = 1) \approx 0.5$ at the
> Born-rule endpoint. The two-parameter finite-size scaling reflects
> independent scaling dimensions: $y_\lambda = 1/2$ from the
> non-Hermitian SSH-Majorana localization length $\xi_{\rm ps} \sim
> \lambda^{-2}$, and $y_\zeta = 1$ from the extensivity of click
> recycling over bonds. The dimensionless competition parameter is
> $\zeta \xi_{\rm ps}$ — the click fugacity per localization volume —
> with criticality at $\zeta \xi_{\rm ps} \sim 1$."

Status: solid for the Binder-cumulant L≤128 data; awaiting L=192, 256
confirmation via the collapse test in `analysis/test_yzeta1_collapse.py`.

## File map

- `theory/qj_pps_theory_summary.md`: original comprehensive summary
- `theory/qj_pps_scenario_C_addendum.md`: first guess (linear ζ) — superseded
- `theory/fss_analysis_results.md`: B_L methodology, rules out A — current
- `theory/two_parameter_FSS_results.md`: $y_\zeta \approx 1$ extraction — current
- `theory/qj_pps_final_synthesis.md`: **this file** (current top-level)
- `analysis/extract_yzeta.py`: $y_\zeta$ extraction from L≤128
- `analysis/test_yzeta1_collapse.py`: decisive test for L=192, 256
- `analysis/yzeta1_collapse_test.png`: pre-FST plot showing current data
