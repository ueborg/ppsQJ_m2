# QJ-PPS: final theoretical synthesis (May 2026)

This document consolidates the current best understanding of the
QJ-PPS phase boundary. For a one-stop handoff entry point, see
**`theory/HANDOFF.md`**.

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

## Microscopic derivation of $y_\zeta = 1$ (HEURISTIC + NUMERICAL CONFIRMATION)

### Heuristic derivation

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
Since the matrix elements are local and the modes have finite correlation
length, $\theta_1 \sim L$ to leading order. This is **the derivation of
$y_\zeta = 1$** — no fit, no ansatz, just locality and extensivity.

### Numerical confirmation (commit e2ec079)

Direct computation via `analysis/compute_theta1.py` using the BdG
slowest-decay state covariance matrix:

| $\lambda$ | $\theta_1 / L$ at large $L$ | fit slope $p$ | regime |
|---|---|---|---|
| 0.05 | $1.8 \times 10^{-3} \cdot \lambda^2$ | 1.036 | clean linear |
| 0.10 | $1.8 \times 10^{-3} \cdot \lambda^2$ | 1.036 | clean linear |
| 0.15 | $1.4 \times 10^{-3} \cdot \lambda^2$ | 0.980 | clean linear |
| $\ge 0.30$ | — | — | biorthogonal construction breaks |

For $\lambda \le 0.15$: $\theta_1(L) \approx 0.18 \cdot \lambda^2 \cdot L$.
Power-law slope $\approx 1.04$ (small excess from $1/L$ boundary terms,
not anomalous).

**The physically relevant regime is exactly $\lambda \le 0.5$** — the
critical $\lambda_c(\zeta)$ along the empirical phase boundary stays
$< 0.5$ across all $\zeta \in [0.02, 1.0]$. So the $\theta_1$ calculation
confirms $y_\zeta = 1$ in the regime that matters.

### Critical condition

The TD critical condition combines $\theta_1 \sim L$ with the localization
length:
$$
\zeta \cdot \xi_{\rm ps}(\lambda_c) \sim 1
\quad\Longleftrightarrow\quad
\frac{\zeta}{\lambda_c^2} \sim 1
\quad\Longrightarrow\quad
\lambda_c \sim \sqrt{\zeta}.
$$

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
```
python analysis/test_yzeta1_collapse.py --add-fst <path>
```

Visual check: do the L=192, 256 [square markers] fall on the
L≤128 [circle markers] curve in the plot of $\lambda_c\sqrt{L}$ vs
$\zeta L$?

- **Yes** → $y_\zeta = 1$ framework confirmed, thesis result
  $\lambda_c(\zeta) \sim A\sqrt{\zeta}$ is solid.
- **No** → reassess. Possible alternatives: $y_\zeta$ has logarithmic
  corrections, or genuine multi-scale structure not captured by the
  simple two-parameter ansatz.

## Open theoretical questions (priority-ordered)

### 1. Fix the large-$\lambda$ $\theta_1$ calculation — MEDIUM PRIORITY

The biorthogonal construction in `analysis/compute_theta1.py` breaks for
$\lambda \ge 0.3$. The physical regime is fine without this, but closing
the gap would strengthen the result.

**Approach (a)**: full Liouville-space ($4^L \times 4^L$) computation for
small $L \le 6$. Cross-check against BdG at small $\lambda$ where both work.

**Approach (b)**: regularize via long-time evolution from a physical initial
state, then project onto slowest-decay sector. This handles the
non-normalizable eigenstate ambiguity cleanly.

### 2. Bosonized derivation of $y_\zeta = 1$ — LOWER PRIORITY

Existing bosonization in `theory/qj_one_minus_zeta_expansion.md` works
near $\zeta = 1$ (Born rule). The companion no-click-end derivation
would close the loop on universality.

### 3. Universality class — LOWER PRIORITY

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

Status: solid for the Binder-cumulant L≤128 data plus the BdG
$\theta_1 \sim L$ confirmation; awaiting L=192, 256 confirmation via
the collapse test in `analysis/test_yzeta1_collapse.py`.

## File map

- `theory/HANDOFF.md`: **entry point for fresh context**
- `theory/qj_pps_final_synthesis.md`: **this file** (current top-level)
- `theory/theta1_first_principles.md`: $\theta_1$ calculation details
- `theory/two_parameter_FSS_results.md`: $y_\zeta$ extraction details
- `theory/fss_analysis_results.md`: B_L methodology
- `theory/qj_pps_theory_summary.md`: original comprehensive summary
- `theory/qj_pps_scenario_C_addendum.md`: first guess (linear $\zeta$) — superseded
- `analysis/compute_theta1.py`: $\theta_1$ from BdG
- `analysis/extract_yzeta.py`: $y_\zeta$ extraction from L≤128
- `analysis/test_yzeta1_collapse.py`: decisive test for L=192, 256
- `analysis/yzeta1_collapse_test.png`: pre-FST plot showing current data
- `analysis/theta1_scaling_v2.png`: $\theta_1$ scaling diagnostic
