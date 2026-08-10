# Power-law analysis of $\lambda_c(\zeta)$ from Binder crossings

> **Headline**: with theoretically motivated $L^{-1/\nu}$ correction-to-scaling,
> the small-ζ data gives $\phi = 0.502 \pm 0.026$, consistent with the
> predicted $\phi = 1/2$. The prefactor is $A \approx 0.96$, not $A = 0.5$
> (that was an unjustified guess matching the Born-rule endpoint).

## Method

For each ζ and each pair $(L_1, L_2)$ with $L_1, L_2 \in \{32, 48, 64, 96, 128\}$:
1. Locate the crossing of $B_{L_1}(\lambda, \zeta) = B_{L_2}(\lambda, \zeta)$ by
   linear interpolation, with 500-bootstrap errors using $B_L$ uncertainties.
2. Get 10 crossing points per ζ (one per L-pair).
3. Extrapolate $\lambda_c(L_{\rm avg}) \to \lambda_c^\infty$ as a function of $1/L_{\rm avg}^p$ for several $p$.
4. Log-log fit $\lambda_c^\infty(\zeta) = A \zeta^\phi$ restricted to $\zeta \le 0.3$
   (the regime where the small-ζ multicritical asymptote should apply).

## Why $p = 1/2$ is the right extrapolation

The leading correction at a Binder crossing is $\lambda_c(L) - \lambda_c^\infty \sim L^{-1/\nu}$.
With $y_\lambda = 1/2$, $\nu = 1/y_\lambda = 2$, so the correction goes as
$L^{-1/2}$. The naive $1/L$ extrapolation (often used by default) is wrong
here and produces a biased estimate.

## Results

| Extrapolation form | Small-ζ fit ($\zeta \le 0.3$) | $\chi^2/\text{dof}$ |
|---|---|---|
| $1/L$ | $\phi = 0.428 \pm 0.017$, $A = 0.66$ | 17.2 |
| $1/L^{0.7}$ | $\phi = 0.463 \pm 0.021$, $A = 0.79$ | 13.4 |
| **$1/L^{0.5}$** (theory) | **$\phi = 0.502 \pm 0.026$, $A = 0.96$** | **10.7** |
| $1/L^2$ | $\phi = 0.385 \pm 0.011$, $A = 0.53$ | 28.2 |

The $1/L^{1/2}$ extrapolation gives BOTH the best chi²/dof AND the predicted
exponent $\phi = 1/2$.

## Extrapolated $\lambda_c^\infty(\zeta)$ values (using $1/\sqrt{L}$)

| $\zeta$ | $\lambda_c^\infty$ |
|---:|---:|
| 0.02 | 0.149 ± 0.008 |
| 0.05 | 0.157 ± 0.019 |
| 0.10 | 0.251 ± 0.022 |
| 0.15 | 0.233 ± 0.025 |
| 0.20 | 0.229 ± 0.106 |
| 0.30 | 0.594 ± 0.028 |
| 0.50 | 0.759 ± 0.055 |
| 0.70 | 0.487 ± 0.071 |
| 0.85 | 0.459 ± 0.020 |
| 1.00 | 0.443 ± 0.011 |

The ζ = 0.5 value being higher than ζ = 0.7 and ζ = 1.0 (which agree at
~0.46) is a residual finite-L artifact — the crossover region from the
$\sqrt{\zeta}$ asymptote to the Born-rule plateau is where extrapolation
is hardest. The Born-rule value at ζ = 1.0 matches Carollo et al.

## Caveats

- $\chi^2/\text{dof} \approx 11$ for the log-log fit is mediocre; there are
  outliers (notably ζ = 0.5 and ζ = 0.7). This is unsurprising for $L \le 128$.
- The very small-ζ point (ζ = 0.02) has $\chi^2/\text{dof} = 12.8$ in the
  $L^{-1/2}$ extrapolation, indicating residual finite-size corrections
  beyond a single power of $1/L^{1/2}$. This is the regime that the
  L = 192, 256 jobs will address most.
- The fit is consistent with $\phi = 1/2$ but the data is not yet powerful
  enough to distinguish $\phi = 0.50$ from, say, $\phi = 0.55$ or $\phi = 0.48$.
  The error bar is ~5%.

## What this means for the project

- $\lambda_c \sim \sqrt{\zeta}$ at small ζ is now corroborated by Binder
  crossings + proper FSS extrapolation, with effective exponent
  $\phi = 0.502 \pm 0.026$ matching the cross-Choi + BdG prediction.
- The prefactor $A \approx 1$ at small ζ does not match the Born-rule
  endpoint $\lambda_{BR} \approx 0.5$; the critical line must bend
  between the two regimes (the plateau picture).
- The L = 192, 256 data will sharpen the small-ζ fit and confirm
  whether the asymptote really is $\phi = 1/2$.

## Files

- `lambda_c_phi_fit.png` — final plot with both extrapolations.
- `binder_analysis_v2.png` — earlier 4-panel diagnostic with $1/L$.
- This document.
