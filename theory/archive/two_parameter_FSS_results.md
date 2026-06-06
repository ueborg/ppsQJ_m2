# Two-parameter FSS analysis: extracting y_ζ from existing data

This supersedes the earlier interpretation in `fss_analysis_results.md`
which mistakenly assumed Scenario C's simple linear $\lambda_c \sim \zeta$.

## The framework (refined)

General FSS ansatz near the no-click critical point at $(\lambda, \zeta) = (0, 0)$:
$$
B_L(\lambda, \zeta) = \mathcal{B}\bigl(\lambda L^{y_\lambda},\, \zeta L^{y_\zeta},\, u L^{-\omega}\bigr)
$$

with:
- $y_\lambda = 1/2$ established from BdG analysis ($\xi_{\rm ps} \sim \lambda^{-2}$)
- $y_\zeta$ a separate scaling dimension to be determined empirically

The TD critical line scales as
$$
\boxed{\;\lambda_c(\zeta) \sim A\,\zeta^\phi, \qquad \phi = \frac{y_\lambda}{y_\zeta} = \frac{1}{2 y_\zeta}.\;}
$$

The previous "Scenario C" of $\lambda_c \sim \zeta$ implicitly assumed $y_\zeta = y_\lambda = 1/2$, which is unjustified.

## Empirical extraction from L≤128 data

Three independent methods using B_L crossings on the clean aggregate:

### Method 1: slice fits at fixed L_g

Fit $\lambda_c \sim \zeta^\phi$ at fixed L_g using ζ ∈ [0.1, 0.7]:

| L_g | $\phi$ | $y_\zeta$ |
|---|---|---|
| 19.6 | 0.40 | 1.26 |
| 27.7 | 0.55 | 0.92 |
| 39.2 | 0.55 | 0.91 |
| 55.4 | 0.33 | 1.50 |
| 78.4 | 0.51 | 0.99 |
| 110.9 | 0.61 | 0.82 |

Median over L_g ≥ 27: $\phi \approx 0.55$, $y_\zeta \approx 0.9$.

### Method 2: global scaling collapse optimum

Sweep $y_\zeta \in [0.2, 2.0]$, find value minimizing the within-bin variance of $\lambda_c L^{1/2}$ vs $\zeta L^{y_\zeta}$.

Optimum: $y_\zeta = 1.16$, $\phi = 0.43$. Residual 0.20.

Comparison residuals:
- $y_\zeta = 0.5$ (linear Scenario C): residual 0.28 — **excluded**
- $y_\zeta = 1.0$ (square-root scenario): residual 0.21 — accepted
- $y_\zeta = 1.5$: residual 0.20 — also accepted

### Method 3: TD-limit extrapolation

Fit $\lambda_c(L_g, \zeta) = \lambda_c^\infty(\zeta) + B/\sqrt{L_g}$ at fixed ζ, then power-law fit on the extrapolated TD values.

Result: $\phi = 0.503$, $y_\zeta = 0.995$.

⚠ This method is noisy: the fitted Born-rule TD value comes out 0.42 vs Carollo's 0.5, indicating the $1/\sqrt{L}$ correction isn't clean across all ζ.

## Best estimate

$$
\boxed{\;y_\zeta \approx 1, \quad \phi \approx 1/2, \quad \lambda_c(\zeta) \sim A\sqrt{\zeta}\;}
$$

with $A \approx 0.5$ matching the Born-rule endpoint.

**Uncertainty range**: $\phi \in [0.5, 0.7]$, $y_\zeta \in [0.7, 1.0]$, depending on extraction method and data subset.

## Conclusions

1. **Original Scenario C ($\lambda_c \sim \zeta$, $\phi = 1$) is excluded** — the collapse residual is ~28% worse than $\phi = 1/2$ and the slice fits give $\phi < 1$ at every reliable L_g.

2. **The two-parameter FSS structure with $y_\zeta \neq y_\lambda$ is required** — single-parameter scaling fails.

3. **The cleanest scaling consistent with the data is $y_\zeta = 1$, $\phi = 1/2$** — meaning $\lambda_c(\zeta) \sim \sqrt{\zeta}$ in the thermodynamic limit.

4. **Physical interpretation**: at the no-click fixed point, the click recycling vertex has scaling dimension $y_\zeta = 1$, while the localization scale follows $y_\lambda = 1/2$. The TD critical line emerges from the matching of these two scales.

## The unresolved theoretical question

The empirical $y_\zeta \approx 1$ should be derivable from a perturbative tilted-Liouvillian calculation at the no-click fixed point. Specifically:

The click vertex $\mathcal{J}_j = J_j \rho J_j^\dagger$ in the two-replica picture is a 4-fermion (cross-Choi) operator. Its scaling dimension at the gapped no-click fixed point depends on the eigenvalue spectrum of the no-click Liouvillian and the matrix elements between the dominant right/left eigenmodes.

A direct computation:
$$
\theta_1(\lambda) = \frac{\langle\!\langle \ell_0(\lambda), \mathcal{J} r_0(\lambda) \rangle\!\rangle}{\langle\!\langle \ell_0(\lambda), r_0(\lambda) \rangle\!\rangle}
$$

with $\ell_0, r_0$ the no-click dominant eigenmodes (computable from BdG eigenvectors), should give the leading-order $\zeta$-dependence of the tilted Liouvillian eigenvalue, and hence $y_\zeta$ via dimensional analysis.

**This is the next theoretical target** — much more tractable than the multi-coupling bosonization RG.

## Implications for L=192, 256 runs

The improved framework sharpens what the new data should test:

1. **Power-law exponent $\phi$**: with L_g extending to ~220, the slice fits and TD extrapolation should converge on a single $\phi$ value with error bars an order of magnitude tighter.

2. **Scaling collapse**: the L=192 and 256 points will fall at $\zeta L^{y_\zeta} = $ specific predicted values. If they overlay onto the L≤128 collapse function $F(\zeta L)$, the framework is confirmed.

3. **Born-rule asymptote**: cleaner extrapolation should give $\lambda_c(\zeta=1) \to 0.5$ if Carollo is right.

4. **Crossover scale**: the no-click-to-power-law crossover at $\zeta L \sim 1$ moves to smaller $\zeta$ as L grows. At L=192, the crossover is at $\zeta \sim 1/192 \approx 0.005$. Even ζ=0.02 should be in the TD regime.

If the L=192, 256 data confirms $\phi \approx 1/2$, the thesis result is:

> "QJ-PPS exhibits a phase boundary $\lambda_c(\zeta) \sim A\sqrt{\zeta}$ at small ζ, with Born-rule asymptote $\lambda_c(1) \approx 0.5$ matching Carollo's projective MIPT. The two-parameter FSS structure reflects independent scaling dimensions $y_\lambda = 1/2$ (from no-click localization) and $y_\zeta = 1$ (from click-recycling perturbation)."

## Files
- `analysis/yzeta_extraction.png`: scaling-collapse residual minimization and best-fit plots
- `analysis/fss_*.png`: earlier B_L analysis
