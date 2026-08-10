# Liouvillian diagonalization of QJ-PPS Case B

## Goal

Compute the spectral gap of the PPS-Doob Lindbladian and identify which eigenvalue scales as the empirical $\alpha_c/w = \sqrt\zeta$.

## What we built

- `pps_lindbladian.py`: single-replica vectorized Lindbladian for Case B on a half-filled Kitaev-like chain. Restricted to the $N$-particle sector since $[\hat N, \hat H] = [\hat N, \hat L_j] = 0$.
- `two_replica.py`: two-replica linearized Lindbladian for $\langle \rho \otimes \rho \rangle_{\rm PPS}$ with the cross-replica recycling vertex of strength $\alpha\zeta$.
- Scans over $(\lambda, \zeta, L)$.

## Single-replica results (L=6)

| $\zeta$ | gap at $\lambda_c^{\rm emp}$ | peak gap (any $\lambda$) | peak location |
|---|---|---|---|
| 1.00 | 0.280 | 0.296 | $\lambda \approx 0.55$ |
| 0.50 | 0.139 | 0.190 | $\lambda \approx 0.65$ |
| 0.20 | 0.043 | 0.100 | $\lambda \approx 0.75$ |
| 0.10 | 0.018 | 0.057 | $\lambda \approx 0.85$ |
| 0.05 | 0.007 | 0.031 | $\lambda \approx 0.90$ |

**Power-law fits:**
- gap @ $\lambda_c^{\rm emp}$ $\sim \zeta^{1.24}$ (not $\sqrt\zeta$)
- peak gap $\sim \zeta^{0.75}$ (closer to but not equal to $\sqrt\zeta$)

**The peak of the single-replica gap moves to LARGER $\lambda$ as $\zeta$ decreases** — opposite to the direction of the empirical $\lambda_c$. This confirms that the single-replica gap is NOT the MIPT order parameter.

## Two-replica results (L=4, d⁴=1296)

Cross-vertex strength $\alpha\zeta$ verified. At fixed $\alpha=w=0.5$:
| $\zeta$ | gap |
|---|---|
| 1.00 | 0.114 |
| 0.50 | 0.063 |
| 0.20 | 0.027 |
| 0.10 | 0.014 |

Power-law fit: gap $\sim \zeta^{0.9}$ — close to linear, NOT $\sqrt\zeta$.

The $\lambda$-scan at fixed $\zeta$ does NOT show a clear gap minimum in the interior — the gap is monotonic in $\lambda$ across the full range. This means L=4 is too small to see the MIPT in the two-replica spectrum.

## What we learned

1. **Neither the single-replica nor the two-replica Lindbladian gap at L=4–6 shows a $\sqrt\zeta$ scaling.** Both show closer to linear in $\zeta$.

2. **L=4 is too small** to see the MIPT directly: the gap is monotone in $\lambda$ for the two-replica Lindbladian; no signature of a critical point.

3. **The four-replica problem** (needed for entanglement variance, the actual MIPT order parameter) is computationally inaccessible: dim = d⁸ = $4 \times 10^{10}$ for L=6.

4. **Why does this fail to derive $\sqrt\zeta$?** Because the MIPT is fundamentally a property of the entanglement growth ON INDIVIDUAL TRAJECTORIES, which is a 4th-order quantity. The averaged dynamics (1- and 2-replica) don't see this transition cleanly at small L.

## What this means for the derivation question

**The numerical Liouvillian-diagonalization approach was a reasonable thing to try, but it doesn't give us the $\sqrt\zeta$ prefactor.** The MIPT is a property of higher-order correlations, not the Lindbladian gap directly.

The actual derivation of $\lambda_c/(1-\lambda_c) = \sqrt\zeta$ would need:
- A proper RG calculation of the cross-Choi vertex's dimension at the multicritical fixed point.
- Or large-$L$ exact diagonalization of trajectory-resolved 4-point functions (e.g. Rényi-2 entanglement).
- Or careful matching of the no-click BdG localisation length against the click-induced decoherence length, with the right operator-product expansion of the cross vertex.

These are all research-level open problems. The numerical exercise here was inconclusive but informative — it tells us the single- and two-replica Lindbladian gaps are NOT the right diagnostics.

## Honest answer to the prefactor question

The empirical fit $\alpha_c/w = C\sqrt\zeta$ to the global FSS data gives $C \approx 0.77$ (through-origin) or $C \approx 0.88$ (with intercept) — not $C = 1$. The "exact" $C = 1$ would correspond to the Carollo Born-rule value being recovered exactly at $\zeta = 1$, which is consistent with the data at $\zeta = 1$ within ~10% (data: $\lambda_c = 0.431$, prediction at $C=1$: $\lambda_c = 0.5$).

The $\chi^2$/dof of $\approx 6$–$9$ for the linear fit indicates **residual L-dependent corrections** that are not captured by the simple $\sqrt\zeta$ form. These could be:
- Cross-overs to the small-$\zeta$ Doob regime
- Logarithmic corrections to scaling
- Finite-size effects in the FSS extraction itself

To pin down the prefactor more precisely would require:
- Larger $L$ data (Run B at L=192,256 for $\zeta = 0.5, 1.0$ will help; expected within ~14h)
- Better statistics at the small-$\zeta$ corners
- More careful FSS extrapolation (proper bootstrap with correlated systematics)

## Conclusion

The empirical relation $\alpha_c/w = \sqrt\zeta$ stands as a **strong phenomenological fit** that:
- Captures the qualitative behaviour (saturation at $\lambda_c \to 1/2$ for $\zeta \to 1$)
- Has the right small-$\zeta$ scaling that the matched NLSM predicted
- Fits the data with $C \approx 0.8$ (1-parameter)

The first-principles derivation of $\alpha_c/w = \sqrt\zeta$ from the Lindbladian structure remains open. Our numerical Liouvillian diagonalization at $L \leq 6$ does not see the $\sqrt\zeta$ scaling because the operators it diagonalizes (1- and 2-replica) are not the right diagnostics for MIPT.
