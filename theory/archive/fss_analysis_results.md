# FSS analysis on L≤128 data: empirical verdict (2026-05-18)

This document supersedes the optimistic interpretation in
`qj_pps_scenario_C_addendum.md`. After running the FSS test with both
c_eff and B_L (Binder cumulant) extraction methods, the picture is
sharper than expected — and **Scenario A is empirically falsified**
while Scenario C is qualitatively supported but quantitatively imperfect.

## Methodology check first: c_eff fails, B_L works

The c_eff threshold method (c_eff = 1 crossing) returned **zero usable
data points** — c_eff in this dataset has median ~3.4 and never drops
below 1.46. The L-pair c_eff crossings happen at c_eff values of 6-8,
not at any universal critical value, meaning those crossings are
**not at the actual MIPT** but somewhere in the volume-law/log-law
boundary.

The B_L (Binder cumulant) method gives clean L-pair crossings. **All
analysis below uses B_L.**

## Born-rule consistency check

B_L L-pair crossings at ζ=1.0:

| pair | λ_c |
|---|---|
| (8,16) | 0.527 |
| (16,24) | 0.531 |
| (24,32) | 0.524 |
| (32,48) | 0.499 |
| (48,64) | 0.479 |
| (64,96) | 0.482 |
| (96,128) | 0.467 |

Mean = 0.497. **Matches Carollo's λ_c(ζ=1) ≈ 0.5 essentially exactly.**
The methodology is validated.

## Small-ζ behavior (ζ=0.02): the decisive test

| L_g | λ_c |
|---|---|
| 19.6 | 0.185 |
| 27.7 | 0.134 |
| 39.2 | 0.124 |
| 55.4 | 0.060 |
| 78.4 | 0.050 |
| 110.9 | 0.139 (noisy, N_c=100) |

Excluding the noisy L=128 endpoint, **λ_c monotonically decreases
with L by a factor of ~4 across the reliable size range.**

This **rules out Scenario A** (λ_c → 0.5 robust for all ζ > 0), which
would require λ_c to drift upward toward 0.5.

This is **qualitatively consistent with Scenario C** (λ_c → 0 as ζ → 0).

## Quantitative test of Scenario C scaling

Scenario C predicts λ_c ∼ 1/√L in the no-click crossover regime:
predicted ratio λ_c(L_max)/λ_c(L_min) = √(L_g_min/L_g_max) = √(20/78) ≈ 0.50.

Observed ratio: 0.27.

**The empirical decrease is faster than 1/√L.** Either:
1. The scaling exponent is closer to 1/L than 1/√L
2. The data is pre-asymptotic with higher-order corrections dominating
3. There is an additional mechanism not captured by the dimensional argument

## Comparison with the proposed closed form

Scenario C's interpolation λ_c = Cζ/(1+Cζ) with C=1 (fitted to match
Carollo at ζ=1) gives:

| ζ | observed λ_c (L_g=78) | predicted |
|---|---|---|
| 0.02 | 0.050 | 0.020 |
| 0.05 | 0.048 | 0.048 |
| 0.10 | 0.175 | 0.091 |
| 0.20 | 0.243 | 0.167 |
| 0.50 | 0.484 | 0.333 |
| 1.00 | 0.482 | 0.500 |

Born-rule end matches reasonably (0.482 vs 0.500). Small-ζ matches at
ζ=0.05 (coincidence?). Intermediate ζ consistently undershoots
prediction. **The simple rational interpolation is not the right form.**

## The "separatrix at ζ ≈ 0.143" was an artifact

The original separatrix identified in c_eff-based drift-sign analysis
**does not appear in the B_L analysis**. With B_L, the drift signs in
λ_c between successive L-pairs are noisy at small and intermediate ζ
with no clean transition point. The "ζ_c = 0.143" anchor used for
theory-building was an artifact of the wrong observable and should be
retired.

## Revised provisional phase diagram

- **ζ = 1 (Born rule)**: λ_c ≈ 0.5 from Carollo + B_L analysis. ✓
- **ζ → 0**: λ_c → 0, with empirical scaling faster than 1/√L.
- **Intermediate ζ**: smooth crossover, no sharp transition at any
  particular ζ value.

Working ansatz (revised, looser than Scenario C's closed form):
$\lambda_c(\zeta) = $ some smooth function that vanishes at ζ=0 and
reaches ~0.5 at ζ=1, with finite-size corrections of unclear
functional form.

## Implications for L=192, 256 runs

The question they answer has sharpened:
- **Not** "is there a finite-ζ separatrix?" (probably no, was artifact)
- **Not** "does λ_c → 0.5 for all ζ > 0?" (no, ruled out)
- **Yes**: "does λ_c continue its decrease toward 0 at small ζ as L
  grows, and with what scaling exponent?"

If at L=192,256 the small-ζ λ_c values continue decreasing toward 0,
Scenario C's qualitative claim is confirmed. The exact functional
form will require fitting once enough sizes are in.

If λ_c stabilizes or reverses, neither Scenario A, B, nor C is right
and a new framework is needed.

## Implications for the thesis framing

Replace the "Ising universality with λ_c ≈ 0.5 robust to PPS" framing
with: **"QJ-PPS has a smooth ζ-dependent critical line λ_c(ζ), with
λ_c → 0 as ζ → 0 (consistent with click-recycling competition with
no-click localization) and λ_c(ζ=1) ≈ 0.5 (Born-rule Carollo limit
recovered)."**

This is honest about what the data shows and avoids overclaiming a
clean universality result.

## Files

- `analysis/fss_collapse_test.png`: original c_eff plot (negative result)
- `analysis/fss_diagnostic.png`: diagnostic of why c_eff fails
- `analysis/fss_binder.png`: B_L-based analysis
- `analysis/fss_final.png`: comprehensive final plot
- `analysis/fss_collapse_data.txt`: raw tables for both methods
