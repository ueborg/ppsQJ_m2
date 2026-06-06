# Numerics Status and Plan — QJ-PPS Case B

**Last update: 2026-06-03.** Companion to `CURRENT_THEORY_STATUS.md` and
`OPEN_ANALYTIC_PROBLEMS.md`. Built from a direct re-analysis of the cloning
aggregates (`clone_aggregate(1).pkl`, 1920 entries, L=8–128; plus the
sparse-but-high-L `aggregate_all.pkl`, L up to 256 at small ζ).

---

## 1. Why λ_c is a bad fitting variable, and r_c is better

λ_c saturates at the Born endpoint (λ_c→1/2 at ζ=1) and is compressed near 1/2,
so a power-law fit to λ_c is dominated by the saturation and reads a *low*
effective exponent. The natural RG variable is the physical ratio
$$r_c = \frac{\lambda_c}{1-\lambda_c} = \frac{\alpha_c}{w},$$
which carries the bare coupling and does not artificially saturate
(r_c(1)≈1). **Fit r_c, not λ_c.** Fitting λ_c is the main reason the earlier
"φ≈0.56" looks like √ζ.

## 2. Crossing table (independent re-extraction)

Binder (B_L) crossings, averaged over all available L-pairs from the clean grid
(16,32),(24,48),(32,64),(48,96),(64,128); error = spread across pairs.
**Provisional** (see drift/floor caveats below).

| ζ | λ_c | σ | r_c |
|---|---|---|---|
| 0.02 | 0.116 | 0.040 | 0.132 |
| 0.05 | 0.151 | 0.043 | 0.178 |
| 0.10 | 0.180 | 0.015 | 0.220 |
| 0.15 | 0.208 | 0.025 | 0.263 |
| 0.20 | 0.241 | 0.036 | 0.317 |
| 0.30 | 0.295 | 0.053 | 0.417 |
| 0.50 | 0.378 | 0.061 | 0.609 |
| 0.70 | 0.494 | 0.036 | 0.975 |
| 0.85 | 0.508 | 0.018 | 1.033 |
| 1.00 | 0.494 | 0.019 | 0.978 |

Born endpoint recovered: λ_c(1) ≈ 0.49. Figure:
`outputs/rc_scaling_analysis.png`.

## 3. What the fits say

Free power law r_c = A ζ^φ, sliding the window:

| window in ζ | φ (from r_c) |
|---|---|
| [0.02, 0.85] | 0.71 ± 0.05 |
| [0.10, 0.85] | 0.75 ± 0.05 |
| [0.15, 0.85] | 0.81 ± 0.08 |
| [0.15, 0.70] | 0.84 ± 0.13 |

- φ climbs as the small-ζ floor is excluded; cleanest window ⟹ **φ ≈ 0.8**.
- √ζ (φ=1/2) **overshoots** mid-range (predicts r_c(0.1)=0.31 vs 0.22);
  disfavored at ~3σ in [0.15,0.85].
- linear (φ=1) **undershoots** (predicts r_c(0.1)=0.10 vs 0.22); ~1.5–2.3σ.
- log form r_c = Aζ|log ζ|^p fits with p≈0.35–0.47 — better than either pure
  power, but not decisive.
- Endpoint interpolations on λ_c: √ζ/(1+√ζ) RMS 0.070 (overshoots);
  ζ/(1+ζ) RMS 0.065 (undershoots). Neither is a good fit.

**The "φ≈0.56" is an artifact of fitting λ_c through a saturating + floored
dataset.** The physical-ratio exponent is ≈0.7–0.85, between √ζ and linear,
closer to linear.

## 4. The two systematics that dominate

- **Finite-size drift.** Crossings drift *upward* with L, more strongly at
  large ζ (ζ=0.5: 0.32→0.39→0.47 across (16,32)→(32,64)→(64,128)). This biases
  small-L data toward looking more √ζ-like. The L=128 cloning B_L is visibly
  noisy near the crossing (ESS collapse; B_L rel-err ~13% at L=96, worse at 128).
- **Small-ζ floor.** r_c is nearly flat (~0.13–0.18) for ζ≤0.05, and the
  L=192/256 data is also flat (~0.18–0.22) across ζ=0.02–0.10. Almost certainly
  the finite-size limit (correlation length exceeds L). **The small-ζ
  asymptotics are not resolved by current data**, so √ζ-vs-linear-vs-floor at
  ζ→0 cannot be decided from what exists.

## 5. Recommended next run (priorities)

Do **not** chase larger sizes blindly. Concrete plan:

1. **ζ-grid:** concentrate where the physics is clean —
   ζ ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70}. This window
   fixes φ. Keep only a couple of ζ∈{0.05,0.08} points, used to *map the floor*
   and confirm it scales with L (the artifact test). **Do not invest in
   ζ<0.05** — it is finite-size-dominated and uninformative for the exponent.
2. **L-values:** the priority is **≥3 clean L at each ζ so the crossing can be
   L-extrapolated**, not one ragged large L. Prefer clean, high-N_c
   (32, 48, 64, 96) over noisy 128/192. L=192,256 are *not* needed for the
   exponent if the smaller sizes are clean; they only help chase the ζ→0 floor,
   which is likely out of cloning's reach anyway.
3. **λ-resolution:** ≥10 points within ±0.05 of each crossing (the crossing sits
   in the small-B_L tail where the slope is shallow and noise bites).
4. **Observables to store now (avoid reruns):** B_L, **CMI** (≈30% tighter than
   B_L at fixed N_c), entanglement **variance**, Rényi-2/3 (with
   PPS_RECORD_RENYI=1), and the full tripartite components. Extract crossings
   from B_L *and* CMI *and* variance, and require consistency — a tighter
   crossing observable is worth more than another decade of L.
5. **Uncertainty:** bootstrap over clones per (L,ζ,λ); propagate to the crossing;
   report the **L→∞ extrapolated** r_c with its extrapolation error, not a
   single-pair crossing.
6. **Pairwise vs global:** use pairwise crossings + explicit 1/L (or 1/√L)
   extrapolation as primary — it is transparent about the drift. A global FSS
   collapse *assumes* a λ_c(ζ) form and a single ν, which is exactly what is in
   question; run it as a cross-check and, if it disagrees, the collapse is hiding
   the drift. **Critically: a global fit with the exponent fixed (e.g. r_c=C√ζ)
   does not test the exponent — always do a free-exponent fit.**

## 6. Decisive outcomes

- L→∞-extrapolated r_c over ζ∈[0.10,0.70], free power-law fit:
  - stable φ ≈ 0.5 ⟹ √ζ vindicated (ν=2);
  - stable φ ≈ 0.8–1.0 ⟹ √ζ falsified; boundary near-linear/intermediate (ν≈1);
  - φ that keeps drifting with the window even after L-extrapolation ⟹
    marginal/log-corrected, no clean power.
- Independent: compare ξ_nc(λ) (from `noclick_spectrum_probe.py`) to the FSS
  collapse length at matched (λ,ζ). If they agree, ν=1 (linear). If the FSS
  length is parametrically larger (~λ⁻²), ν=2 (√ζ).

## 7. Currently-running campaign (actual, 2026-06-03) and assessment

| L | Coverage | Source | N_c |
|---|---|---|---|
| 8,16,24,32 | complete: 21 ζ × dense λ | pps_clone_dense | 1000–4000 |
| 48 | complete: 21 ζ × dense λ | pps_clone_dense | 600 |
| 64 | complete after backfill | dense + L64 backfill | 400 |
| 96 | partial ~380–450/514 (large-ζ missing) | pps_clone_dense | 450 |
| 128 | narrow: 10 ζ × 13 λ on measured crossings | pps_clone_rescue | 250 |
| 160 (opt) | narrow: 10 ζ × 13 λ | pps_clone_rescue | 120 |

**Is this enough to distinguish √ζ / ζ / ζ|logζ|^p?** The decisive asset is the
**five complete clean sizes L∈{16,24,32,48,64}** at every ζ (treat L=8 as a
check; L=48,96 carry OBC Friedel oscillations — use with care). That is enough
for a proper **crossing L-extrapolation** at each ζ, which removes the
single-pair drift and partly the floor — what single-pair crossings cannot. So
YES for the mid-ζ window, *provided you extrapolate* rather than quote one pair.
96/128 extend reach but are noisier; the binding limitation is small-ζ (floor),
which no size here removes.

**Q1 — 128 narrow-window: good or dangerous?** Partly dangerous. Windows
(±0.07) centered on L≤96 crossings, but crossings drift **up** with L (ζ=0.5:
0.32→0.39→0.47). For ζ≳0.3 the L=128 crossing may sit at/outside the window
edge. Fix: for ζ≥0.3 shift the L=128 center **up by ~+0.05–0.10** (or widen to
±0.10). Small/mid-ζ centers are fine.

**Q2 — L=160 (N_c=120) worth it?** No / low priority. ~20%+ B_L error (ESS
collapse); one noisy point won't sharpen φ. Redirect that compute to **more N_c
at L=96/128** or denser λ. Only run L=160 if L=128 returns clean and you want a
4th extrapolation node.

**Q3 — decisive ζ region.** Confirmed **ζ∈[0.10,0.70]**, most discriminating
sub-window **ζ∈[0.10,0.40]** (√ζ vs linear diverge most while resolvable).
ζ<0.05 is floored (skip for the exponent); ζ>0.70 saturates toward Born.

**Q4 — λ windows (≥10 λ each; for L≥96 shift centers up ~+0.03–0.07):**
ζ=0.10→[0.13,0.27]; 0.15→[0.15,0.30]; 0.20→[0.18,0.33]; 0.30→[0.23,0.40];
0.40→[0.27,0.44]; 0.50→[0.32,0.50]; 0.70→[0.42,0.58].

**Q5 — N_c too small at 128/160?** Yes, marginal for B_L (crossing sits in the
noisy small-B_L tail). More stable locators, in order: entanglement **S** /
S·(log L)⁻¹ (low variance, broad crossing), **CMI** (~30% tighter than B_L),
**Var(S)** (peaks at criticality), B_L last. Extract λ_c from CMI *and* Var(S)
*and* B_L and require agreement; do not trust B_L alone at L≥96.

**Q6 — outputs to store every run (future-proofing).** Your minimum
(L,ζ,λ,N_c,T,burn-in,S,σ_S,B_L,σ_B) **plus**: CMI components
(S_AB,S_BC,S_B,S_ABC), Var(S), Rényi-2/3, ESS time series, jump-count stats
(n_T mean+var), and — most important — **per-clone (trajectory-level) S samples**
(or full histogram / higher moments) so any cumulant/Binder variant is
recomputable post-hoc without rerunning. Also store clone genealogy / resampling
diagnostics.

**Q7 — pairwise vs global FSS.** Both; trust order **L-extrapolated pairwise
crossings > global FSS collapse > single-pair crossings.** Diagnose a collapse
hiding drift by: (a) χ²/dof ≫ 1; (b) leave-one-size-out refits — if λ_c(ζ) or ν
shifts, drift is unconverged; (c) compare the collapse's λ_c(ζ) to raw pairwise
crossings — a systematic offset means drift is absorbed into ν / the scaling
function.

**Slope grid (ζ∈{0.7,0.8,0.9}, L=192,256):** targets the ζ=1 slope (1/8 vs 1/4),
a *separate* question from φ; does not help the mid-ζ exponent and is noisy at
those sizes. Lower priority than a clean mid-ζ multi-L extrapolation.

**The decisive plot.** L→∞-extrapolated **r_c(ζ)** over [0.10,0.70] from
{16,24,32,48,64}(+128 if clean), with (i) a **free-exponent** power-law fit and
(ii) the diagnostic r_c/ζ vs ζ: → const (linear/log) vs diverging as ζ^{−1/2}
(√ζ). Outcomes: stable φ≈1/2 ⟹ √ζ; stable φ≈0.8–1 ⟹ linear/log; persistent
window-drift after extrapolation ⟹ marginal/log-corrected.
