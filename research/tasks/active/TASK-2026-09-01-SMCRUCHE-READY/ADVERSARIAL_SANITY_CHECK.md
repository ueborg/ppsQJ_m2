# ADVERSARIAL_SANITY_CHECK — a cheap independent pass over SMCCERT

TASK-2026-09-01-SMCRUCHE-READY §6. **Analysis only. No new numerical campaign.**
Attacks run against the raw `N_c` ladders in
`TASK-2026-08-31-SMCCERT/scratch/` and `TASK-2026-08-30-SMCSTAT/scratch/`, not
against the memos. No frozen SMCCERT file was modified.

Labels: `[E]` · `[I]` · `[C]` · `[J]`

---

## Verdicts

| # | conclusion | **verdict** |
|---|---|---|
| 1 | finite-`N_c` CMI bias replicated on independent seeds | **WEAKEN** |
| 2 | the bias depends on `L` and proximity to criticality | **WEAKEN** (splits: criticality survives and strengthens; `L` does not separate) |
| 3 | VIF does not predict the bias | **SURVIVES** (one of its two pairs weakens) |
| 4 | budget should raise `N_c` before `R` in biased cells | **WEAKEN** (holds, but the headline factors do not) |
| 5 | `N_c` = 128 **and** 512 are inadequate at the calibrated hard cell | **WEAKEN** (128 robust; **512 not established**) |
| 6 | `N_c` ≈ 2724, `R` = 24 is scoped to that calibrated cell | **SURVIVES** |
| 7 | the simple `1/N_c` bias law is NOT established | **SURVIVES** |
| 8 | high-VIF variance scaling remains unresolved | **SURVIVES** |

**No KILL. Nothing here invalidates the Ruche arms; two findings strengthen the
case for ARM 1.** One finding has a concrete production consequence (§1).

---

## 1. Bias replicated on independent seeds → **WEAKEN**

`[E]` **The attack:** SMCCERT reports that `A-HV` and `S-REP` — the *same* cell,
disjoint seed blocks — have CIs "overlapping only in [6.77, 7.77]" and calls the
replication "consistent but only just". That is a comparison of two intervals.
The right test is a bootstrap on the **difference**.

`[E]` **Result**, matched 4-rung window, 4000 resamples:

```
B(A-HV)  = +8.899        within-block SE 1.113   (R=48)
B(S-REP) = +5.629        within-block SE 1.024   (R=32)
difference = +3.262   95% CI [+0.302, +6.254]   two-sided p ~ 0.031
```

`[E]` **The CI on the difference EXCLUDES ZERO.** The two disjoint seed blocks at
one identical cell **formally disagree** on the coefficient.

`[J]` **What this does and does not touch.** It does not touch existence: both
estimates are large, positive, and exclude zero by wide margins (z = +7.97 and
+6.89). It destroys the *magnitude* being reproducible. So "the bias replicated"
is true of the **phenomenon** and false of the **coefficient**, and SMCCERT's
"consistent but only just" is one degree too generous.

### The production consequence, quantified

`[E]` The shipped calibration treats the pooled `R` = 80 bootstrap as the whole
uncertainty. A method-of-moments random-effects widening for the observed
between-block dispersion gives:

| | `B` | 95% CI | upper end | planner `N_c` floor at `bias_tol` = 0.005 |
|---|---:|---|---:|---:|
| as shipped (pooled bootstrap) | +7.591 | [+6.035, +9.170] | 9.170 | **1835** |
| widened for between-block τ = 2.05 | +7.591 | [+3.285, +11.897] | 11.897 | **2380** (+30%) |

`[I]` The planner already uses the **conservative end** of `B`'s CI, so this is
partially absorbed — but the CI itself is too narrow, and the `L` = 64 `N_c`
floor is understated by roughly **30%**.

`[J]` This moves the recommendation in the **safe** direction (a larger `N_c`),
so it is a caveat rather than a blocker, and it does not change either Ruche arm.
It should be recorded before the calibration is ever promoted to canonical state.

## 2. Bias depends on `L` and criticality → **WEAKEN**, and it splits

`[E]` **The attack:** at fixed `T`, `n_steps ∝ λ·(L−1)`, so a contrast in `L` is
also a contrast in the number of resampling events. Measured `n_steps` at the
four corners:

| | λ = 0.2793 | λ = 0.35 |
|---|---:|---:|
| `L` = 64 | 188 | 236 |
| `L` = 32 | 93 | 116 |

`[E]` The **Δ_L** contrast changes `n_steps` by **2.02×** (near-critical) and
**2.03×** (off-critical).
`[E]` The **Δ_λ** contrast changes `n_steps` by **0.80×** in both rows.

`[J]` **The two halves are in completely different evidential positions, and
SMCCERT presents them as symmetric.**

- `[I]` **Δ_L is fully confounded.** A 2× rise in `L` is a 2× rise in `n_steps`,
  and nothing in the 2×2 separates "bias grows with system size" from "bias grows
  with the number of resampling events". → the `L` half is **not established**.
- `[I]` **Δ_λ survives, and is strengthened by this analysis.** It is nearly
  `n_steps`-matched (0.80×), and it moves `n_steps` **downward** while the bias
  moves **up** — the opposite direction to any "more windows → more bias" story.
  So the criticality effect cannot be an `n_steps` artifact.

`[E]` SMCCERT does record the confound (`REDTEAM.yaml` C2/A9, `CANDIDATES.md`),
but as a symmetric caveat on both effects. `[J]` It is not symmetric: it is fatal
to one and exculpatory for the other. The claim should read **"the bias grows
with proximity to criticality, and with `L` and/or the number of resampling
windows, which this design cannot separate."**

`[C]` Cheap follow-up that would separate them, not run here: one cell at
`L` = 64 with `T` = 16 (halving `n_steps` at fixed `L`). No new physics.

## 3. VIF does not predict the bias → **SURVIVES**

`[E]` **The attack:** the "overlapping VIF" claim compares VIF **point ranges**
without propagating their bootstrap CIs. Inflated envelopes:

| cell | VIF point range | CI-inflated envelope | `B` |
|---|---|---|---:|
| `S-CRIT32` | 2.35 – 5.86 | **1.43 – 8.87** | +2.324 |
| `A-MV` | 3.32 – 4.09 | **2.10 – 6.81** | +0.096 |
| `S-OFF64` | 9.90 – 24.16 | **4.82 – 33.69** | +2.462 |

`[E]` **The core pair survives intact.** `S-CRIT32` and `A-MV` overlap even more
strongly once inflated (1.43–8.87 against 2.10–6.81) and differ in `B` by a
factor of **24**, with only λ different. That alone establishes VIF is not
sufficient.

`[E]` **The second pair weakens.** `S-CRIT32` vs `S-OFF64` was cited as "~4×
apart in VIF with indistinguishable `B`". Their inflated envelopes now **overlap**
(1.43–8.87 against 4.82–33.69), so the 4× separation is not established and that
sentence should be dropped.

`[J]` The conclusion stands on one pair rather than two. Since the conclusion is
what makes the planner **refuse**, and refusing is the conservative action, a
one-pair basis is acceptable.

## 4. Budget: `N_c` before `R` → **WEAKEN**

`[E]` **The attack:** `PRODUCTION_ACCEPTANCE_RULES.md` Gate 2 sets a coverage
floor of `R_min` = 12 (24 at intermediate ζ with `L` ≥ 96). Four of the winning
equal-budget allocations sit **below it**:

```
L=64  M=1536  (R= 6, N_c=256)   <-- below R_min=12
L=64  M=1536  (R= 3, N_c=512)   <-- below R_min=12
L=64  M=3072  (R= 6, N_c=512)   <-- below R_min=12
L=96  M=2048  (R= 8, N_c=256)   <-- below R_min=12
```

`[J]` So the headline gains — 18×, 21×, 19× — are partly earned by allocations
the task's own acceptance rules declare **inadmissible**. An allocation whose
interval nobody can honestly compute is not a usable configuration.

`[E]` **Restricted to admissible `R` ≥ 12, the ordering still holds**:
at `M` = 1536, (24, 64) = 1.76e-2 against (12, 128) = 5.14e-3 — a **3.4×** gain;
at `M` = 3072, (24, 128) = 4.87e-3 against (12, 256) = 7.83e-4 — a **6.2×** gain.

`[I]` The **rule survives**; the **advertised magnitude does not**. The honest
figure is 3–6× within the admissible set, not 18–21×.

## 5. `N_c` = 128 and 512 both inadequate at the hard cell → **WEAKEN**, splits

`[E]` **The attack:** "inadequate" is measured relative to `I_∞`, which is a
**fitted intercept**, not an observed quantity. At the hard cell
(`A-P96`+`A-BUD`), refitting the same ladder under the two forms the frozen spec
admits:

```
1/N_c form :  I_inf = 0.28131,  B = +11.281,  bias@512 = +0.02203
free-beta  :  I_inf = 0.25801,  beta = 0.860
I_inf spread between admissible forms = 0.02330  (8.3% of the 1/N_c value)
```

- `[E]` **`N_c` = 128 is robust.** bias@128 = **+0.0881**, which is **3.8×** the
  0.0233 form-uncertainty. The claim survives comfortably.
- `[E]` **`N_c` = 512 is NOT established.** bias@512 = **+0.0220**, which is
  **smaller than the 0.0233 uncertainty in `I_∞` itself**. `[J]` The statement
  "`N_c` = 512 is still 3.2× the SEM" quotes a bias whose own systematic
  uncertainty exceeds it.

`[J]` **This strengthens the case for ARM 1**, which adds a 512 rung at exactly
this cell and would settle it.

## 6. `N_c` ≈ 2724, `R` = 24 is scoped → **SURVIVES**

`[E]` Executed against the merged planner:

```
L=128 production        -> CALIBRATION_REQUIRED
L=96  calibrated cell   -> ok, N_c=2724 R=24
L=96  small zeta        -> CALIBRATION_REQUIRED
L=64  same cell, T=L    -> CALIBRATION_REQUIRED
```

`[J]` It refuses even for the *same* `L` and ζ at a different `T`. The scoping is
real and enforced in code, not just asserted in prose.

## 7. The `1/N_c` law is NOT established → **SURVIVES**

`[E]` G2 **KILLED** on the pre-registered block (β = 0.290, CI [0.200, 0.872],
robust to widening the profile grid to [0.02, 4.0]); **INCONCLUSIVE** pooled at
`R` = 80 (β = 0.760, CI [0.390, 1.170], containing 1 but not contained in
[0.5, 1.5]). `[J]` "Not established" is the correct and conservative reading of
a verdict pair that is KILLED-then-not-sustained. `[J]` I attacked this from the
other side too — is the *denial* too strong? — and it is not: no window at any
cell produces a β CI contained in [0.5, 1.5] **and** containing 1.

## 8. High-VIF variance scaling unresolved → **SURVIVES**

`[E]` `S-HV96`, median VIF 84.8: γ = +0.630, CI **[+0.389, +0.856]**. The frozen
rule returns INCONCLUSIVE because the CI is not contained in [0.5, 1.5].

`[J]` The attack here is that "unresolved" is *too neutral*. The full-window CI
**excludes 1**, five of six windows have γ below 0.8, and the pre-registered KILL
was missed by **0.003** at a single CI endpoint. `[I]` The evidence **leans
against** `Var ∝ 1/N_c` at high VIF; it is not balanced. `[E]` SMCCERT's
`HIGH_VIF_SCALING.md` §6 does say "the sharpest attempt leans against it", so the
detailed document is right and only the one-word summary is too neutral.

`[J]` This is the second finding that **strengthens** the case for ARM 1.

---

## What a reader should take away

`[J]` Nothing is killed, and the four WEAKENs are all of one kind: **SMCCERT's
detailed documents are more careful than its summaries.** The confound is
recorded but presented symmetrically when it is not; the inadmissible `R` values
are visible in the tables but not flagged; the `I_∞` form-dependence is discussed
at one cell but not carried to the `N_c` = 512 claim; and the seed-block
comparison was made between intervals rather than on the difference.

`[J]` Two of the four are **directly addressed by ARM 1** (the 512 rung and the
high-VIF γ), which is an argument for running it rather than against.

`[E]` One item should be carried forward before any promotion to canonical
state: **the calibrated `B` intervals are too narrow**, and the `L` = 64 `N_c`
floor is understated by ≈30% (§1).
