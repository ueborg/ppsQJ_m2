# Claim timeline: the Cut B boundary and its exponents

Audit 2026-08-10, Stage 2. Format: `claim → evidence → correction → surviving statement`.

## 1. The amplitude and the parameterization

| date | statement | evidence | source |
|---|---|---|---|
| ≤2026-05-19 | A ≈ 0.5 assumed | — | pre-history |
| 2026-05-20 | **A = 0.96 ± 0.05**, φ = 0.502 ± 0.026 | 1/√L extrapolation of Binder crossings; table shows 1/L → 0.428, 1/L^0.7 → 0.463, 1/√L → 0.502, 1/L² → 0.385 | `archive/SESSION_2026_05_20.md` |
| 2026-05-22 | carried into NLSM framework as "consistent" | — | `archive/NLSM_FRAMEWORK.md:286` |
| 2026-05-27 | into M1 report body text | — | `Chapter3.tex:236` |
| 2026-06-03 | **λ_c is a bad fitting variable**; fit r_c | λ_c saturates at 1/2, compressing the fit; r_c = λ_c/(1−λ_c) does not saturate | `NUMERICS_STATUS_AND_PLAN.md` §1 |
| 2026-06-07 | **the 0.96 is the r_c prefactor, not λ_c** | dense χ²/dof 0.76 → A = 0.51; v2 → 0.53; r_c → 0.78–0.90 | HANDOFF cont.-1 #2 |
| 2026-06-07 | debiased (32,64,128) triple: **λ_c = 0.501√ζ**, φ = 0.523 ± 0.019, R² = 0.986 | N_c-ladder 1/N_c extrapolation | HANDOFF cont.-2 #1 |
| 2026-06-10 | correction written into the source doc | — | `Y_ZETA_DERIVATION.md:180` |
| 2026-07-22 | manuscript adopts 0.50√ζ | — | SciPost draft |
| 2026-08-10 | **reproduced: λ_c A = 0.494** | independent wide-pair crossings on `agg_caseB_combined.pkl` | this audit |
| current | memory asserts A = 0.96 for λ_c | — | **regression** |

**Surviving statement.** λ_c amplitude ≈ 0.49–0.51 `[V]`. The 0.96 was never a
λ_c amplitude in the corrected reading.

### The 1/L versus 1/√L choice

The 1/√L extrapolation was selected in `SESSION_2026_05_20.md` on the grounds
that it is "theory" motivated, i.e. justified by ν = 2. It also produced the
lowest χ²/dof (10.7 versus 17.2, 13.4, 28.2) — but all four are poor fits, which
is itself a signal that none of the four forms is right. Since ν = 2 is not
measured (`CB-NU-001`, confidence set [1.5, 3]), the justification is circular.
Surviving statement: `[X]` as a justified choice, `[O]` as a numerical preference.

## 2. λ_c versus r_c: resolved as an artifact, not a disagreement

Stage 1 preserved this as genuine disagreement #1. Stage 2 tested it directly
(`audit/.../scripts/lambda_vs_rc.py`), holding crossings, ζ range, and procedure
identical between the two parameterizations.

Since `d ln r / d ln λ = 1/(1−λ)`, over a finite window the fitted exponents must
satisfy `φ_r ≈ φ_λ · ⟨1/(1−λ_c)⟩`, with equality of exponents only asymptotically
as λ_c → 0.

| ζ window | n | φ(λ_c) | φ(r_c) | ratio | ⟨1/(1−λ_c)⟩ |
|---|---|---|---|---|---|
| [0.05, 0.85] | 15 | 0.495 | 0.681 | 1.374 | **1.392** |
| [0.05, 0.40] | 11 | 0.390 | 0.490 | 1.257 | **1.257** |
| [0.05, 0.20] | 7 | 0.255 | 0.304 | 1.194 | **1.199** |
| [0.05, 0.125] | 4 | 0.185 | 0.217 | 1.173 | **1.177** |
| [0.15, 0.85] | 11 | 0.630 | 0.910 | 1.444 | 1.471 |
| [0.25, 0.85] | 8 | 0.670 | 1.021 | 1.524 | 1.562 |

The ratio tracks the Jacobian to three decimals in every window.

**Surviving statement.** The λ_c-versus-r_c exponent gap is **entirely the
Jacobian of a nonlinear map over a range where λ_c is not small**. It carries no
physical information. `[V]`.

Consequences:
- HANDOFF's "λ_c-φ ≈ 0.5 is a saturation artifact and the physical r_c exponent
  is ~0.7–0.85" is half right. The saturation diagnosis is correct. The
  conclusion that r_c carries "the physical exponent" is not: over this window
  r_c is just as window-dependent, and neither is asymptotic.
- The far more serious finding is that **both exponents drift enormously with
  the ζ window**: φ(λ_c) runs 0.185 → 0.670 and φ(r_c) runs 0.217 → 1.021.
  As ζ → 0 both fall toward ≈0.19, which is the small-ζ finite-size floor that
  `NUMERICS_STATUS_AND_PLAN.md` §4 documents (r_c flat ≈0.13–0.18 for ζ ≤ 0.05).
  At large ζ both rise past 1 as λ_c saturates toward the Born value.
- By `NUMERICS_STATUS_AND_PLAN.md` §6's own decision rule, "φ that keeps drifting
  with the window even after L-extrapolation ⟹ marginal/log-corrected, no clean
  power". My fits are **not** L-extrapolated, so that rule is not yet triggered,
  but the drift is large enough that L-extrapolation is now the decisive test
  and it has not been done on the current data.

**There may be no ζ window in which the asymptotic exponent is visible**: small ζ
is floored by finite size, large ζ is saturated by the Born endpoint. `[O]`

## 3. φ: every estimate on record

| estimate | method | window | date | status |
|---|---|---|---|---|
| 0.502 ± 0.026 | 1/√L Binder extrapolation | small-ζ | 2026-05-20 | `[X]` |
| 0.56 ± 0.05 | global FSS collapse on λ_c | [0.03, 0.85] | ≤2026-06-03 | `[X]`, collapse cannot resolve exponents |
| 0.71–0.84 | free power on r_c, sliding | various | 2026-06-03 | `[P]`, now explained as Jacobian + drift |
| 0.523 ± 0.019 | debiased (32,64,128) on λ_c | [0.02, 0.5] | 2026-06-07 | `[P]` |
| 0.65–0.81 | free power on r_c | dense + v2 | 2026-06-07 | `[P]` |
| 0.5–0.6 | crossings, clean window | ζ ≥ 0.25 | 2026-06-17 | `[P]` |
| ~0.50 | fixed-ratio (64,128) bootstrapped CMI | complete data | 2026-08-05 | `[P]`, `CHAT` |
| 0.36–0.57 | spread across observables/pairs | — | 2026-08-05 | the honest error bar |
| **1** | x_J route, φ = 1/(3−2x_J) with x_J ≈ 1.04 | ζ → 0 corner | 2026-06-17 | `[O]`, **unrefuted** |
| 0.495 | this audit, wide pairs | [0.05, 0.85] | 2026-08-10 | audit-grade |

**Surviving statement.** φ is consistent with 1/2 over the mid-ζ window under
several methods, but the honest spread is ≈0.36–0.57 at fixed method-choice and
0.19–1.02 across ζ windows, and an independent analytic route points to 1. `[O]`.
The manuscript's hedged "consistent with 0.5√ζ over the accessible window,
locator-dependent drifts prevent a precise exponent" is the defensible statement.

## 4. ν

| estimate | basis | status |
|---|---|---|
| ν ≈ 2 | class-DIII literature, **Fulga et al. PRB 86 054505** | `[X]` as an input: that is the **n→0 forced/Anderson** exponent; Jian et al. prove n→0 and n→1 are distinct classes in DIII |
| ν ≈ 2 "plateau" | global collapse, ζ ∈ [0.05, 0.7] | `[X]`, collapse recovers 1.2–1.7 regardless of ν_true |
| 1.3 (B_L), 1.6 (CMI) | collapse | `[X]`, artifacts |
| 3.10 at ζ=0.02, 1.82 at ζ=0.03 | per-ζ FSS | `[X]`, noise |
| ν̂ ≈ 1.5, "consistent with 2" | LMR interpolation collapse, cross-size | `[P]` |
| **confidence set [1.5, 3]** | simulation-based construction marginalising over a correction-model family | `[V]` as a statement about resolving power |
| ν_A ≈ 0.9–1.09 | Cut A curvature + fixed-λ_c LMR | `[P]`, and only in the small-ζ regime where no crossover is expected |

**Surviving statement.** ν_B is **not measured**. The confidence set does not
shrink with n_real because correction-model uncertainty dominates. Pinning it
requires larger L. Every "ν ≈ 2" in memory, in HANDOFF's bottom half, and in the
derivation chain (`y_λ = 1/ν = 1/2`) is a consistency statement, not a measurement.

**Data adequacy.** T/L ≤ 1 everywhere against a stated requirement of T ≥ 2L.
That requirement is itself `[P]` (`NUMERICS-T-001`): it derives from the
2026-06-17 observation that production T was capped at 128 for L ≥ 96 plus the
argument that relaxation grows as L^z. The τ_int pilot that would establish the
actual T(L) rule is listed as owed since 2026-06-17 and has not been run. Do not
treat T ≥ 2L or n_real ≈ 25 as audit axioms.

## 5. y_ζ and the cross-vertex dimension

`claim` — Δ_ζ = 1, so y_ζ = 2 − Δ_ζ = 1.
`evidence` — measurement of a single bilinear or the raw correlator, both ~ r^{−2}.
`correction` (2026-06-03, `analysis/cross_vertex_dimension.py`, L = 600, pure-state
check 1.3e-14) — the raw operator splits as
`B₊B₋ = ⟨B⟩² + ⟨B⟩(δB₊+δB₋) + :B₊B₋:`. The dim-1 admixture is what made the raw
correlator look relevant. Measured: single-copy `C(r) ~ r^{−2.018}` ⟹ Δ_B = 1.01;
genuine vertex `C(r)² ~ r^{−4.036}` ⟹ **Δ = 2.02, marginal, y_ζ = 0**; raw
`~ r^{−2.015}`, "the most likely origin of the project's earlier Δ_ζ = 1 claim".
`surviving statement` — Δ_B ≈ 1.01 `[V]`; Δ_{:B₊B₋:} ≈ 2.02 marginal `[V]`;
y_ζ = 1 `[X]`.

**Still asserted by memory.** And the θ₁ inference chain still rests on Δ_ζ = 1
(`CHAT_ARCHAEOLOGY.md` §1).

## 6. ξ_ps, λ⁻¹ versus λ⁻² — a naming collision, partly a reformulation

| date | statement | source |
|---|---|---|
| 2026-06-03 | **two distinct scales**: band/dimerization ξ_dim ~ λ⁻², steady-state ξ_nc ~ λ⁻¹ with prefactor 4–5 (ξ·λ flat over λ ∈ [0.10, 0.40]). The entanglement-relevant one is λ⁻¹. The NLSM's λ⁻² is the band length. | `CURRENT_THEORY_STATUS.md` §7, `noclick_spectrum_probe.py` |
| 2026-06-10 | "ξ_ps ~ λ⁻² **REFUTED**"; the λ⁻¹ scale is renamed the **selection length** ℓ_λ = 4w/λ, "a formation scale, not a state ξ" | HANDOFF, from `anchor_scan.py` |
| 2026-06-15 | `anchor_scan.py` is **wrong** (its kernel drops the hopping w). Band-structure ξ_nc = 2/ln(1+κ²/w²) ~ λ⁻² **CONFIRMED**; area-law for all λ > 0, critical only at λ = 0 | HANDOFF, deterministic run |
| 2026-06-15 | `[O]` measured steady-state exponent over λ ∈ [0.2, 0.5] is **~λ^{−1.5}**, flatter than λ⁻²; called a crossover | HANDOFF |

**Unresolved.** Two independent measurements of the steady-state length disagree:
λ⁻¹ over [0.10, 0.40] (2026-06-03, spectrum probe, ξ·λ flat) versus λ^{−1.5} over
[0.2, 0.5] (2026-06-15). The 2026-06-10 reframing of λ⁻¹ as a "formation scale"
was asserted on the strength of `anchor_scan.py`, which was **falsified five days
later**, yet the reframing was retained. §7's evidence is a steady-state
correlation length from the actual spectrum, not a formation scale.

**This is a reformulation that has never been justified by a surviving
calculation.** `[O]`, and it matters: `CURRENT_THEORY_STATUS.md` §10 states that
with the corrected inputs (Δ = 2 marginal, ξ ~ λ⁻¹) the same crossover machinery
gives **φ = 1 (linear)**, not 1/2 — the same answer the independent x_J route
reached.

## 7. NLSM: survived, invalidated, or reformulated

| component | verdict |
|---|---|
| ζ enters only the cross-replica vertex; replica-diagonal part ζ-independent | **survived** `[V]`, exact and structural |
| Choi generator `L = −i(H₊−H₋) − (α/2)(P₊+P₋) + ζα P₊P₋` | **survived** `[V]` |
| operator split into single-copy mass ∝ α(ζ−1)/4 and cross vertex ∝ ζα/4 | **survived** `[V]` |
| Δ_ζ = 1 (relevant cross vertex) | **invalidated** `[X]` |
| ξ ~ λ⁻² as the matching scale | **invalidated** for the steady-state length `[X]`; survives as the band length |
| matching ζλ⁻² ~ K* ⟹ λ_c ~ √ζ | **invalidated**, both inputs failed `[X]` |
| SO(2n)/U(n) coset target | **reformulated, not verified**: the monitored-Majorana literature uses principal-chiral SO(N); HANDOFF says the target "must be rederived from the Choi action". Not done. `[O]` |
| φ = 1/2 as corner-matching at the ζ=0 anchor | **reformulated** `[P]`, and **conditional on the area-phase ξ gate, which has never been run** (`worker_areaphase_pps.py`, written 2026-06-10) |
| Case A Ising | **relocated** to the ζ=0 endpoint only `[V]`; Born line SU(2)₁ `[P]` |
| QSD-relevant versus QJ-marginal dichotomy | **retracted** `[X]` |

**The pattern.** Each time a derivation of √ζ failed, it was replaced by a new
derivation of √ζ rather than by re-opening the exponent. The empirical answer was
held fixed while the mechanism was rebuilt three times. Meanwhile two independent
routes (x_J ≈ 1.04; §10's corrected-input matching) both give **linear**.
That asymmetry deserves explicit attention.
