# Chat archaeology

Audit 2026-08-10, Stage 2. Sources are prior conversations in this Claude
project. **Nothing here is promoted to `[V]` on chat evidence alone.**

Conversations located and used:
`bafee3bb` (2026-04-11, Doob transform), `fb3e2501` (2026-04-10, cloning
literature), `58cffe81` (2026-06-03, theory review), `193d0047` (2026-06-03,
FSS + θ₁ commit + **memory writes**), `e285f5ec` (2026-06-05, fabricated
citation), `601b6758` (2026-06-17, Cut B analytics, docs 5/6/7),
`c49698e2` (2026-07-21, SciPost draft), `ca2b054c` (2026-08-05, variance
reduction + Cut A + boundary campaign).

---

## 1. θ₁ / SCGF / parity doublet

**Chronology.**
`claim` — early: θ₁ ~ L, and separately K_eff ~ L, used as a route to y_ζ = 1.
`evidence` — a BdG-level quantity α Σ_j ⟨J_j† J_j⟩.
`refutation` — exact Liouville-space computation: `H_eff` commutes with total
fermion parity P, so |R₀⟩ and |L₀⟩ share parity p₀ and d_j is parity-odd, hence
⟨L₀|d_j|R₀⟩ = 0. **θ₁^SCGF = 0 identically.** The BdG quantity is the no-click
*activity*, not the SCGF derivative.
`replacement` — a 2×2 parity-doublet effective generator on slow modes r₊, r₋
with off-diagonal ζK₊₋, ζK₋₊. Reproduces the full L_ζ leading eigenvalue at
ζ = 1e-3 to 1e-8 at L = 4,5,6,7.
`surviving statement` — θ₁^SCGF = 0 by parity `[V]`; the 2×2 generator
`[V]` at L ≤ 7.

**Evidence class** `EXEC` + `CODE`. Committed: `analysis/compute_theta1_exact.py`,
`analysis/parity_resolved_theta.py`, `analysis/parity_resolved_data.pkl`,
`analysis/parity_resolved.png`, `theory/theta1_first_principles.md`.

**The load-bearing problem.** The conclusions drawn *from* the parity result are
conditioned on **Δ_ζ = 1**: "K_eff ~ L^{1−Δ} = L⁰ is the correct expectation for
a cross-Choi click vertex of dimension Δ_ζ = 1", and "y_ζ = 2 − Δ_ζ = 1 via the
CFT level-spacing argument". Both were written on **2026-06-03**, the same day
`CURRENT_THEORY_STATUS.md` established Δ_{:B₊B₋:} = 2.02 (marginal).

So: **the parity result survives the Δ ≈ 2 correction; the y_ζ = 1 conclusion
built on top of it does not.** K_eff ~ L^{1−Δ} with Δ = 2 predicts L^{−1}, not
L⁰. The observed κ ~ 0.35–0.5 at L = 5,7 was read as "consistent with the L⁰
floor plus edge corrections" and has never been re-read against Δ = 2.
Status of the y_ζ inference: `[X]` as stated, `[O]` pending re-derivation.
**This is a reformulation-versus-refutation case and it has never been closed.**

## 2. Cut A observable: CMI versus end-to-end MI

**Chronology.** `claim` — the stored Cut A order parameter (CMI tripartition) is
the right observable. `correction` (user, 2026-08-05) — the region assignment
should be ABDC/end-to-end, not the peaked CMI.

**What the executed check actually found** (chat `ca2b054c`, run on the Mac
against the real `pps_qj` backend):

The Cut A trajectory state is **exactly pure**, S(total) = 0.0 measured. For a
pure state S_ABC = S_D exactly, so
`CMI = S_AB + S_BC − S_B − S_ABC`, the `ABDC` form
`S_AB + S_BC − S_B − S_D`, and `gaussian_backend.topological_entanglement_entropy`
are **the same number**. All three returned 0.0563 on a real covariance.

**Therefore the ABDC-versus-ABCD labelling is not the issue.** What changes the
physics is the **spatial placement**: reading ABDC as spatial order, A is the
first quarter and C the last, putting them at opposite ends of the chain.

Measured `I(A_left : C_right)` versus stored CMI:

| λ | stored CMI | MI(ends) |
|---|---|---|
| 0.15–0.35 | 0.000 | 0.000 |
| 0.45 | 0.038 | 0.002 |
| 0.50 | 0.420 | 0.177 |
| 0.55 | 0.525 | 0.447 |
| 0.65 | 0.001 | **0.994** |
| 0.75–0.85 | 0.000 | **1.000** |

The stored CMI is peaked and returns to zero on both sides. End-to-end MI
saturates at exactly 1 bit. `[V]` as a statement about these observables at
these λ.

**Gaps.** The L-dependence check (does the step sharpen toward λ = 1/2 with L?)
**timed out and was never completed**. So the claim that MI(ends) is an *order
parameter for the transition* rests on a single L. The numerical output above
was not written to disk.

**Code status.** The observable **was implemented**:
`pps_qj/parallel/worker_caseA.py` is modified (+16 lines, uncommitted) adding
`_batched_compute_MI_ends` and emitting `MI_ends_q4_mean/err`, `MI_ends_q8_mean/err`.
Whether any production run has used it is not established.

**Memory's rendering is imprecise.** Memory says "region assignment must be ABDC
(end-to-end), not peaked CMI", which reads as a labelling fix. The actual finding
is that labelling is a no-op and geometry is everything.

## 3. The 2026-08-05 boundary campaign

Chat `ca2b054c` summary records a **completed boundary map campaign of 5880
realizations across L = 64–128, 14 ζ values, 7 λ multipliers**, plus dense
focused-window refinement, a Cut A guided campaign, a χ₂ two-click response
estimator, and small-ζ large-L asymptotics.

**On-disk corroboration found** (`EXEC` + `DATA`, not chat-only):
- `results/boundary_aggregate.csv` — 470 rows, 5634 realizations, L ∈ {64, 80,
  96, 112, 128}, ζ ∈ 14 values including **1.0**, λ 0.067–0.800, nreal 6 or 12,
  fields CMI/CMI_se/B_L/B_L_se.
- `results/ruche_pull/pps/boundary/` and `.../smoke/` — 6725 files, 2026-07-08 to 07-23.
- `results/ruche_pull/refine_smallz/` — 3675 files, 2026-07-27 to 07-29.
- `results/ruche_pull/caseA_guided/` — 1394 files, 2026-07-26 to 07-27.

**None of this appears in HANDOFF.** HANDOFF's most recent data discussion is
the 2026-06-17 rebuild plus the 2026-07-07 Ruche migration. The campaign that
the migration was built to enable was then run, pulled back, and aggregated,
and left unrecorded.

Findings claimed in that session, evidence class `CHAT` with on-disk data
available to re-derive but **no preserved analysis output located**:
1. φ ≈ 0.50 from the fixed-ratio (64,128) bootstrapped CMI pair on complete
   data; linear excluded, χ²/dof ≈ 1 versus 5–13. Observable-dependent spread
   0.36–0.57 across pairs.
2. The MIPT is confirmed **independently of crossings** via entropy scaling:
   S_{L/2} versus ln L shows a log-law to area-law transition. This is a
   fit-free confirmation and is arguably the most robust single result in the
   project. It is recorded nowhere on disk.
3. χ₂ u/c robustness scan gave p = 0.46–1.40 across slices, so **x_J = 1 is not
   confirmed** and that route is inconclusive.
4. Small-ζ batch windows were **mis-centred** (multipliers 0.85–1.45 placed the
   whole window above λ_c), so no asymptotic crossings were extracted. Rerun
   required with multipliers 0.45–1.05.
5. Cut A self-duality validated at ζ ≤ 0.25 (crossings 0.49–0.51); drift to
   ~0.42 at higher ζ, most likely a partial-data artifact in incomplete
   L = 64/96 cells.

## 4. The x_J route (2026-06-17, chat `601b6758`)

Three AI-generated theory memos, evaluated with container-side numerics.

`claim` (doc 5) — the whole linear-versus-√ζ question reduces to one measurable,
the scaling dimension x_J of a single click insertion at the no-click endpoint,
via φ = 1/(3 − 2x_J). Its own §9 bulk argument concludes **x_J = 1, i.e. linear**.
`evidence` — explicit §9 click-update test on the exact critical L = 800 state:
perturbation decays as r^{−2.07}, **x_J ≈ 1.04**.
`obstruction` — the Level 1 test (one-click CMI susceptibility χ_ζ(L) ~ L^p)
failed twice, non-monotonic across L = 12–28 (5.04, 3.79, 2.25, 2.40, 2.53).
`correction` (doc 6) — the failure is structural, not finite-size: the universal
first-order CMI response is **identically zero** because the energy operator's
one-point function on the uniformized replica cover vanishes for a primary with
⟨ε⟩_C = 0. Leading universal term is O(ζ²), not O(ζ).
`consequence` — validated: ∂_g²CMI ~ L^{1.975} ≈ L² at L = 32–256, so
∂_ζ²I = (u²/L)∂_g²CMI ~ L¹, the x_J = 1 linear-boundary signature.
`later` — the 2026-08-05 χ₂ scan gave p = 0.46–1.40, so x_J = 1 is **not
confirmed**.

**This is a live threat to √ζ that is absent from HANDOFF, memory, and the
manuscript.** An independent route concluded *linear* with a direct numerical
measurement (x_J ≈ 1.04, r^{−2.07}), and it has neither been refuted nor
integrated. Status `[O]`, and it is the strongest single argument against the
headline result.

## 5. Where memory came from

Chat `193d0047` (2026-06-03) shows memory being written by `memory_user_edits`
with, among others, "Empirically: Binder FSS supports y_zeta~1. Born-rule
endpoint lambda_c(1)~0.5 matches Carollo. Critical condition: zeta*xi_ps ~ 1",
and a pending-actions entry ending "(4) Verify Delta_zeta=1 directly via
cross-Choi two-point function".

This dates the memory snapshot precisely to **2026-06-03** and explains its
character: it is a faithful record of that day, including the Δ_ζ = 1 assumption
and the Carollo attribution, both corrected within days and never re-written.
The `A = 0.96` entry is older still, from `SESSION_2026_05_20.md`.

Memory has not been substantively updated in **~10 weeks**, across the entire
Ruche migration, the July boundary campaign, and the August variance-reduction
programme.
