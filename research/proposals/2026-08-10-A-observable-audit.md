# PROPOSAL 2026-08-10-A — Observable definition audit: split OBS-BL-001

Status: **PROPOSED. Not applied.** Canonical state unchanged.
Charter: Stage 7, §10 item 1 (define all objects before use).

---

## 1. What the code actually computes

Traced from `pps_qj/parallel/worker_clone_pps.py` through the observable
routines to the stored fields. No definition below is inferred from a variable
name.

### Primary path — `_batched_compute_B_L`, worker_clone_pps.py:214

Tripartition on **Majorana** indices of the covariance `Γ` (2L modes total):

```
A   = [0,     L/2)        B    = [L/2,  L)         C = [L,  3L/2)
A∪B = [0,     L)          B∪C  = [L/2,  3L/2)      A∪B∪C = [0, 3L/2)
```

In **site** terms these are the four contiguous quarters of the chain. `A∪B∪C`
is three quarters, so the final quarter `D = [3L/2, 2L)` is the complement. The
global trajectory state is pure, so `S_ABC = S_D`.

```
S_X   = Σ_k h((1+ν_k)/2),  ν_k symplectic eigenvalues of the restricted Γ,  log base 2
CMI   = S_AB + S_BC − S_B − S_ABC          # genuine I(A:C|B) on quarters
B_L   = CMI × S_AB                          # PER CLONE
```

`S_AB` is the entropy of Majorana modes `[0, L)`, i.e. the **first half of the
chain**. So `S_AB` is the half-chain entropy at the final time.

Averaging (worker_clone_pps.py:524–541): `B_L` and `CMI` are formed **per clone
at t = T**, then `mean` is taken over the clone population per realisation, then
`_nanstat` over realisations.

**Therefore the stored `B_L_mean` is ⟨CMI × S_half⟩, a product formed BEFORE
averaging.**

Guard: `can_compute_B_L = (L % 4 == 0)`. B_L and all CMI components are NaN for
L not divisible by 4.

### Fallback path — `observables/topological.py:132`

Fires only on `numpy.linalg.LinAlgError` in the batched path.

```
S_half = subsystem_entropy(Γ, sites 1..L/2)
S_top  = topological_entropy(Γ, L)          # S_AB + S_BD − S_B − S_ABD on quarters
B_L    = S_top × S_half
```

Same functional form and the same regions, computed by a different code route.
The code deliberately NaNs the component entropies on this path so the two are
not silently mixed. **Not yet verified numerically to agree.**

### `S_mean` is a DIFFERENT quantity from `S_AB_mean`

`S_mean` comes from `cloning.py`: `_batched_entanglement_entropy(covs, ell)` on
Majorana modes `[0, 2ℓ)`, **time-averaged over the recorded history after
burn-in** (`cloning.py:547`). `S_AB_mean` is the same region **at final time
only**. They are not interchangeable and must not be substituted for one
another.

### Campaign coverage — one definition, verified

- `scripts/run_local_boundary.py:69` imports `_batched_compute_B_L` from the
  Case B worker and calls it directly. The **July/Ruche campaign uses the
  identical definition.**
- `pps_qj/parallel/worker_caseA.py:41` imports the same function. **Cut A uses
  the identical definition.**

So all guided campaigns share one B_L definition. Good news, and it means the
problem is naming, not divergence.

## 2. The load-bearing finding: B_L is not what KMR call B_L

`SRC-KMR-2023`, inspected 2026-08-10, Contents p.2, Appendix B.1:

> "Discussion of **B_L ≡ S̄top_L × S̄_L** for the particle number conserving limit α = 0"

The overbars are trajectory averages. **KMR's B_L is a product of averages,
⟨S_top⟩ × ⟨S⟩. Our stored B_L is an average of products, ⟨S_top × S⟩.**

These differ by the trajectory covariance `Cov(CMI, S_half)`, which is not
small: the frozen HANDOFF records the product-of-averages form as
"~30% tighter" and calls the trajectory product "noisy".

**One label, two mathematical quantities.** Charter §10 item 1 requires they be
separated.

Additional facts established, and NOT assumed:
- **B_L is not a Binder cumulant.** No fourth moment, no cumulant ratio appears
  anywhere. The source docstring itself calls it a "Binder-like proxy". Every
  document calling it "the Binder cumulant" is wrong. The **crossing method** is
  Binder-like; the **quantity** is not.
- Our `CMI` is a genuine conditional mutual information, correctly implemented.

## 3. Proposed OBS-ID split

| new ID | definition | stored? |
|---|---|---|
| `OBS-BLPROD-001` | ⟨CMI × S_half⟩, product per clone then averaged. **What the code computes and what every dataset stores.** | yes, `B_L_mean` |
| `OBS-BLKMR-001` | ⟨CMI⟩ × ⟨S_half⟩, product of averages. **KMR's B_L.** | no — recomputable post-hoc from `CMI_mean` and `S_AB_mean` |
| `OBS-CMI-001` | I(A:C\|B) on contiguous quarters, final time. Definition now **verified**. | yes, `CMI_mean` |
| `OBS-SHALF-001` | half-chain entropy. **Split needed**: `S_AB_mean` (final time) vs `S_mean` (time-averaged post-burn-in). Propose `OBS-SHALF-FINAL-001` and `OBS-SHALF-TAVG-001`. | both |
| `OBS-BL-001` | **retire**, `superseded_by: [OBS-BLPROD-001, OBS-BLKMR-001]`, retained as a historical alias | — |

## 4. Dependency consequences — exact state diffs

Nine claims reference `OBS-BL-001`. Proposed edits:

| claim | change | consequence |
|---|---|---|
| `CB-AMP-001` | `observable_id: OBS-BL-001 → OBS-BLPROD-001` | **Not ill-posed.** The audit reproduction used stored `B_L_mean`, which is unambiguously OBS-BLPROD-001. The value stands; only the label changes. Add to `assumptions`: "crossings located with the trajectory-product locator, not KMR's product-of-averages." |
| `CB-PHI-HALF-001` | same relabel | not ill-posed; same reasoning |
| `CB-WINDOW-001` | same relabel | not ill-posed |
| `CB-XI-LAMBDA{1,15,2}-001` | remove `observable_id` (they concern ξ, not B_L) | no change in meaning |
| `VR-CLOSE-001`, `VR-SNAPSHOT-001`, `VR-SNAPSHOT-NULL-001`, `VR-SNAPSHOT-PROD-001` | relabel to `OBS-BLPROD-001` | not ill-posed; variance-reduction work used the stored field throughout |
| `CASEA-DUAL-001`, `CASEA-DRIFT-001` | relabel to `OBS-BLPROD-001` | not ill-posed |

**No claim becomes ill-posed.** Every affected claim was computed from the same
stored field, and that field has one unambiguous definition. The defect was in
the documentation of the observable, not in the numbers.

**One claim gains a new caveat.** `CB-AMP-001` compares our amplitude against
KMR. `03_AMPLITUDE_TRACE.md` already flags a **parameterization** mismatch
(λ_c versus r_c). This audit adds a second, independent mismatch: an
**observable** mismatch, ⟨CMI×S⟩ versus ⟨CMI⟩⟨S⟩. Any KMR comparison must
reconcile both. Recorded as a new `assumptions` entry, not as a status change.

**Follow-on task, not part of this proposal:** recompute crossings using
`OBS-BLKMR-001` from the stored `CMI_mean` and `S_AB_mean` and compare against
`OBS-BLPROD-001`. That is a T0 read-only analysis and it directly tests whether
the amplitude is locator-dependent.

## 5. Unresolved

- The two code paths for B_L have **not** been shown to agree numerically. A
  paired test on one covariance would settle it. Until then, `OBS-BLPROD-001`
  should carry `definition_verified: primary_path_only`.
- `topological_entropy(Γ, L)` region indexing was read but not diffed
  line-by-line against the batched path.
- Whether KMR's reported `A ~ 0.5` is stated for `r_c` and for which observable
  was not established. The abstract does not contain it; it is in the body,
  which was not read.
