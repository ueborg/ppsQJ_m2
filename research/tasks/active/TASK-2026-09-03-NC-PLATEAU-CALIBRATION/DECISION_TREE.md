# DECISION_TREE — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

**Frozen before any HPC datum exists.** Every branch below was written while the
only numbers available were the ones already on disk. A tree written after the
results arrive is a rationalisation, and the phase lock exists to make the
difference checkable.

Labels `[E]` `[I]` `[C]` `[J]`. Verdict names are the ones
`SUCCESS_CRITERIA.yaml` defines and the analysis prints.

---

## 0. Entry condition

`[E]` The tree is entered only after the immediate group has returned and
`analysis/nc_plateau_analysis.py` has run once, unedited, on the complete set.
No branch may be taken on a partial array: a ladder missing its top rung reads
as `NO_STEP`, not as a plateau.

---

## 1. `L = 64` CENTRAL  (campaign A)

```
Delta_4096 = I_8192 - I_4096
```

**A-1 · `PLATEAU_OBSERVED`** — P1–P5 all pass at the top step.
→ Identify the SMALLEST `N_c` on the ladder whose own step satisfies P1 and P2.
→ That is the calibrated `N_c` for `L = 64` **at this lambda only**.
→ Attempt a LOCAL `B` (section 12 of the brief) **only if** the asymptotic-form
  test also passes; report it as cell-specific and refuse to generalise it.

**A-2 · `UNRESOLVED_R_LIMITED`** — P1 passes, P2 fails.
→ **Not a plateau.** Report the matched `R` the analysis computes for P2.
→ Do **not** add `N_c`: the binding budget is `R`, and more `N_c` cannot narrow
  an interval set by `R`.
→ Decide separately whether that `R` is worth buying, at ~6.9 h per population
  at `N_c = 8192`.

**A-3 · `STILL_DRIFTING`** — P1 fails at `4096 -> 8192`.
→ Report: **even `L = 64` remains pre-asymptotic**.
→ Do **not** extrapolate `I_inf`. Do **not** quote `B` or `gamma`.
→ Do **not** automatically queue `N_c = 16384`: that needs a fresh task and a
  fresh justification that is not "the last one did not work".
→ `[I]` The programme then has no `L` at which this estimator is known to
  converge, and every downstream absolute-CMI number stays uncalibrated.

**A-4 · P3 or P5 fails while P1 and P2 pass.**
→ `UNRESOLVED`. A plateau whose conclusion depends on its lowest rung, or whose
  successive `|Delta|` are growing, is a fit artefact and is reported as one.

---

## 2. `L = 64` CROSSING REGION  (campaigns B + B2) — **load-bearing**

Two questions, and they are answered in this order, because the second is the
one the programme actually needs.

### 2a. Shape

**B-1 · H1 or H2 not rejected, H3 not resolved.**
→ The correction is compatible with an additive constant or a multiplicative
  rescaling across the region. Record which, and that "not rejected" is not
  "confirmed".

**B-2 · both H1 and H2 rejected, H3 resolved.**
→ Resolved `lambda`-dependent shape distortion at that `N_c`.
→ Recommend a HIGHER `N_c` for transition-region production than the level test
  alone suggests, and say by how many doublings the distortion would have to
  shrink.

**B-3 · neither H1 nor H2 rejected AND H3 not resolved.**
→ `UNRESOLVED`. Seven points did not separate the hypotheses. This is a real
  outcome, pre-registered in `FALSIFICATION_PLAN.md` Y3, and it is **not**
  reported as support for either.

### 2b. Locator  ← the branch that decides production

```
lambda_x(N_c) from the fully matched cross-L difference, N_c = 512, 1024, 2048
```

**L-1 · the crossing displacement shrinks and `|lambda_x(2048) - lambda_x(1024)|
< tau_lambda`.**
→ **Define production `N_c` from the CROSSING tolerance, not the absolute
  level**, and state that as the basis in every downstream report.
→ This branch is available EVEN IF P2 fails on the absolute level at that `N_c`.
  A smooth but shifted curve is not unusable for locating a crossing when the
  shift cancels in the cross-`L` difference.
→ `[E]` The reverse substitution is never permitted: absolute-level convergence
  does not by itself certify a locator.

**L-2 · the displacement shrinks but is still outside `tau_lambda` at 2048.**
→ Extrapolate the *trend* only to state how many doublings would be needed, and
  cost them. Do not declare adequacy.

**L-3 · the displacement does not shrink.**
→ C3 is dead. The locator inherits the absolute level's problem, production
  `N_c` must be set by `tau_I`, and — `[E]` per `agent_reports/numerics.md` F9 —
  that is unaffordable at `L = 128`. `[I]` The programme would then need a
  variance-reduction or estimator change before `L = 128` production, not more
  `N_c`.

**L-0 · any crossing flagged `ENDPOINT_INDUCED`.**
→ Report as a boundary-sensitive locator, not as a located crossing, whatever
  else the branch says. `[E]` On the 7-point grid this should not occur for
  either interior crossing; if it does, the grid moved and the design changed.

**Slope re-derivation, unconditional.** `[E]` `tau_D = tau_lambda x 2.965` uses a
slope measured at `N_c = 1024`. The analysis recomputes it from the new curves.
If it has moved materially, `tau_D` and `tau_I` are re-derived BEFORE any
adequacy verdict, and the re-derivation is recorded as an amendment.

---

## 3. `L = 96`  (campaign C)

**C-1 · `Delta_1024` small and P1+P2 pass.**
→ Candidate production `N_c = 2048`, or 1024 if the full ladder's own smallest
  passing rung is 1024.
→ Release EXACTLY ONE of `cond_M96_nc1024` / `cond_M96_nc2048`. Never both:
  they are the same scan at two `N_c`.

**C-2 · `Delta_1024` small but P2 fails (expected at `R = 24`).**
→ `UNRESOLVED_R_LIMITED`. Report the required `R`.
→ `[J]` This is the most likely single outcome at `L = 96` and it is a real
  answer: it converts an open question into a costed one.
→ The M96 scans stay blocked. Decide between buying `R` and taking the locator
  route.

**C-3 · `Delta_1024` resolved and comparable to `Delta_256 = -0.0696`.**
→ `L = 96` is pre-asymptotic. Further calibration before any production there.
→ **Exclude `L = 96` from any cross-`L` comparison of `B`.** A pre-asymptotic
  ladder is excluded, never averaged in.

**C-4 · the `1/N` form.** `[E]` Whatever happens, record whether the four/five-rung
`L = 96` ladder rejects `1/N`, because the three-rung version does **not**
(`p = 0.168`) and the accepted framing says it does. Either the discrepancy
resolves or it is reported as an open provenance item.

---

## 4. `L = 128`  (campaign D)

**D-1 · `|Delta_1024|` comparable to `|Delta_512| = 0.0602`.**
→ Still materially drifting. **Recommend `cond_D2_L128_nc4096`**, and state in
  the same sentence that one population there is ~71 h (~100 h pessimistic)
  against `cpu_long`'s 168 h `MaxTime`, i.e. feasible but fragile and
  uncheckpointed.
→ `cond_M128_*` stay blocked.

**D-2 · `|Delta_1024|` clearly smaller than `|Delta_512|` but the interval is
wide (the design's expected case at `R = 16`).**
→ **Do NOT declare convergence.** Increase `R` before any plateau claim, or take
  the locator route.
→ `[E]` Do not run `N_c = 4096` on this branch either: it answers a question
  about `N_c` when the binding budget is `R`.

**D-3 · `|Delta_1024|` small AND confidently inside tolerance.**
→ `[C]` Not expected: at `R = 16` the half-width is ~0.026, about 4.3× `tau_I`,
  so this branch is close to unreachable by construction. If it is reached,
  something about the variance changed at `N_c = 2048` and that is itself the
  finding.
→ Do **not** automatically run `N_c = 4096`. Candidate production `N_c = 2048`,
  and `cond_M128_nc2048` becomes eligible.

**The `N_c = 4096` trigger is fixed NOW**, in
`conditional/cond_D2_L128_nc4096/README.md`, and may not be made to depend on
the observed 2048 value after the fact.

---

## 5. DISCRETISATION  (campaign E)

**E-a · results depend materially on `dtau_mult`** (a clean `1/dtau_mult`, `E1`).
→ Production control requires an explicit discretisation calibration in addition
  to an `N_c` calibration. `[I]` Every existing corpus number at `dtau_mult = 6`
  then carries an uncalibrated discretisation term.
→ `K` may be named as the causal variable — and only on this branch.

**E-b · `dtau_mult = 3` and 6 agree while 12 differs.**
→ A possible discretisation-stable regime with an onset between 6 and 12.
→ `[E]` This is the branch that would *retrospectively justify* the certified
  `dtau_mult = 6` against the corpus value 12, which is worth having in writing
  either way.

**E-c · all three agree within sensitivity** (`E2`).
→ Window frequency is **disfavoured** as the dominant explanation of finite-`N_c`
  drift. `K`-accumulation is killed.
→ `[I]` Attention moves to the accumulated-weight route — which is exactly why
  `logw_carry_var_final` and the per-window histories are recorded by this
  campaign and by nothing before it.

**E-d · anything in between.**
→ `INCONCLUSIVE`. Kills neither mechanism. Reported as such and **not** resolved
  toward whichever side is closer.

**Both `N_c` are read separately.** `[E]` The `K`-dependence may itself depend on
population size; if `N_c = 64` and `N_c = 256` disagree, that disagreement is the
result and neither is pooled into the other.

---

## 6. THE PRODUCTION DECISION — the only place these join

`[E]` **ROUGH PRODUCTION may be recommended only if all four hold:**

1. at least one `L` has a calibrated `N_c` — by branch A-1, or by branch L-1 on
   the locator tolerance with the basis stated;
2. campaign B has not returned B-2 (a resolved shape distortion at the `N_c`
   proposed for production);
3. campaign E has not returned E-a, or, if it has, a discretisation calibration
   is scheduled alongside;
4. the required `N_c` at each `L` intended for production is affordable —
   including the check that `L = 128` at `N_c = 4096` sits at the partition
   `MaxTime` ceiling and `N_c = 8192` at `L = 128` is not runnable at all.

`[E]` **If any fails, the answer is `NOT READY`**, with the specific missing
calibration named. `[J]` The task must not claim readiness before the
calibration is seen, and "three of four" is not readiness.

`[E]` **REFINED PRODUCTION** is not designed here and is not designed until
ROUGH has run: its `lambda` window comes from the observed rough crossing region
and from nothing else — not from `sqrt(zeta)`, not from `zeta^(1/3)`, not from
this campaign's 7-point locator window, which is a `L <= 64`, `N_c = 1024`
artefact until proven otherwise.
